# tests/test_whisper_recognition_services.py

import importlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.domain.errors import ConfigurationError, RecognitionError
from src.infrastructure.groq_recognition_service import GroqRecognitionService
from src.infrastructure.openai_recognition_service import OpenAIRecognitionService
from src.infrastructure.whisper_api_support import to_recognition_error, to_whisper_language

SAMPLE_RATE = 16000

# OpenAI・Groqは同じ形のAPIなので、共通の振る舞いを両方のクラスで確かめる
SERVICES = [
    pytest.param(OpenAIRecognitionService, "whisper-1", id="openai"),
    pytest.param(GroqRecognitionService, "whisper-large-v3-turbo", id="groq"),
]


class FakeStatusError(Exception):
    """SDKのAPIStatusErrorと同じくstatus_codeを持つ例外"""

    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.fixture
def audio():
    return np.full(1600, 0.5, dtype=np.float32)


@pytest.fixture
def client():
    client = MagicMock()
    client.audio.transcriptions.create.return_value = MagicMock(text="こんにちは")
    return client


def request_of(client):
    return client.audio.transcriptions.create.call_args.kwargs


@pytest.mark.parametrize("service_class, default_model", SERVICES)
def test正常系_transcribeメソッドがAPIの認識結果を返すこと(service_class, default_model, client, audio):
    assert service_class(client=client).transcribe(audio, SAMPLE_RATE) == "こんにちは"


@pytest.mark.parametrize("service_class, default_model", SERVICES)
def test正常系_transcribeメソッドが従来の既定モデルを使うこと(service_class, default_model, client, audio):
    service_class(client=client).transcribe(audio, SAMPLE_RATE)
    assert request_of(client)["model"] == default_model


@pytest.mark.parametrize("service_class, default_model", SERVICES)
def test正常系_transcribeメソッドが指定されたモデルを使うこと(service_class, default_model, client, audio):
    service_class(client=client, model="custom-model").transcribe(audio, SAMPLE_RATE)
    assert request_of(client)["model"] == "custom-model"


@pytest.mark.parametrize("service_class, default_model", SERVICES)
def test正常系_transcribeメソッドが言語未指定のとき言語を送らないこと(service_class, default_model, client, audio):
    service_class(client=client).transcribe(audio, SAMPLE_RATE)
    assert "language" not in request_of(client)


@pytest.mark.parametrize("service_class, default_model", SERVICES)
def test正常系_transcribeメソッドが言語タグをISO639_1コードにして送ること(service_class, default_model, client, audio):
    service_class(client=client, language="ja-JP").transcribe(audio, SAMPLE_RATE)
    assert request_of(client)["language"] == "ja"


@pytest.mark.parametrize("service_class, default_model", SERVICES)
def test正常系_transcribeメソッドがWAV形式の音声を送ること(service_class, default_model, client, audio):
    service_class(client=client).transcribe(audio, SAMPLE_RATE)
    assert request_of(client)["file"][1][:4] == b"RIFF"


@pytest.mark.parametrize("service_class, default_model", SERVICES)
def test正常系_transcribeメソッドがwav拡張子のファイル名を付けること(service_class, default_model, client, audio):
    service_class(client=client).transcribe(audio, SAMPLE_RATE)
    assert request_of(client)["file"][0] == "audio.wav"


@pytest.mark.parametrize("service_class, default_model", SERVICES)
def test正常系_transcribeメソッドが空の音声に対して空文字を返すこと(service_class, default_model, client):
    assert service_class(client=client).transcribe(np.array([], dtype=np.float32), SAMPLE_RATE) == ""


@pytest.mark.parametrize("service_class, default_model", SERVICES)
def test正常系_transcribeメソッドが空の音声ではAPIを呼ばないこと(service_class, default_model, client):
    service_class(client=client).transcribe(np.array([], dtype=np.float32), SAMPLE_RATE)
    client.audio.transcriptions.create.assert_not_called()


@pytest.mark.parametrize("service_class, default_model", SERVICES)
def test正常系_transcribeメソッドがテキストの無い応答に対して空文字を返すこと(
    service_class, default_model, client, audio
):
    client.audio.transcriptions.create.return_value = MagicMock(text=None)
    assert service_class(client=client).transcribe(audio, SAMPLE_RATE) == ""


@pytest.mark.parametrize("service_class, default_model", SERVICES)
def test異常系_transcribeメソッドがAPIの失敗でRecognitionErrorを送出すること(
    service_class, default_model, client, audio
):
    client.audio.transcriptions.create.side_effect = FakeStatusError(500)
    with pytest.raises(RecognitionError):
        service_class(client=client).transcribe(audio, SAMPLE_RATE)


@pytest.mark.parametrize("service_class, default_model", SERVICES)
def test異常系_transcribeメソッドが認証失敗を継続不能なRecognitionErrorとして送出すること(
    service_class, default_model, client, audio
):
    client.audio.transcriptions.create.side_effect = FakeStatusError(401)
    with pytest.raises(RecognitionError) as error:
        service_class(client=client).transcribe(audio, SAMPLE_RATE)
    assert error.value.fatal is True


@pytest.mark.parametrize("service_class, default_model", SERVICES)
def test異常系_transcribeメソッドがサーバーエラーを継続可能なRecognitionErrorとして送出すること(
    service_class, default_model, client, audio
):
    client.audio.transcriptions.create.side_effect = FakeStatusError(500)
    with pytest.raises(RecognitionError) as error:
        service_class(client=client).transcribe(audio, SAMPLE_RATE)
    assert error.value.fatal is False


def test正常系_transcribeメソッドがGroqへverbose_json形式を要求すること(client, audio):
    GroqRecognitionService(client=client).transcribe(audio, SAMPLE_RATE)
    assert request_of(client)["response_format"] == "verbose_json"


def test正常系_create_clientメソッドがOpenAIクライアントへAPIキーとタイムアウトを渡すこと():
    pytest.importorskip("openai")
    with patch("openai.OpenAI") as openai_client:
        OpenAIRecognitionService(api_key="test-key", timeout=12.0)
    openai_client.assert_called_once_with(api_key="test-key", timeout=12.0)


def test正常系_create_clientメソッドがGroqクライアントへAPIキーとタイムアウトを渡すこと():
    pytest.importorskip("groq")
    with patch("groq.Groq") as groq_client:
        GroqRecognitionService(api_key="test-key", timeout=12.0)
    groq_client.assert_called_once_with(api_key="test-key", timeout=12.0)


@pytest.mark.parametrize(
    "service_class, module_name",
    [
        pytest.param(OpenAIRecognitionService, "openai", id="openai"),
        pytest.param(GroqRecognitionService, "groq", id="groq"),
    ],
)
def test異常系_create_clientメソッドがSDK未導入でConfigurationErrorを送出すること(service_class, module_name):
    with patch.dict("sys.modules", {module_name: None}):
        with pytest.raises(ConfigurationError):
            service_class(api_key="test-key")


@pytest.mark.parametrize(
    "language, expected",
    [("ja-JP", "ja"), ("en_US", "en"), ("JA", "ja"), ("ja", "ja"), (None, None), ("", None)],
)
def test正常系_to_whisper_languageメソッドが言語タグをISO639_1コードに変換すること(language, expected):
    assert to_whisper_language(language) == expected


@pytest.mark.parametrize("status_code", [401, 403, 404])
def test正常系_to_recognition_errorメソッドが回復しないステータスを継続不能と判定すること(status_code):
    assert to_recognition_error("OpenAI", FakeStatusError(status_code)).fatal is True


@pytest.mark.parametrize("status_code", [400, 408, 429, 500, 503])
def test正常系_to_recognition_errorメソッドが一時的なステータスを継続可能と判定すること(status_code):
    assert to_recognition_error("OpenAI", FakeStatusError(status_code)).fatal is False


def test正常系_to_recognition_errorメソッドがステータスの無い例外を継続可能と判定すること():
    assert to_recognition_error("OpenAI", TimeoutError("timed out")).fatal is False


def test正常系_to_recognition_errorメソッドがサービス名をメッセージに含めること():
    assert "Groq" in str(to_recognition_error("Groq", TimeoutError("timed out")))


# --- 実SDKとの結合確認（HTTP通信をMockTransportに差し替え、音声は外部へ送らない） ---


def build_sdk_service(service_name, handler):
    """実SDKのクライアントを、指定ハンドラが応答するサービスとして組み立てる"""
    if service_name == "openai":
        openai = pytest.importorskip("openai")
        # openaiが使うHTTPライブラリ（httpx / httpx2）はSDKの版で変わるため、SDK自身の継承元から特定する
        http_module_name = next(
            base.__module__.split(".")[0]
            for base in openai.DefaultHttpxClient.__mro__
            if not base.__module__.startswith("openai")
        )
        httpx = importlib.import_module(http_module_name)
        http_client = httpx.Client(transport=httpx.MockTransport(lambda request: handler(httpx, request)))
        client = openai.OpenAI(api_key="test-key", http_client=http_client, max_retries=0)
        return OpenAIRecognitionService(client=client, language="ja-JP")

    groq = pytest.importorskip("groq")
    httpx = pytest.importorskip("httpx")
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: handler(httpx, request)))
    client = groq.Groq(api_key="test-key", http_client=http_client, max_retries=0)
    return GroqRecognitionService(client=client, language="ja-JP")


SDK_NAMES = ["openai", "groq"]


@pytest.mark.parametrize("service_name", SDK_NAMES)
def test正常系_transcribeメソッドが実SDK経由で応答のテキストを取り出すこと(service_name, audio):
    service = build_sdk_service(service_name, lambda httpx, request: httpx.Response(200, json={"text": "こんにちは"}))
    assert service.transcribe(audio, SAMPLE_RATE) == "こんにちは"


@pytest.mark.parametrize("service_name", SDK_NAMES)
def test正常系_transcribeメソッドが実SDK経由で文字起こしAPIへ送信すること(service_name, audio):
    requested_paths = []

    def handler(httpx, request):
        requested_paths.append(request.url.path)
        return httpx.Response(200, json={"text": "こんにちは"})

    build_sdk_service(service_name, handler).transcribe(audio, SAMPLE_RATE)
    assert requested_paths[0].endswith("/audio/transcriptions")


@pytest.mark.parametrize("service_name", SDK_NAMES)
def test正常系_transcribeメソッドが実SDK経由のリクエストに言語を含めること(service_name, audio):
    bodies = []

    def handler(httpx, request):
        bodies.append(request.read())
        return httpx.Response(200, json={"text": "こんにちは"})

    build_sdk_service(service_name, handler).transcribe(audio, SAMPLE_RATE)
    assert b'name="language"\r\n\r\nja\r\n' in bodies[0]


@pytest.mark.parametrize("service_name", SDK_NAMES)
def test異常系_transcribeメソッドが実SDKの認証エラーを継続不能なRecognitionErrorとして送出すること(
    service_name, audio
):
    service = build_sdk_service(
        service_name, lambda httpx, request: httpx.Response(401, json={"error": {"message": "invalid api key"}})
    )
    with pytest.raises(RecognitionError) as error:
        service.transcribe(audio, SAMPLE_RATE)
    assert error.value.fatal is True


@pytest.mark.parametrize("service_name", SDK_NAMES)
def test異常系_transcribeメソッドが実SDKのサーバーエラーを継続可能なRecognitionErrorとして送出すること(
    service_name, audio
):
    service = build_sdk_service(
        service_name, lambda httpx, request: httpx.Response(500, json={"error": {"message": "server error"}})
    )
    with pytest.raises(RecognitionError) as error:
        service.transcribe(audio, SAMPLE_RATE)
    assert error.value.fatal is False
