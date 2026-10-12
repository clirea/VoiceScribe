# tests/test_google_cloud_recognition_service.py

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.domain.errors import ConfigurationError, RecognitionError
from src.infrastructure.google_cloud_recognition_service import GoogleCloudRecognitionService
from src.infrastructure.wav_encoder import to_pcm16

speech = pytest.importorskip("google.cloud.speech", reason="requirements-google-cloud.txt が必要")

from google.api_core import exceptions as api_exceptions  # noqa: E402
from google.auth import exceptions as auth_exceptions  # noqa: E402
from google.auth.credentials import AnonymousCredentials  # noqa: E402

SAMPLE_RATE = 16000
SECRET_MARKER = "SECRET-MARKER"


@pytest.fixture
def audio():
    return np.full(1600, 0.5, dtype=np.float32)


def response_with(*transcripts):
    return speech.RecognizeResponse(
        results=[
            speech.SpeechRecognitionResult(alternatives=[speech.SpeechRecognitionAlternative(transcript=transcript)])
            for transcript in transcripts
        ]
    )


@pytest.fixture
def client():
    """認証もAPI呼び出しも行わないクライアント（送信内容だけを記録する）"""
    client = MagicMock()
    client.recognize.return_value = response_with("こんにちは")
    return client


@pytest.fixture
def service(client):
    return GoogleCloudRecognitionService(client=client)


def test正常系_transcribeメソッドが認識結果を返すこと(service, audio):
    assert service.transcribe(audio, SAMPLE_RATE) == "こんにちは"


def test正常系_transcribeメソッドが複数の結果を空白でつなげて返すこと(service, client, audio):
    client.recognize.return_value = response_with("こんにちは ", "今日の会議を始めます")
    assert service.transcribe(audio, SAMPLE_RATE) == "こんにちは 今日の会議を始めます"


def test正常系_transcribeメソッドが結果の無い応答に対して空文字を返すこと(service, client, audio):
    client.recognize.return_value = speech.RecognizeResponse()
    assert service.transcribe(audio, SAMPLE_RATE) == ""


def test正常系_transcribeメソッドが候補の無い結果を読み飛ばすこと(service, client, audio):
    client.recognize.return_value = speech.RecognizeResponse(results=[speech.SpeechRecognitionResult()])
    assert service.transcribe(audio, SAMPLE_RATE) == ""


def test正常系_transcribeメソッドが空の音声に対して空文字を返すこと(service):
    assert service.transcribe(np.array([], dtype=np.float32), SAMPLE_RATE) == ""


def test正常系_transcribeメソッドが空の音声ではAPIを呼ばないこと(service, client):
    service.transcribe(None, SAMPLE_RATE)
    client.recognize.assert_not_called()


def test正常系_transcribeメソッドが言語未指定のとき日本語で認識すること(service, client, audio):
    service.transcribe(audio, SAMPLE_RATE)
    assert client.recognize.call_args.kwargs["config"].language_code == "ja-JP"


def test正常系_transcribeメソッドが指定されたタイムアウトを渡すこと(client, audio):
    GoogleCloudRecognitionService(client=client, timeout=12.0).transcribe(audio, SAMPLE_RATE)
    assert client.recognize.call_args.kwargs["timeout"] == 12.0


def test正常系_transcribeメソッドがSDK既定の再試行を無効にすること(service, client, audio):
    service.transcribe(audio, SAMPLE_RATE)
    assert client.recognize.call_args.kwargs["retry"] is None


@pytest.mark.parametrize(
    "error",
    [
        api_exceptions.Unauthenticated("invalid credentials"),
        api_exceptions.PermissionDenied("speech api disabled"),
        api_exceptions.NotFound("model not found"),
    ],
    ids=["認証失敗", "権限なし", "対象なし"],
)
def test異常系_transcribeメソッドが回復しないAPIエラーを継続不能なRecognitionErrorとして送出すること(
    service, client, audio, error
):
    client.recognize.side_effect = error
    with pytest.raises(RecognitionError) as raised:
        service.transcribe(audio, SAMPLE_RATE)
    assert raised.value.fatal is True


@pytest.mark.parametrize(
    "error",
    [
        api_exceptions.DeadlineExceeded("timeout"),
        api_exceptions.ServiceUnavailable("unavailable"),
        api_exceptions.ResourceExhausted("quota"),
        api_exceptions.InvalidArgument("audio too long"),
    ],
    ids=["タイムアウト", "一時的な停止", "割り当て超過", "不正な音声"],
)
def test異常系_transcribeメソッドが一時的なAPIエラーを継続可能なRecognitionErrorとして送出すること(
    service, client, audio, error
):
    client.recognize.side_effect = error
    with pytest.raises(RecognitionError) as raised:
        service.transcribe(audio, SAMPLE_RATE)
    assert raised.value.fatal is False


def test異常系_transcribeメソッドが認証の失敗を継続不能なRecognitionErrorとして送出すること(service, client, audio):
    client.recognize.side_effect = auth_exceptions.RefreshError("invalid_grant")
    with pytest.raises(RecognitionError) as raised:
        service.transcribe(audio, SAMPLE_RATE)
    assert raised.value.fatal is True


def test異常系_transcribeメソッドが認証の失敗の応答内容をエラーに含めないこと(service, client, audio):
    client.recognize.side_effect = auth_exceptions.RefreshError(SECRET_MARKER)
    with pytest.raises(RecognitionError) as raised:
        service.transcribe(audio, SAMPLE_RATE)
    assert SECRET_MARKER not in str(raised.value)


def test異常系_transcribeメソッドが予期せぬ例外でRecognitionErrorを送出すること(service, client, audio):
    client.recognize.side_effect = RuntimeError("channel closed")
    with pytest.raises(RecognitionError):
        service.transcribe(audio, SAMPLE_RATE)


# --- __init__ ---


def test正常系_initメソッドがクライアント未指定のとき既定の認証でクライアントを生成すること():
    with patch.object(speech, "SpeechClient") as speech_client:
        service = GoogleCloudRecognitionService()
    assert service.client is speech_client.return_value


def test異常系_initメソッドが既定の認証情報が無いときConfigurationErrorを送出すること():
    with patch.object(speech, "SpeechClient", side_effect=auth_exceptions.DefaultCredentialsError("not found")):
        with pytest.raises(ConfigurationError):
            GoogleCloudRecognitionService()


def test異常系_initメソッドがgoogle_cloud_speech未導入でConfigurationErrorを送出すること(client, monkeypatch):
    import google.cloud

    monkeypatch.delattr(google.cloud, "speech")
    with patch.dict("sys.modules", {"google.cloud.speech": None}):
        with pytest.raises(ConfigurationError):
            GoogleCloudRecognitionService(client=client)


# --- from_service_account（実際のSDKで認証情報を読み込む。鍵はテストのたびに生成し、外部へは接続しない） ---


@pytest.fixture(scope="module")
def service_account_info():
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
    rsa = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.rsa")
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    return {
        "type": "service_account",
        "project_id": "voicescribe-test",
        "private_key_id": "0" * 40,
        "private_key": private_key.decode("ascii"),
        "client_email": "voicescribe-test@voicescribe-test.iam.gserviceaccount.com",
        "client_id": "1",
        "token_uri": "https://oauth2.googleapis.com/token",
    }


@pytest.fixture
def service_account_file(tmp_path, service_account_info):
    path = tmp_path / "service_account.json"
    path.write_text(json.dumps(service_account_info), encoding="utf-8")
    return str(path)


def signer_email(service):
    return service.client.transport._credentials.signer_email


def test正常系_from_service_accountメソッドがJSON文字列の内容からSDKのクライアントを生成すること(
    service_account_info,
):
    service = GoogleCloudRecognitionService.from_service_account(credentials_info=service_account_info)
    assert isinstance(service.client, speech.SpeechClient)


def test正常系_from_service_accountメソッドがJSON文字列の内容のサービスアカウントで認証すること(
    service_account_info,
):
    service = GoogleCloudRecognitionService.from_service_account(credentials_info=service_account_info)
    assert signer_email(service) == service_account_info["client_email"]


def test正常系_from_service_accountメソッドがファイルパスからSDKのクライアントを生成すること(service_account_file):
    service = GoogleCloudRecognitionService.from_service_account(credentials_path=service_account_file)
    assert isinstance(service.client, speech.SpeechClient)


def test正常系_from_service_accountメソッドがファイルのサービスアカウントで認証すること(
    service_account_file, service_account_info
):
    service = GoogleCloudRecognitionService.from_service_account(credentials_path=service_account_file)
    assert signer_email(service) == service_account_info["client_email"]


def test正常系_from_service_accountメソッドが言語とタイムアウトを引き継ぐこと(service_account_info):
    service = GoogleCloudRecognitionService.from_service_account(
        credentials_info=service_account_info, language="en-US", timeout=12.0
    )
    assert (service.language, service.timeout) == ("en-US", 12.0)


def test異常系_from_service_accountメソッドがサービスアカウントでないJSONでConfigurationErrorを送出すること():
    with pytest.raises(ConfigurationError):
        GoogleCloudRecognitionService.from_service_account(credentials_info={"type": "service_account"})


def test異常系_from_service_accountメソッドが壊れた秘密鍵でConfigurationErrorを送出すること(service_account_info):
    with pytest.raises(ConfigurationError):
        GoogleCloudRecognitionService.from_service_account(
            credentials_info={**service_account_info, "private_key": SECRET_MARKER}
        )


def test異常系_from_service_accountメソッドが認証情報の内容をエラーに含めないこと(service_account_info):
    with pytest.raises(ConfigurationError) as raised:
        GoogleCloudRecognitionService.from_service_account(
            credentials_info={**service_account_info, "private_key": SECRET_MARKER}
        )
    assert SECRET_MARKER not in str(raised.value)


def test異常系_from_service_accountメソッドが読み込み時の例外を連鎖させないこと(service_account_info):
    with pytest.raises(ConfigurationError) as raised:
        GoogleCloudRecognitionService.from_service_account(
            credentials_info={**service_account_info, "private_key": SECRET_MARKER}
        )
    assert raised.value.__cause__ is None


def test異常系_from_service_accountメソッドがJSONでないファイルでConfigurationErrorを送出すること(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        GoogleCloudRecognitionService.from_service_account(credentials_path=str(path))


# --- 実SDKの呼び出し契約（gRPCの送信だけを差し替え、音声は外部へ送らない） ---


@pytest.fixture
def sdk_client():
    return speech.SpeechClient(credentials=AnonymousCredentials())


@pytest.fixture
def grpc_call(sdk_client):
    """SDKが最後に呼ぶgRPCの送信処理を差し替え、渡されたリクエストとタイムアウトを記録する"""
    with patch.object(type(sdk_client.transport.recognize), "__call__") as grpc_call:
        grpc_call.return_value = response_with("こんにちは")
        yield grpc_call


@pytest.fixture
def sdk_service(sdk_client, grpc_call):
    return GoogleCloudRecognitionService(client=sdk_client, timeout=12.0)


def sent_request(grpc_call):
    return grpc_call.call_args.args[0]


def test正常系_transcribeメソッドが実SDK経由で応答からテキストを取り出すこと(sdk_service, audio):
    assert sdk_service.transcribe(audio, SAMPLE_RATE) == "こんにちは"


def test正常系_transcribeメソッドが実SDK経由でRecognizeRequestを送ること(sdk_service, grpc_call, audio):
    sdk_service.transcribe(audio, SAMPLE_RATE)
    assert isinstance(sent_request(grpc_call), speech.RecognizeRequest)


def test正常系_transcribeメソッドが実SDK経由で16bitのPCMとして送ること(sdk_service, grpc_call, audio):
    sdk_service.transcribe(audio, SAMPLE_RATE)
    assert sent_request(grpc_call).config.encoding == speech.RecognitionConfig.AudioEncoding.LINEAR16


def test正常系_transcribeメソッドが実SDK経由でサンプルレートを送ること(sdk_service, grpc_call, audio):
    sdk_service.transcribe(audio, SAMPLE_RATE)
    assert sent_request(grpc_call).config.sample_rate_hertz == SAMPLE_RATE


def test正常系_transcribeメソッドが実SDK経由で言語コードを送ること(sdk_client, grpc_call, audio):
    GoogleCloudRecognitionService(language="en-US", client=sdk_client).transcribe(audio, SAMPLE_RATE)
    assert sent_request(grpc_call).config.language_code == "en-US"


def test正常系_transcribeメソッドが実SDK経由で変換した音声を送ること(sdk_service, grpc_call, audio):
    sdk_service.transcribe(audio, SAMPLE_RATE)
    assert sent_request(grpc_call).audio.content == to_pcm16(audio)


def test正常系_transcribeメソッドが実SDK経由のgRPC呼び出しにタイムアウトを設定すること(
    sdk_service, grpc_call, audio
):
    sdk_service.transcribe(audio, SAMPLE_RATE)
    assert grpc_call.call_args.kwargs["timeout"] == 12.0


def test異常系_transcribeメソッドが実SDK経由の一時的な失敗を再試行せずRecognitionErrorを送出すること(
    sdk_service, grpc_call, audio
):
    grpc_call.side_effect = api_exceptions.ServiceUnavailable("unavailable")
    with pytest.raises(RecognitionError):
        sdk_service.transcribe(audio, SAMPLE_RATE)
    assert grpc_call.call_count == 1
