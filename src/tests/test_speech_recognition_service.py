# tests/test_speech_recognition_service.py

import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.domain.errors import ConfigurationError, RecognitionError
from src.infrastructure.speech_recognition_service import SpeechRecognitionService

sr = pytest.importorskip("speech_recognition")

SAMPLE_RATE = 16000


@pytest.fixture
def audio():
    return np.full(1600, 0.5, dtype=np.float32)


@pytest.fixture
def recognizer():
    recognizer = MagicMock()
    recognizer.recognize_google.return_value = "こんにちは"
    return recognizer


@pytest.fixture
def service(recognizer):
    return SpeechRecognitionService(recognizer=recognizer)


def test正常系_transcribeメソッドが無料枠の認識結果を返すこと(service, audio):
    assert service.transcribe(audio, SAMPLE_RATE) == "こんにちは"


def test正常系_transcribeメソッドが言語未指定のとき日本語で認識すること(service, recognizer, audio):
    service.transcribe(audio, SAMPLE_RATE)
    assert recognizer.recognize_google.call_args.kwargs["language"] == "ja-JP"


def test正常系_transcribeメソッドが指定された言語で認識すること(recognizer, audio):
    SpeechRecognitionService(language="en-US", recognizer=recognizer).transcribe(audio, SAMPLE_RATE)
    assert recognizer.recognize_google.call_args.kwargs["language"] == "en-US"


def test正常系_transcribeメソッドがサンプルレートを保った音声データを渡すこと(service, recognizer, audio):
    service.transcribe(audio, SAMPLE_RATE)
    assert recognizer.recognize_google.call_args.args[0].sample_rate == SAMPLE_RATE


def test正常系_transcribeメソッドが16bitの音声データを渡すこと(service, recognizer, audio):
    service.transcribe(audio, SAMPLE_RATE)
    assert recognizer.recognize_google.call_args.args[0].sample_width == 2


def test正常系_transcribeメソッドが空の音声に対して空文字を返すこと(service):
    assert service.transcribe(np.array([], dtype=np.float32), SAMPLE_RATE) == ""


def test正常系_transcribeメソッドが空の音声ではAPIを呼ばないこと(service, recognizer):
    service.transcribe(np.array([], dtype=np.float32), SAMPLE_RATE)
    recognizer.recognize_google.assert_not_called()


def test正常系_transcribeメソッドがNoneの音声に対して空文字を返すこと(service):
    assert service.transcribe(None, SAMPLE_RATE) == ""


def test正常系_transcribeメソッドが聞き取れない音声に対して空文字を返すこと(service, recognizer, audio):
    recognizer.recognize_google.side_effect = sr.UnknownValueError()
    assert service.transcribe(audio, SAMPLE_RATE) == ""


def test正常系_transcribeメソッドが結果の無い応答に対して空文字を返すこと(service, recognizer, audio):
    recognizer.recognize_google.return_value = None
    assert service.transcribe(audio, SAMPLE_RATE) == ""


def test異常系_transcribeメソッドがリクエスト失敗でRecognitionErrorを送出すること(service, recognizer, audio):
    recognizer.recognize_google.side_effect = sr.RequestError("connection failed")
    with pytest.raises(RecognitionError):
        service.transcribe(audio, SAMPLE_RATE)


def test異常系_transcribeメソッドがリクエスト失敗を継続可能なRecognitionErrorとして送出すること(
    service, recognizer, audio
):
    recognizer.recognize_google.side_effect = sr.RequestError("connection failed")
    with pytest.raises(RecognitionError) as error:
        service.transcribe(audio, SAMPLE_RATE)
    assert error.value.fatal is False


def test異常系_transcribeメソッドが予期せぬ例外でRecognitionErrorを送出すること(service, recognizer, audio):
    recognizer.recognize_google.side_effect = OSError("FLAC conversion utility not available")
    with pytest.raises(RecognitionError):
        service.transcribe(audio, SAMPLE_RATE)


def test正常系_transcribeメソッドが暗号化されたエンドポイントを指定すること(service, recognizer, audio):
    service.transcribe(audio, SAMPLE_RATE)
    assert recognizer.recognize_google.call_args.kwargs["endpoint"].startswith("https://")


def test正常系_initメソッドが認識器へタイムアウトを設定すること(recognizer):
    SpeechRecognitionService(recognizer=recognizer, timeout=12.0)
    assert recognizer.operation_timeout == 12.0


def test正常系_initメソッドが認識器未指定のときRecognizerを生成すること():
    assert isinstance(SpeechRecognitionService().recognizer, sr.Recognizer)


def test異常系_initメソッドがSpeechRecognition未導入でConfigurationErrorを送出すること():
    with patch.dict("sys.modules", {"speech_recognition": None}):
        with pytest.raises(ConfigurationError):
            SpeechRecognitionService()


# --- 実ライブラリとの結合確認（HTTP送信だけを差し替え、音声は外部へ送らない） ---


@pytest.fixture
def google_urlopen():
    """Google Speech APIの応答を模したurlopenに差し替える"""
    response_body = "\n".join(
        [
            json.dumps({"result": []}),
            json.dumps(
                {
                    "result": [{"alternative": [{"transcript": "こんにちは", "confidence": 0.9}], "final": True}],
                    "result_index": 0,
                }
            ),
        ]
    ).encode("utf-8")
    # FLAC変換は外部コマンドに依存するため差し替え、HTTP層へ渡るリクエストだけを確かめる
    with (
        patch("speech_recognition.recognizers.google.urlopen") as urlopen,
        patch.object(sr.AudioData, "get_flac_data", return_value=b"fLaC"),
    ):
        urlopen.side_effect = lambda *args, **kwargs: io.BytesIO(response_body)
        yield urlopen


def test正常系_transcribeメソッドが実ライブラリ経由でGoogleの応答からテキストを取り出すこと(google_urlopen, audio):
    assert SpeechRecognitionService().transcribe(audio, SAMPLE_RATE) == "こんにちは"


def test正常系_transcribeメソッドが実ライブラリ経由のリクエストに言語を含めること(google_urlopen, audio):
    SpeechRecognitionService().transcribe(audio, SAMPLE_RATE)
    assert "lang=ja-JP" in google_urlopen.call_args.args[0].full_url


def test正常系_transcribeメソッドが実ライブラリ経由のリクエストにタイムアウトを設定すること(google_urlopen, audio):
    SpeechRecognitionService(timeout=12.0).transcribe(audio, SAMPLE_RATE)
    assert google_urlopen.call_args.kwargs["timeout"] == 12.0


def test正常系_transcribeメソッドが実ライブラリ経由のリクエストをHTTPSで送ること(google_urlopen, audio):
    SpeechRecognitionService().transcribe(audio, SAMPLE_RATE)
    assert google_urlopen.call_args.args[0].type == "https"


def test正常系_transcribeメソッドが実ライブラリ経由のリクエストを従来と同じAPIへ送ること(google_urlopen, audio):
    SpeechRecognitionService().transcribe(audio, SAMPLE_RATE)
    request = google_urlopen.call_args.args[0]
    assert (request.host, request.selector.split("?")[0]) == ("www.google.com", "/speech-api/v2/recognize")


def test正常系_transcribeメソッドが実ライブラリ経由でHTTP層を1回だけ呼ぶこと(google_urlopen, audio):
    SpeechRecognitionService().transcribe(audio, SAMPLE_RATE)
    assert google_urlopen.call_count == 1
