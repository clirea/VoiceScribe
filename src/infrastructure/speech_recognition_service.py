# infrastructure/speech_recognition_service.py

import numpy as np

from src.domain.errors import ConfigurationError, RecognitionError
from src.infrastructure.wav_encoder import SAMPLE_WIDTH, to_pcm16

DEFAULT_LANGUAGE = "ja-JP"
# SpeechRecognitionの既定（http://）と同じAPIを、暗号化して呼び出す
GOOGLE_SPEECH_ENDPOINT = "https://www.google.com/speech-api/v2/recognize"


class SpeechRecognitionService:
    """
    speech_recognitionを使い、無料枠のGoogle Speech Recognitionで音声をテキストに変換するクラス。
    """

    def __init__(self, language=None, recognizer=None, timeout=30.0):
        try:
            import speech_recognition as sr
        except ImportError as e:
            raise ConfigurationError(
                "SpeechRecognition を読み込めません。`pip install -r requirements.txt` を実行してください。"
            ) from e

        self._sr = sr
        self.language = language or DEFAULT_LANGUAGE
        self.recognizer = recognizer if recognizer is not None else sr.Recognizer()
        # 応答が返らないままループが止まり続けないようにする
        self.recognizer.operation_timeout = timeout

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        NumPy配列を16bit PCMに変換して音声認識を行う。
        聞き取れなかった場合は空文字を返し、APIの失敗はRecognitionErrorにする。
        """
        if audio_data is None or audio_data.size == 0:
            return ""

        audio = self._sr.AudioData(to_pcm16(audio_data), sample_rate, SAMPLE_WIDTH)

        try:
            # 無料枠のGoogle Speech Recognition（リクエスト数に制限あり）
            text = self.recognizer.recognize_google(
                audio, language=self.language, endpoint=GOOGLE_SPEECH_ENDPOINT
            )
        except self._sr.UnknownValueError:
            # 音声がはっきりしない場合
            return ""
        except self._sr.RequestError as e:
            raise RecognitionError(f"Google音声認識へのリクエストに失敗しました: {e}") from e
        except Exception as e:
            raise RecognitionError(f"Google音声認識で予期せぬエラーが発生しました: {e}") from e
        return text or ""
