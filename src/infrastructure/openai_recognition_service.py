# infrastructure/openai_recognition_service.py

import numpy as np

from src.domain.errors import ConfigurationError
from src.infrastructure.wav_encoder import encode_wav
from src.infrastructure.whisper_api_support import to_recognition_error, to_whisper_language

DEFAULT_MODEL = "whisper-1"


class OpenAIRecognitionService:
    """
    OpenAI APIを使って音声をテキストに変換するクラス。
    既定ではWhisperモデルを利用するわ。
    """

    def __init__(self, api_key=None, language=None, model=None, client=None, timeout=30.0):
        self.language = to_whisper_language(language)
        self.model = model or DEFAULT_MODEL
        self.client = client if client is not None else self._create_client(api_key, timeout)

    @staticmethod
    def _create_client(api_key, timeout):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ConfigurationError(
                "openai を読み込めません。`pip install -r requirements.txt` を実行してください。"
            ) from e
        return OpenAI(api_key=api_key, timeout=timeout)

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        NumPy配列をWAVに変換してOpenAI APIで音声認識を行う。
        """
        if audio_data is None or audio_data.size == 0:
            return ""

        request = {
            "file": ("audio.wav", encode_wav(audio_data, sample_rate), "audio/wav"),
            "model": self.model,
        }
        # 言語を指定しない場合はAPI側の自動判定に任せる
        if self.language:
            request["language"] = self.language

        try:
            transcription = self.client.audio.transcriptions.create(**request)
        except Exception as e:
            raise to_recognition_error("OpenAI", e) from e
        return transcription.text or ""
