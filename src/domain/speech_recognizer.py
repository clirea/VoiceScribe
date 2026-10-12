# domain/speech_recognizer.py

from typing import Protocol

import numpy as np


class SpeechRecognizer(Protocol):
    """音声認識サービスが満たすインターフェース。"""

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        モノラルのfloat音声をテキストに変換する。

        聞き取れなかった場合は空文字を返し、失敗した場合はRecognitionErrorを送出する。
        """
        ...
