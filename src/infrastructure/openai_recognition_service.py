# infrastructure/openai_recognition_service.py

import os
import io
import wave
import numpy as np
from openai import OpenAI

class OpenAIRecognitionService:
    """
    OpenAI APIを使って音声をテキストに変換するクラス。
    Whisperモデルを利用するわ。
    """

    def __init__(self, language="ja-JP"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        NumPy配列をWAVに変換してOpenAI APIで音声認識を行う。
        """
        if audio_data is None or audio_data.size == 0:
            return ""

        # NumPy配列 → WAVバイナリに変換
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16bit
                wf.setframerate(sample_rate)
                wf.writeframes((audio_data * 32767).astype('int16').tobytes())

            try:
                # バッファを巻き戻してOpenAI APIに送信
                wav_buffer.seek(0)
                transcription = self.client.audio.transcriptions.create(
                    file=("audio.wav", wav_buffer.read()),
                    model="whisper-1"
                )
                return transcription.text
            except Exception as e:
                print(f"OpenAIRecognitionService: 予期せぬエラーが発生したわ: {e}")
                return ""