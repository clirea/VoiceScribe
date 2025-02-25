# infrastructure/groq_recognition_service.py

import os
import io
import wave
import numpy as np
from groq import Groq

class GroqRecognitionService:
    """
    Groq APIを使って音声をテキストに変換するクラス。
    Whisper Large V3 Turboモデルを利用するわ。
    """

    def __init__(self, language="ja-JP"):
        self.language = language
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        NumPy配列をWAVに変換してGroq APIで音声認識を行う。
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
                # バッファを巻き戻してGroq APIに送信
                wav_buffer.seek(0)
                transcription = self.client.audio.transcriptions.create(
                    file=("audio.wav", wav_buffer.read()),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json"
                )
                return transcription.text
            except Exception as e:
                print(f"GroqRecognitionService: 予期せぬエラーが発生したわ: {e}")
                return ""