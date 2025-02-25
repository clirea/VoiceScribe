# infrastructure/speech_recognition_service.py

import speech_recognition as sr
import io
import wave
import numpy as np
import os

class SpeechRecognitionService:
    """
    speech_recognitionを使って音声をテキストに変換するクラス。
    Google Speech APIなど、デフォルトの認識エンジンを利用可能。
    """

    def __init__(self, language="ja-JP"):
        self.language = language
        self.recognizer = sr.Recognizer()
        self.api_key = os.getenv("GOOGLE_API_KEY")

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        NumPy配列をWAVに変換して音声認識を行う。
        Google Cloud Speech APIキーがある場合はそれを使用し、
        ない場合は無料枠のGoogle Speech Recognitionを使用する。
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

            # バッファを巻き戻してspeech_recognitionに渡す
            wav_buffer.seek(0)
            with sr.AudioFile(wav_buffer) as source:
                audio = self.recognizer.record(source)

            try:
                # Google Cloud Speech APIキーがある場合はそれを使用
                if self.api_key:
                    text = self.recognizer.recognize_google_cloud(
                        audio, 
                        language=self.language,
                        credentials_json=self.api_key
                    )
                else:
                    # 無料枠のGoogle Speech Recognition（1分間に20回までのリクエスト制限あり）
                    text = self.recognizer.recognize_google(audio, language=self.language)
                return text
            except sr.UnknownValueError:
                # 音声がはっきりしない場合
                return ""
            except sr.RequestError as e:
                print(f"SpeechRecognitionService: APIへのリクエストに失敗しました: {e}")
                return ""
            except Exception as e:
                print(f"SpeechRecognitionService: 予期せぬエラーが発生しました: {e}")
                return ""
