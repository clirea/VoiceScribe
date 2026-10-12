# infrastructure/wav_encoder.py

import io
import wave

import numpy as np

SAMPLE_WIDTH = 2  # 16bit


def to_pcm16(audio_data: np.ndarray) -> bytes:
    """-1.0〜1.0のfloat音声を16bitリトルエンディアンPCMに変換する。範囲外の値は折り返さず丸める。"""
    clipped = np.clip(np.asarray(audio_data, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (clipped * 32767).astype("<i2").tobytes()


def encode_wav(audio_data: np.ndarray, sample_rate: int) -> bytes:
    """NumPyのモノラル音声配列をWAVバイナリに変換する。"""
    with io.BytesIO() as wav_buffer:
        _write(wav_buffer, audio_data, sample_rate)
        return wav_buffer.getvalue()


def write_wav(path, audio_data: np.ndarray, sample_rate: int) -> None:
    """NumPyのモノラル音声配列をWAVファイルに保存する。"""
    with open(path, "wb") as f:
        _write(f, audio_data, sample_rate)


def _write(file, audio_data: np.ndarray, sample_rate: int) -> None:
    with wave.open(file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(to_pcm16(audio_data))
