# tests/test_wav_encoder.py

import io
import wave

import numpy as np
import pytest

from src.infrastructure.wav_encoder import encode_wav, to_pcm16, write_wav


def read_wav(data: bytes):
    return wave.open(io.BytesIO(data), "rb")


@pytest.fixture
def audio():
    return np.full(1600, 0.5, dtype=np.float32)


def test正常系_to_pcm16メソッドがfloat音声を16bitリトルエンディアンに変換すること():
    assert to_pcm16(np.array([0.5], dtype=np.float32)) == (16383).to_bytes(2, "little", signed=True)


def test正常系_to_pcm16メソッドが上限を超える値を最大値に丸めること():
    assert np.frombuffer(to_pcm16(np.array([2.0])), dtype="<i2")[0] == 32767


def test正常系_to_pcm16メソッドが下限を下回る値を最小値に丸めること():
    assert np.frombuffer(to_pcm16(np.array([-2.0])), dtype="<i2")[0] == -32767


def test正常系_to_pcm16メソッドが2次元配列を1次元の並びとして変換すること():
    assert len(to_pcm16(np.zeros((512, 1), dtype=np.float32))) == 512 * 2


def test正常系_encode_wavメソッドがモノラルのWAVを生成すること(audio):
    with read_wav(encode_wav(audio, 16000)) as wf:
        assert wf.getnchannels() == 1


def test正常系_encode_wavメソッドが16bitのWAVを生成すること(audio):
    with read_wav(encode_wav(audio, 16000)) as wf:
        assert wf.getsampwidth() == 2


def test正常系_encode_wavメソッドが指定したサンプルレートを記録すること(audio):
    with read_wav(encode_wav(audio, 8000)) as wf:
        assert wf.getframerate() == 8000


def test正常系_encode_wavメソッドが全サンプルを書き込むこと(audio):
    with read_wav(encode_wav(audio, 16000)) as wf:
        assert wf.getnframes() == 1600


def test正常系_write_wavメソッドが読み取り可能なWAVファイルを保存すること(audio, tmp_path):
    path = tmp_path / "audio.wav"
    write_wav(path, audio, 16000)
    with wave.open(str(path), "rb") as wf:
        assert wf.getnframes() == 1600
