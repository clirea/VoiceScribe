# tests/test_speech_segmenter.py

import numpy as np
import pytest

from src.domain.speech_segmenter import SpeechSegmenter

CHUNK = 512
# 16kHz・512サンプルのチャンクは0.032秒。無音1.0秒に達するのは32チャンク目
SILENCE_CHUNKS_TO_END = 32


def chunk(value=0.5):
    return np.full(CHUNK, value, dtype=np.float32)


def feed_many(segmenter, count, is_speech, value=0.5):
    """同じチャンクをcount回入力し、各回の戻り値を返す"""
    return [segmenter.feed(chunk(value), is_speech) for _ in range(count)]


@pytest.fixture
def segmenter():
    return SpeechSegmenter(sample_rate=16000, silence_duration=1.0, pre_buffer_duration=0.0, max_speech_duration=0)


def test正常系_feedメソッドが無音のみの入力では区間を返さないこと(segmenter):
    results = feed_many(segmenter, 100, is_speech=False)
    assert all(result is None for result in results)


def test正常系_feedメソッドが発話の継続中は区間を返さないこと(segmenter):
    results = feed_many(segmenter, 100, is_speech=True)
    assert all(result is None for result in results)


def test正常系_feedメソッドが発話開始でis_speech_activeをTrueにすること(segmenter):
    segmenter.feed(chunk(), True)
    assert segmenter.is_speech_active is True


def test正常系_feedメソッドが無音の継続時間に達したら区間を返すこと(segmenter):
    feed_many(segmenter, 3, is_speech=True)
    results = feed_many(segmenter, SILENCE_CHUNKS_TO_END, is_speech=False)
    assert results[-1] is not None


def test正常系_feedメソッドが無音の継続時間に満たない間は区間を確定しないこと(segmenter):
    feed_many(segmenter, 3, is_speech=True)
    results = feed_many(segmenter, SILENCE_CHUNKS_TO_END - 1, is_speech=False)
    assert all(result is None for result in results)


def test正常系_feedメソッドが末尾の無音を区間に含めないこと(segmenter):
    feed_many(segmenter, 3, is_speech=True)
    segment = feed_many(segmenter, SILENCE_CHUNKS_TO_END, is_speech=False)[-1]
    assert len(segment) == 3 * CHUNK


def test正常系_feedメソッドが発話の合間の短い無音を区間に含めること(segmenter):
    feed_many(segmenter, 2, is_speech=True)
    feed_many(segmenter, 5, is_speech=False)
    feed_many(segmenter, 2, is_speech=True)
    segment = feed_many(segmenter, SILENCE_CHUNKS_TO_END, is_speech=False)[-1]
    assert len(segment) == 9 * CHUNK


def test正常系_feedメソッドが発話再開で無音の計測をやり直すこと(segmenter):
    feed_many(segmenter, 2, is_speech=True)
    feed_many(segmenter, SILENCE_CHUNKS_TO_END - 1, is_speech=False)
    segmenter.feed(chunk(), True)
    results = feed_many(segmenter, SILENCE_CHUNKS_TO_END - 1, is_speech=False)
    assert all(result is None for result in results)


def test正常系_feedメソッドが発話直前の音声をプリバッファとして区間に含めること():
    segmenter = SpeechSegmenter(silence_duration=1.0, pre_buffer_duration=0.5, max_speech_duration=0)
    # 0.5秒=8000サンプルに収まるのは15チャンクまで
    feed_many(segmenter, 40, is_speech=False)
    feed_many(segmenter, 3, is_speech=True)
    segment = feed_many(segmenter, SILENCE_CHUNKS_TO_END, is_speech=False)[-1]
    assert len(segment) == (15 + 3) * CHUNK


def test正常系_feedメソッドがプリバッファを発話より前に並べること():
    segmenter = SpeechSegmenter(silence_duration=1.0, pre_buffer_duration=0.5, max_speech_duration=0)
    feed_many(segmenter, 5, is_speech=False, value=0.1)
    feed_many(segmenter, 3, is_speech=True, value=0.5)
    segment = feed_many(segmenter, SILENCE_CHUNKS_TO_END, is_speech=False, value=0.1)[-1]
    assert segment[0] == pytest.approx(0.1)


def test正常系_feedメソッドが区間の末尾を発話の音声にすること():
    segmenter = SpeechSegmenter(silence_duration=1.0, pre_buffer_duration=0.5, max_speech_duration=0)
    feed_many(segmenter, 5, is_speech=False, value=0.1)
    feed_many(segmenter, 3, is_speech=True, value=0.5)
    segment = feed_many(segmenter, SILENCE_CHUNKS_TO_END, is_speech=False, value=0.1)[-1]
    assert segment[-1] == pytest.approx(0.5)


def test正常系_feedメソッドが区間確定後にプリバッファを持ち越さないこと():
    segmenter = SpeechSegmenter(silence_duration=1.0, pre_buffer_duration=0.5, max_speech_duration=0)
    feed_many(segmenter, 5, is_speech=False)
    feed_many(segmenter, 3, is_speech=True)
    feed_many(segmenter, SILENCE_CHUNKS_TO_END, is_speech=False)
    feed_many(segmenter, 2, is_speech=True)
    segment = feed_many(segmenter, SILENCE_CHUNKS_TO_END, is_speech=False)[-1]
    assert len(segment) == 2 * CHUNK


def test正常系_feedメソッドが区間確定後にis_speech_activeをFalseに戻すこと(segmenter):
    feed_many(segmenter, 3, is_speech=True)
    feed_many(segmenter, SILENCE_CHUNKS_TO_END, is_speech=False)
    assert segmenter.is_speech_active is False


def test正常系_feedメソッドが最大長に達した発話を無音を待たずに区切ること():
    segmenter = SpeechSegmenter(silence_duration=1.0, pre_buffer_duration=0.0, max_speech_duration=0.064)
    results = feed_many(segmenter, 2, is_speech=True)
    assert len(results[-1]) == 2 * CHUNK


def test正常系_feedメソッドが最大長で区切った後も次の発話を新しい区間として扱うこと():
    segmenter = SpeechSegmenter(silence_duration=1.0, pre_buffer_duration=0.0, max_speech_duration=0.064)
    feed_many(segmenter, 2, is_speech=True)
    results = feed_many(segmenter, 2, is_speech=True)
    assert len(results[-1]) == 2 * CHUNK


def test正常系_flushメソッドが切り出し途中の発話を返すこと(segmenter):
    feed_many(segmenter, 4, is_speech=True)
    assert len(segmenter.flush()) == 4 * CHUNK


def test正常系_flushメソッドが発話が無いときNoneを返すこと(segmenter):
    feed_many(segmenter, 4, is_speech=False)
    assert segmenter.flush() is None


def test正常系_flushメソッドが返した発話を二重に返さないこと(segmenter):
    feed_many(segmenter, 4, is_speech=True)
    segmenter.flush()
    assert segmenter.flush() is None


def test正常系_flushメソッドがis_speech_activeをFalseに戻すこと(segmenter):
    feed_many(segmenter, 4, is_speech=True)
    segmenter.flush()
    assert segmenter.is_speech_active is False



def test正常系_buffered_samplesメソッドが切り出し途中の発話のサンプル数を返すこと(segmenter):
    segmenter.feed(np.zeros(512, dtype=np.float32), True)
    assert segmenter.buffered_samples == 512


def test正常系_buffered_samplesメソッドが区間を返した後に0を返すこと(segmenter):
    segmenter.feed(np.zeros(512, dtype=np.float32), True)
    segmenter.flush()
    assert segmenter.buffered_samples == 0
