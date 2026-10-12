# domain/speech_segmenter.py

from collections import deque

import numpy as np


class SpeechSegmenter:
    """
    VADの判定結果つきチャンクを受け取り、発話区間を切り出す状態機械。

    マイクや音声認識には依存せず、無音が silence_duration 続いた時点、
    または発話が max_speech_duration に達した時点で区間を返す。
    """

    def __init__(
        self,
        sample_rate=16000,
        silence_duration=1.0,      # 無音判定に必要な継続秒数
        pre_buffer_duration=0.5,   # 音声区間開始前のバッファ保持時間（秒）
        max_speech_duration=30.0,  # 1区間の最大秒数（0以下で無制限）
    ):
        self.sample_rate = sample_rate
        self.silence_duration = silence_duration
        self.pre_buffer_size = int(pre_buffer_duration * sample_rate)
        self.max_speech_samples = int(max_speech_duration * sample_rate) if max_speech_duration > 0 else 0

        self.is_speech_active = False
        self._pre_buffer = deque()
        self._pre_buffer_samples = 0
        self._speech_buffer = []
        self._speech_samples = 0
        # 発話中の無音は、発話が再開した場合だけ区間に含める
        self._pending_silence = []
        self._silence_seconds = 0.0

    def feed(self, chunk: np.ndarray, is_speech: bool):
        """チャンクを1つ取り込み、区間が確定したらその音声データを、未確定ならNoneを返す。"""
        if is_speech:
            self.is_speech_active = True
            self._silence_seconds = 0.0
            self._append_speech(self._pending_silence)
            self._pending_silence = []
            self._append_speech([chunk])
            if self.max_speech_samples and self._speech_samples >= self.max_speech_samples:
                return self._finish_segment()
            return None

        if not self.is_speech_active:
            self._update_pre_buffer(chunk)
            return None

        self._pending_silence.append(chunk)
        self._silence_seconds += len(chunk) / self.sample_rate
        if self._silence_seconds >= self.silence_duration:
            return self._finish_segment()
        return None

    def flush(self):
        """終了時などに、切り出し途中の発話があれば返す。"""
        if not self._speech_buffer:
            return None
        return self._finish_segment()

    @property
    def buffered_samples(self) -> int:
        """切り出し途中の発話として保持しているサンプル数。"""
        return self._speech_samples

    def _append_speech(self, chunks):
        for chunk in chunks:
            self._speech_buffer.append(chunk)
            self._speech_samples += len(chunk)

    def _update_pre_buffer(self, chunk: np.ndarray):
        """プリバッファを更新し、保持時間を超えた古いチャンクを捨てる。"""
        self._pre_buffer.append(chunk)
        self._pre_buffer_samples += len(chunk)
        while self._pre_buffer_samples > self.pre_buffer_size:
            self._pre_buffer_samples -= len(self._pre_buffer.popleft())

    def _finish_segment(self) -> np.ndarray:
        # プリバッファと音声バッファを結合（末尾の無音は含めない）
        segment = np.concatenate(list(self._pre_buffer) + self._speech_buffer)
        self.is_speech_active = False
        self._pre_buffer.clear()
        self._pre_buffer_samples = 0
        self._speech_buffer = []
        self._speech_samples = 0
        self._pending_silence = []
        self._silence_seconds = 0.0
        return segment
