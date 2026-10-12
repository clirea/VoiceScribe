# application/audio_controller.py

import logging
import queue
import time
from datetime import datetime

import numpy as np

from src.domain.errors import AudioInputError, ConfigurationError, RecognitionError
from src.domain.speech_segmenter import SpeechSegmenter
from src.domain.wakeword_detector import WakewordDetector

logger = logging.getLogger(__name__)


class AudioController:
    """
    常時録音してVADで音声を判定し、無音が続いたタイミングで録音を切り出す。
    音声認識サービスでテキスト化し、音声データとメタデータを返す。

    VAD・音声認識・マイク入力は外から受け取り、このクラスは流れの制御だけを担う。
    """

    QUEUE_POLL_SECONDS = 0.1
    PROBLEM_REPORT_INTERVAL_SECONDS = 5.0

    def __init__(
        self,
        vad_service,
        stt_service,
        audio_input,
        wakeword_detector=None,
        sample_rate=16000,
        block_size=512,
        silence_duration=1.0,      # 無音判定に必要な継続秒数
        min_amplitude=0.0,         # 最大振幅がこれ未満の区間は認識に送らない（0で無効）
        pre_buffer_duration=0.5,   # 音声区間開始前のバッファ保持時間（秒）
        max_speech_duration=30.0,  # 1区間の最大秒数（0以下で無制限）
        max_queue_seconds=60.0,    # 認識待ちの間に溜めておける音声の上限（秒）
        on_speech_detected=None,   # 音声検出時のコールバック関数
        on_recognition_error=None,  # 認識失敗時のコールバック関数
        device=None,               # 使用するマイクデバイスID
    ):
        required_block_size = vad_service.required_chunk_size(sample_rate)
        if block_size != required_block_size:
            raise ConfigurationError(
                f"{sample_rate}Hz ではブロックサイズを {required_block_size} にする必要があります"
                f"（指定値: {block_size}）。"
            )

        self.sample_rate = sample_rate
        self.channels = 1
        self.block_size = block_size
        self.device = device

        self.vad_service = vad_service
        self.stt_service = stt_service
        self.audio_input = audio_input
        self.wakeword_detector = wakeword_detector if wakeword_detector is not None else WakewordDetector()
        self.segmenter = SpeechSegmenter(
            sample_rate=sample_rate,
            silence_duration=silence_duration,
            pre_buffer_duration=pre_buffer_duration,
            max_speech_duration=max_speech_duration,
        )
        self.min_amplitude = min_amplitude

        max_queued_blocks = max(1, int(max_queue_seconds * sample_rate / block_size))
        self.audio_queue = queue.Queue(maxsize=max_queued_blocks)
        self.is_running = False
        self.stream = None

        # コールバックスレッドで数え、メインループ側で報告する
        self._dropped_chunks = 0
        self._input_overflows = 0
        self._last_problem_report = float("-inf")
        # 認識の途中で割り込まれた区間。flush() でもう一度だけ認識する
        self._interrupted_segment = None

        # コールバック
        self.on_speech_detected = on_speech_detected
        self.on_recognition_error = on_recognition_error

    def _audio_callback(self, indata, frames, time_info, status):
        if not self.is_running:
            return
        if status and getattr(status, "input_overflow", False):
            self._input_overflows += 1
        try:
            self.audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self._dropped_chunks += 1

    def start_listening(self):
        """音声入力ストリームを開始する"""
        if self.stream is not None:
            return
        logger.info("AudioController: 録音開始（%s）", self.audio_input.describe_device(self.device))
        self.vad_service.reset()
        self.is_running = True
        try:
            self.stream = self.audio_input.open_stream(
                sample_rate=self.sample_rate,
                block_size=self.block_size,
                device=self.device,
                callback=self._audio_callback,
            )
        except Exception:
            self.is_running = False
            raise

    def stop_listening(self):
        """音声入力ストリームを停止する（何度呼んでもよい）"""
        self.is_running = False
        stream, self.stream = self.stream, None
        if stream is None:
            return
        logger.info("AudioController: 録音停止")
        try:
            stream.stop()
        except Exception as e:
            logger.warning("AudioController: ストリーム停止時にエラー %s", e)
        finally:
            stream.close()

    def process_speech_segment(self, audio_data: np.ndarray):
        """音声区間を認識して (音声データ, メタデータ) を返す。出力するものが無ければNoneを返す。"""
        if audio_data is None or audio_data.size == 0:
            return None

        if float(np.max(np.abs(audio_data))) < self.min_amplitude:
            logger.info("AudioController: 振幅が小さいためノイズとして破棄")
            return None

        # 音声認識実行（Ctrl+Cなどで中断された場合に備えて、終わるまで区間を控えておく）
        self._interrupted_segment = audio_data
        try:
            recognized_text = self.stt_service.transcribe(audio_data, self.sample_rate)
        except RecognitionError as e:
            self._interrupted_segment = None
            if self.on_recognition_error:
                self.on_recognition_error(audio_data, e)
            if e.fatal:
                raise
            logger.error("AudioController: この発話は記録されません。%s", e)
            return None
        self._interrupted_segment = None

        if not recognized_text or not recognized_text.strip():
            return None

        # メタデータ作成
        metadata = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "is_wake_word": self.wakeword_detector.detect(recognized_text),
            "text": recognized_text,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }

        # コールバック実行
        if self.on_speech_detected:
            self.on_speech_detected(audio_data, metadata)

        return audio_data, metadata

    def run_forever(self):
        """常時ループで音声を取り続け、認識できた音声区間の (音声データ, メタデータ) を順に返す"""
        logger.info("AudioController: ループ開始。Ctrl+Cで終了します。")

        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=self.QUEUE_POLL_SECONDS)
            except queue.Empty:
                self._ensure_stream_is_active()
                continue

            self._report_input_problems()
            result = self._consume(chunk)
            if result is not None:
                yield result

        logger.info("AudioController: run_forever終了")

    def flush(self):
        """
        終了時に、認識を中断された区間・キューに残った音声・切り出し途中の発話を処理し、結果を順に返す。

        1件ずつ返すため、後の区間で失敗しても、それまでの結果は呼び出し元へ渡っている。
        中断された区間の再認識は1回だけで、再び中断された場合は呼び出し元が打ち切る。
        """
        interrupted_segment, self._interrupted_segment = self._interrupted_segment, None
        result = self.process_speech_segment(interrupted_segment)
        if result is not None:
            yield result

        while True:
            try:
                chunk = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            result = self._consume(chunk)
            if result is not None:
                yield result

        result = self.process_speech_segment(self.segmenter.flush())
        if result is not None:
            yield result

    def unprocessed_seconds(self) -> float:
        """認識に送れていない音声（中断された区間・キュー・切り出し途中）のおおよその秒数を返す。"""
        samples = self.audio_queue.qsize() * self.block_size + self.segmenter.buffered_samples
        if self._interrupted_segment is not None:
            samples += len(self._interrupted_segment)
        return samples / self.sample_rate

    def _consume(self, chunk):
        """チャンクを1つVADと区間切り出しに通し、区間が確定したら認識結果を返す。"""
        mono_chunk = np.asarray(chunk).reshape(-1)
        was_active = self.segmenter.is_speech_active

        # VADで音声判定
        is_speech = self.vad_service.is_speech(mono_chunk, self.sample_rate)
        segment = self.segmenter.feed(mono_chunk, is_speech)

        if is_speech and not was_active:
            logger.info("AudioController: 音声区間開始")
        if segment is None:
            return None
        logger.info("AudioController: 音声区間終了")
        return self.process_speech_segment(segment)

    def _ensure_stream_is_active(self):
        """音声が届かないとき、マイクが切断されていないか確かめる。"""
        if self.stream is not None and not self.stream.active:
            raise AudioInputError(
                "マイク入力が停止しました。マイクの接続を確認して再実行してください。"
            )

    def _report_input_problems(self):
        """取りこぼした音声があれば、一定間隔で警告する。"""
        if not (self._dropped_chunks or self._input_overflows):
            return
        now = time.monotonic()
        if now - self._last_problem_report < self.PROBLEM_REPORT_INTERVAL_SECONDS:
            return
        dropped_seconds = self._dropped_chunks * self.block_size / self.sample_rate
        logger.warning(
            "AudioController: 音声を取りこぼしました（キュー溢れ: 約%.1f秒, 入力オーバーフロー: %d回）",
            dropped_seconds,
            self._input_overflows,
        )
        self._dropped_chunks = 0
        self._input_overflows = 0
        self._last_problem_report = now
