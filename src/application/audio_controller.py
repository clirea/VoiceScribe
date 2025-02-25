# application/audio_controller.py

import numpy as np
import sounddevice as sd
import queue
import time
import wave
import io
from datetime import datetime

from src.domain.vad_service import VADService
from src.domain.wakeword_detector import WakewordDetector
from src.infrastructure.speech_recognition_service import SpeechRecognitionService
from src.infrastructure.groq_recognition_service import GroqRecognitionService


class AudioController:
    """
    常時録音してVADで音声を判定し、無音が続いたタイミングで録音を切り出す。
    speech_recognitionでテキスト化し、ウェイクワードを検知したら音声データを返す。
    """

    def __init__(
        self,
        sample_rate=16000,
        channels=1,
        dtype=np.float32,
        block_size=512,
        silence_threshold=0.7,   # VADの閾値
        silence_duration=1.0,    # 無音判定に必要な継続秒数
        min_amplitude=0.01,      # ノイズ判定用の最低振幅
        pre_buffer_duration=0.5,  # 音声区間開始前のバッファ保持時間（秒）
        on_speech_detected=None,  # 音声検出時のコールバック関数
        device=None,             # 使用するマイクデバイスID
        device_name=None         # 使用するマイクデバイス名
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.block_size = block_size

        # マイクデバイス設定
        self.device = device
        # デバイス名が指定された場合、IDを検索
        if device_name is not None:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if device_name.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
                    self.device = i
                    print(f"マイク '{device_name}' を見つけました (ID: {i})")
                    break
            else:
                print(f"警告: '{device_name}'という名前のマイクが見つかりませんでした。デフォルトマイクを使用します。")

        self.audio_queue = queue.Queue()
        self.is_running = False
        self.stream = None

        # VAD・Wakeword・STTサービス
        self.vad_service = VADService(threshold=silence_threshold)
        self.wakeword_detector = WakewordDetector()
        self.stt_service = GroqRecognitionService(language="ja-JP")

        # 無音関連設定
        self.silence_duration = silence_duration
        self.min_amplitude = min_amplitude

        # 音声バッファ関連
        self.speech_buffer = []  # 音声区間のバッファ
        self.is_speech_active = False  # 現在音声区間かどうか
        
        # プリバッファ関連
        self.pre_buffer_size = int(pre_buffer_duration * sample_rate)
        self.pre_buffer = []

        # コールバック
        self.on_speech_detected = on_speech_detected

    def _audio_callback(self, indata, frames, time_info, status):
        if self.is_running:
            self.audio_queue.put(indata.copy())

    def start_listening(self):
        """音声入力ストリームを開始する"""
        print("AudioController: 録音開始")
        self.is_running = True
        
        # 使用するデバイスの情報を表示
        if self.device is not None:
            try:
                device_info = sd.query_devices(self.device)
                print(f"使用マイク: {device_info['name']} (ID: {self.device})")
            except Exception as e:
                print(f"マイク情報取得エラー: {e}")
                print("デフォルトマイクを使用します。")
                self.device = None
        else:
            print("デフォルトマイクを使用します。")
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.block_size,
            callback=self._audio_callback,
            device=self.device
        )
        self.stream.start()

    def stop_listening(self):
        """音声入力ストリームを停止する"""
        print("AudioController: 録音停止")
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def save_to_wav(self, audio_data: np.ndarray, filename: str):
        """NumPyの音声配列をWAVファイルに保存する"""
        if audio_data is None or audio_data.size == 0:
            return
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes((audio_data * 32767).astype('int16').tobytes())
        print(f"AudioController: 音声ファイル保存完了 -> {filename}")

    def update_pre_buffer(self, chunk: np.ndarray):
        """プリバッファを更新する"""
        self.pre_buffer.append(chunk)
        # プリバッファのサイズを制限
        total_samples = sum(len(c) for c in self.pre_buffer)
        while total_samples > self.pre_buffer_size:
            removed = self.pre_buffer.pop(0)
            total_samples -= len(removed)

    def process_speech_segment(self):
        """音声区間を処理して音声データとメタデータを返す"""
        if not self.speech_buffer:
            return None, None

        # プリバッファと音声バッファを結合
        audio_data = np.concatenate(self.pre_buffer + self.speech_buffer)
        
        # 音声認識実行
        recognized_text = self.stt_service.transcribe(audio_data, self.sample_rate)
        is_wake = self.wakeword_detector.detect(recognized_text)

        # メタデータ作成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            "timestamp": timestamp,
            "is_wake_word": is_wake,
            "text": recognized_text,
            "sample_rate": self.sample_rate,
            "channels": self.channels
        }
        
        # コールバック実行
        if self.on_speech_detected:
            self.on_speech_detected(audio_data, metadata)
        
        # バッファクリア
        self.speech_buffer = []
        
        return audio_data, metadata

    def run_forever(self):
        """常時ループで音声を取り続け、音声区間のみを返す"""
        silence_counter = 0.0
        print("AudioController: ループ開始。Ctrl+Cで終了してちょうだい。")

        while self.is_running:
            try:
                if not self.audio_queue.empty():
                    chunk = self.audio_queue.get()
                    flattened_chunk = chunk.flatten()

                    # VADで音声判定
                    is_speech = self.vad_service.is_speech(flattened_chunk, self.sample_rate)

                    if is_speech:
                        # 音声区間
                        silence_counter = 0
                        if not self.is_speech_active:
                            # 音声区間開始
                            self.is_speech_active = True
                            print("AudioController: 音声区間開始")
                        self.speech_buffer.append(flattened_chunk)
                    else:
                        # 無音区間
                        silence_counter += len(flattened_chunk) / self.sample_rate
                        # 音声区間外ならプリバッファを更新
                        if not self.is_speech_active:
                            self.update_pre_buffer(flattened_chunk)

                        if silence_counter >= self.silence_duration and self.is_speech_active:
                            # 十分な無音期間で音声区間終了
                            self.is_speech_active = False
                            print("AudioController: 音声区間終了")
                            yield self.process_speech_segment()
                            silence_counter = 0
                            # プリバッファをクリアして再開
                            self.pre_buffer = []

                else:
                    time.sleep(0.01)

            except KeyboardInterrupt:
                print("AudioController: Ctrl+Cを検知。停止するわ。")
                if self.is_speech_active:
                    # 終了時に処理中の音声があれば返す
                    yield self.process_speech_segment()
                break
            except Exception as e:
                print(f"AudioController: ループ中にエラー発生 {e}")
                if self.is_speech_active:
                    # エラー時に処理中の音声があれば返す
                    yield self.process_speech_segment()

        print("AudioController: run_forever終了")

    @staticmethod
    def list_audio_devices():
        """利用可能なオーディオデバイスの一覧を表示する"""
        print("利用可能なオーディオデバイス一覧:")
        print("-" * 70)
        devices = sd.query_devices()
        
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # 入力デバイス（マイク）のみ表示
                mark = '* ' if device.get('default_input') else '  '
                print(f"{mark}{i}: {device['name']} (入力チャンネル: {device['max_input_channels']})")
                input_devices.append((i, device['name']))
                
        print("-" * 70)
        print("* がデフォルトの入力デバイス")
        
        return input_devices

