#!/usr/bin/env python
# mic_check.py

import sys
from datetime import datetime

import numpy as np

from src.domain.errors import VoiceScribeError
from src.infrastructure.audio_devices import SoundDeviceAudioInput, format_input_devices
from src.infrastructure.wav_encoder import write_wav


def list_audio_devices(audio_input):
    """利用可能なオーディオデバイスの一覧を表示する"""
    input_devices = audio_input.list_input_devices()
    print("\n".join(format_input_devices(input_devices)))
    return [(device.index, device.name) for device in input_devices]


def test_microphone(audio_input, device_id=None, device_name=None, duration=3, sample_rate=16000, sleep=None):
    """指定されたマイクでテスト録音を行い、(音声データ, ファイル名) を返す"""
    device = audio_input.resolve_device(device_id=device_id, device_name=device_name)
    print(f"選択したマイク: {audio_input.describe_device(device)}")

    # 録音データを格納する配列
    recording = []

    # コールバック関数
    def callback(indata, frames, time_info, status):
        if status:
            print(f"ステータス: {status}")
        recording.append(indata[:, 0].copy())

    # 録音開始
    print(f"{duration}秒間の録音を開始します...")
    stream = audio_input.open_stream(
        sample_rate=sample_rate,
        block_size=0,
        device=device,
        callback=callback,
    )
    try:
        (sleep or _sleep)(int(duration * 1000))
    finally:
        stream.stop()
        stream.close()

    # 録音データを結合
    if not recording:
        print("録音データがありません。")
        return None

    audio_data = np.concatenate(recording)

    # WAVファイルに保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mic_test_{timestamp}.wav"
    write_wav(filename, audio_data, sample_rate)

    print(f"録音完了: {filename} に保存しました。(最大振幅: {float(np.max(np.abs(audio_data))):.3f})")
    return audio_data, filename


def _sleep(milliseconds):
    import sounddevice as sd

    sd.sleep(milliseconds)


def run_menu(audio_input, read_input=input):
    """対話メニューを終了が選ばれるまで繰り返す"""
    while True:
        print("\n操作を選択してください:")
        print("1: マイクをIDで選択してテスト録音")
        print("2: マイク名で検索してテスト録音")
        print("3: デフォルトマイクでテスト録音")
        print("4: 終了")

        choice = read_input("> ").strip()

        try:
            if choice == "1":
                device_id = int(read_input("使用するマイクのIDを入力してください: "))
                test_microphone(audio_input, device_id=device_id)
            elif choice == "2":
                device_name = read_input("検索するマイク名を入力してください: ")
                test_microphone(audio_input, device_name=device_name)
            elif choice == "3":
                test_microphone(audio_input)
            elif choice == "4":
                print("終了します。")
                break
            else:
                print("無効な選択です。もう一度お試しください。")
        except ValueError:
            print("有効な数値を入力してください。")
        except VoiceScribeError as e:
            print(f"エラー: {e}")


def main():
    """メイン関数"""
    print("マイクチェックツール")
    print("=" * 50)

    try:
        audio_input = SoundDeviceAudioInput()
        # マイク一覧を表示
        list_audio_devices(audio_input)
        run_menu(audio_input)
    except VoiceScribeError as e:
        print(f"エラー: {e}")
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\n終了します。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
