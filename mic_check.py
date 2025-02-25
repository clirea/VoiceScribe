#!/usr/bin/env python
# mic_check.py

import sounddevice as sd
import numpy as np
import time
import wave
import os
from datetime import datetime

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

def test_microphone(device_id=None, device_name=None, duration=3, sample_rate=16000, channels=1):
    """指定されたマイクでテスト録音を行う"""
    # デバイス名が指定された場合、IDを検索
    if device_name is not None:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device_name.lower() in device['name'].lower() and device['max_input_channels'] > 0:
                device_id = i
                break
        else:
            print(f"エラー: '{device_name}'という名前のマイクが見つかりませんでした。")
            return None
    
    if device_id is None:
        print("デフォルトのマイクを使用します。")
    else:
        device_info = sd.query_devices(device_id)
        print(f"選択したマイク: {device_info['name']} (ID: {device_id})")
    
    # 録音データを格納する配列
    recording = []
    
    # コールバック関数
    def callback(indata, frames, time, status):
        if status:
            print(f"ステータス: {status}")
        recording.append(indata.copy())
    
    # 録音開始
    print(f"{duration}秒間の録音を開始します...")
    with sd.InputStream(samplerate=sample_rate, channels=channels, 
                       callback=callback, device=device_id):
        sd.sleep(int(duration * 1000))
    
    # 録音データを結合
    if not recording:
        print("録音データがありません。")
        return None
    
    audio_data = np.concatenate(recording)
    
    # WAVファイルに保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mic_test_{timestamp}.wav"
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes((audio_data * 32767).astype('int16').tobytes())
    
    print(f"録音完了: {filename} に保存しました。")
    return audio_data, filename

def main():
    """メイン関数"""
    print("マイクチェックツール")
    print("=" * 50)
    
    # マイク一覧を表示
    input_devices = list_audio_devices()
    
    while True:
        print("\n操作を選択してください:")
        print("1: マイクをIDで選択してテスト録音")
        print("2: マイク名で検索してテスト録音")
        print("3: デフォルトマイクでテスト録音")
        print("4: 終了")
        
        choice = input("> ")
        
        if choice == "1":
            device_id = input("使用するマイクのIDを入力してください: ")
            try:
                device_id = int(device_id)
                test_microphone(device_id=device_id)
            except ValueError:
                print("有効な数値を入力してください。")
            except sd.PortAudioError as e:
                print(f"エラー: {e}")
                
        elif choice == "2":
            device_name = input("検索するマイク名を入力してください: ")
            test_microphone(device_name=device_name)
            
        elif choice == "3":
            test_microphone()
            
        elif choice == "4":
            print("終了します。")
            break
            
        else:
            print("無効な選択です。もう一度お試しください。")

if __name__ == "__main__":
    main() 