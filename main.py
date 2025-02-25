# main.py

import os
import argparse
from datetime import datetime
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

from src.application.audio_controller import AudioController

def append_to_text_file(text, metadata, output_file="output.txt"):
    """音声認識結果をテキストファイルに追記する"""
    if text is None or metadata is None:
        return

    # 出力ディレクトリがなければ作成
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # タイムスタンプと認識テキストを追記
    timestamp = metadata["timestamp"]
    prefix = "ウェイクワード" if metadata["is_wake_word"] else "音声認識"
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {prefix}: {text}\n")

    if metadata["is_wake_word"]:
        print(f"ウェイクワード検出: {text}")
    else:
        print(f"音声認識結果: {text}")

def on_speech_detected(audio_data, metadata):
    """音声検出時のコールバック関数"""
    print(f"音声検出: {metadata['text']}")

def main(output_file="output.txt", device_id=None, device_name=None, list_devices=False, recognition_service="google"):
    # デバイス一覧表示モード
    if list_devices:
        AudioController.list_audio_devices()
        return
    
    # コントローラを初期化
    controller = AudioController(
        sample_rate=16000,
        channels=1,
        block_size=512,
        silence_threshold=0.5,  # VADの発話閾値
        silence_duration=1.0,   # これだけ連続で無音なら切り出し
        min_amplitude=0.01,     # これ以下の振幅はノイズ扱い
        on_speech_detected=on_speech_detected,  # コールバック関数を設定
        device=device_id,       # マイクデバイスID
        device_name=device_name, # マイクデバイス名
        recognition_service=recognition_service  # 音声認識サービス
    )
    
    # 録音開始
    controller.start_listening()
    try:
        # 常時ループで音声データを取得
        for audio_data, metadata in controller.run_forever():
            if audio_data is not None and metadata is not None:
                append_to_text_file(metadata['text'], metadata, output_file)
    finally:
        # 終了時には確実に停止
        controller.stop_listening()

if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='音声認識アプリケーション')
    parser.add_argument('-o', '--output', type=str, default='output.txt',
                        help='出力ファイルのパス (デフォルト: output.txt)')
    parser.add_argument('-d', '--device', type=int, help='使用するマイクデバイスのID')
    parser.add_argument('-n', '--name', type=str, help='使用するマイクデバイスの名前（部分一致）')
    parser.add_argument('-l', '--list', action='store_true', help='利用可能なマイクデバイスの一覧を表示')
    parser.add_argument('-s', '--service', type=str, default='google', choices=['google', 'openai', 'groq'],
                        help='使用する音声認識サービス (デフォルト: google)')
    
    args = parser.parse_args()
    
    main(
        output_file=args.output,
        device_id=args.device,
        device_name=args.name,
        list_devices=args.list,
        recognition_service=args.service
    )
