# main.py

import argparse
import logging
import math
import os
import sys
from datetime import datetime

from src.domain.errors import AudioInputError, ConfigurationError, TranscriptWriteError, VoiceScribeError
from src.infrastructure.recognition_service_factory import SERVICE_NAMES

logger = logging.getLogger("voicescribe")

EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_CONFIGURATION_ERROR = 2


def build_parser():
    """コマンドライン引数の設定"""
    parser = argparse.ArgumentParser(description='音声認識アプリケーション')
    parser.add_argument('-o', '--output', type=str, default='output.txt',
                        help='出力ファイルのパス (デフォルト: output.txt)')
    parser.add_argument('-d', '--device', type=int, help='使用するマイクデバイスのID')
    parser.add_argument('-n', '--name', type=str, help='使用するマイクデバイスの名前（部分一致）')
    parser.add_argument('-l', '--list', action='store_true', help='利用可能なマイクデバイスの一覧を表示')
    parser.add_argument('-s', '--service', type=str, default='google', choices=SERVICE_NAMES,
                        help='使用する音声認識サービス (デフォルト: google)')
    parser.add_argument('--language', type=str, default=None,
                        help='認識言語 (例: ja-JP)。未指定時は google が ja-JP、openai / groq は自動判定')
    parser.add_argument('--model', type=str, default=None,
                        help='openai / groq で使うモデル名 (デフォルト: whisper-1 / whisper-large-v3-turbo)')
    parser.add_argument('--max-segment-seconds', type=float, default=30.0,
                        help='無音が来なくても発話を区切る秒数。0で無制限 (デフォルト: 30)')
    parser.add_argument('--save-failed-audio', type=str, default=None, metavar='DIR',
                        help='音声認識に失敗した発話をWAVとして保存するディレクトリ (デフォルト: 保存しない)')
    return parser


def validate_args(args):
    """引数の型だけでは弾けない値（負数・inf・nan）を設定の不備として扱う。"""
    seconds = args.max_segment_seconds
    if not math.isfinite(seconds) or seconds < 0:
        raise ConfigurationError(
            f"--max-segment-seconds には0以上の有限の秒数を指定してください（指定値: {seconds}）。"
        )


def configure_console():
    """ログ出力を設定し、コンソールの文字コードで表せない文字が来ても止まらないようにする。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)


def report_transcript(writer, metadata):
    """認識結果をファイルへ追記し、コンソールにも表示する。"""
    try:
        written = writer.append(metadata)
    except TranscriptWriteError:
        # 保存できなかった結果が失われないよう、コンソールには残す
        print(f"保存できなかった認識結果: {metadata['text']}")
        raise
    if not written:
        return
    if metadata["is_wake_word"]:
        print(f"ウェイクワード検出: {metadata['text']}")
    else:
        print(f"音声認識結果: {metadata['text']}")


def create_failed_audio_saver(directory, sample_rate):
    """認識に失敗した発話をWAVで残すコールバックを返す。"""
    from src.infrastructure.wav_encoder import write_wav

    def save_failed_audio(audio_data, error):
        path = os.path.join(directory, f"failed_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav")
        try:
            os.makedirs(directory, exist_ok=True)
            write_wav(path, audio_data, sample_rate)
        except OSError as e:
            logger.error("認識に失敗した音声を保存できませんでした: %s", e)
            return
        logger.warning("認識に失敗した音声を保存しました -> %s", path)

    return save_failed_audio


def transcribe_loop(controller, writer):
    """
    録音を開始し、停止されるまで認識結果を書き出す。

    Ctrl+Cやマイクの停止では、認識中だった発話・未処理の音声・話している途中の発話も書き出す。
    保存の失敗や継続不能な認識失敗では、終了時の書き出し中であっても、録音を止めて例外を呼び出し元へ伝える。
    """
    try:
        controller.start_listening()
        # 常時ループで音声データを取得
        for _audio_data, metadata in controller.run_forever():
            report_transcript(writer, metadata)
    except KeyboardInterrupt:
        logger.info("Ctrl+Cを検知。処理中の発話を書き出して停止します（もう一度押すと書き出さずに終了）。")
        controller.stop_listening()
        flush_pending_speech(controller, writer)
    except AudioInputError:
        controller.stop_listening()
        try:
            flush_pending_speech(controller, writer)
        except VoiceScribeError as flush_error:
            # 終了の原因はマイクの停止なので、書き出しの失敗は表示にとどめて元の例外を伝える
            logger.error("終了時の書き出しに失敗しました: %s", flush_error)
        raise
    finally:
        # 終了時には確実に停止
        controller.stop_listening()


def flush_pending_speech(controller, writer):
    """
    認識できていない音声を認識して書き出す。

    保存の失敗や継続不能な認識失敗は、書き出せなかった音声のおおよその長さを知らせたうえで呼び出し元へ伝える。
    再度のCtrl+Cでは再試行せずに打ち切り、長さを知らせて終了を妨げない。
    """
    try:
        for _audio_data, metadata in controller.flush():
            report_transcript(writer, metadata)
    except KeyboardInterrupt:
        logger.warning("書き出しを中断しました。%s", describe_lost_audio(controller))
    except VoiceScribeError:
        logger.error("終了時の書き出しを完了できませんでした。%s", describe_lost_audio(controller))
        raise


def describe_lost_audio(controller):
    return f"約{controller.unprocessed_seconds():.1f}秒の音声は文字起こしされていません。"


def run(args, env=None):
    """引数に従って依存を組み立て、一覧表示または文字起こしを実行する。"""
    env = os.environ if env is None else env

    from src.infrastructure.audio_devices import SoundDeviceAudioInput, format_input_devices

    audio_input = SoundDeviceAudioInput()

    # デバイス一覧表示モード
    if args.list:
        print("\n".join(format_input_devices(audio_input.list_input_devices())))
        return

    from src.application.audio_controller import AudioController
    from src.domain.vad_service import VADService
    from src.infrastructure.recognition_service_factory import create_recognition_service
    from src.infrastructure.transcript_writer import TranscriptWriter

    # 録音やモデル読み込みの前に、設定の不備を検出する
    stt_service = create_recognition_service(args.service, env, language=args.language, model=args.model)
    device = audio_input.resolve_device(device_id=args.device, device_name=args.name)
    writer = TranscriptWriter(args.output)
    writer.prepare()

    sample_rate = 16000
    on_recognition_error = None
    if args.save_failed_audio:
        on_recognition_error = create_failed_audio_saver(args.save_failed_audio, sample_rate)

    # コントローラを初期化
    controller = AudioController(
        vad_service=VADService(threshold=0.5),  # VADの発話閾値
        stt_service=stt_service,
        audio_input=audio_input,
        sample_rate=sample_rate,
        block_size=512,
        silence_duration=1.0,   # これだけ連続で無音なら切り出し
        max_speech_duration=args.max_segment_seconds,
        on_recognition_error=on_recognition_error,
        device=device,          # マイクデバイスID
    )

    transcribe_loop(controller, writer)


def main(argv=None):
    # --help や引数エラーの表示にも効くよう、引数解析より先に構成する
    configure_console()
    args = build_parser().parse_args(argv)

    # .envファイルを読み込む
    from dotenv import load_dotenv
    load_dotenv()

    try:
        validate_args(args)
        run(args)
    except ConfigurationError as e:
        logger.error("設定エラー: %s", e)
        return EXIT_CONFIGURATION_ERROR
    except VoiceScribeError as e:
        logger.error("エラー: %s", e)
        return EXIT_RUNTIME_ERROR
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
