# tests/test_main.py

import logging
import os
import subprocess
import sys
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import main as app
from src.application.audio_controller import AudioController
from src.domain.errors import (
    AudioInputError,
    ConfigurationError,
    RecognitionError,
    TranscriptWriteError,
    VoiceScribeError,
)
from src.infrastructure.audio_devices import InputDevice
from src.infrastructure.transcript_writer import TranscriptWriter


def metadata(text="こんにちは", is_wake_word=False):
    return {"timestamp": "20261012_101530", "is_wake_word": is_wake_word, "text": text}


def parse(*argv):
    return app.build_parser().parse_args(list(argv))


# --- build_parser ---


def test正常系_build_parserメソッドが既定の音声認識サービスをgoogleにすること():
    assert parse().service == "google"


def test正常系_build_parserメソッドが既定の出力先をoutput_txtにすること():
    assert parse().output == "output.txt"


def test正常系_build_parserメソッドが既定の最大区間長を30秒にすること():
    assert parse().max_segment_seconds == 30.0


def test正常系_build_parserメソッドが既定で言語を指定しないこと():
    assert parse().language is None


def test正常系_build_parserメソッドが既定で失敗音声を保存しないこと():
    assert parse().save_failed_audio is None


def test正常系_build_parserメソッドが従来の短縮オプションを受け付けること():
    args = parse("-o", "notes.txt", "-d", "2", "-n", "Headset", "-s", "groq")
    assert (args.output, args.device, args.name, args.service) == ("notes.txt", 2, "Headset", "groq")


def test異常系_build_parserメソッドが未対応のサービス名でSystemExitを送出すること():
    with pytest.raises(SystemExit):
        parse("-s", "azure")


def test正常系_build_parserメソッドが最大区間長の0を無制限の指定として受け付けること():
    assert parse("--max-segment-seconds", "0").max_segment_seconds == 0.0


def test異常系_build_parserメソッドが数値でない最大区間長でSystemExitを送出すること():
    with pytest.raises(SystemExit):
        parse("--max-segment-seconds=abc")


# --- validate_args ---


@pytest.mark.parametrize("value", ["0", "0.5", "30"])
def test正常系_validate_argsメソッドが0以上の有限の最大区間長を受け付けること(value):
    assert app.validate_args(parse(f"--max-segment-seconds={value}")) is None


@pytest.mark.parametrize("value", ["-1", "-0.1", "inf", "-inf", "nan"])
def test異常系_validate_argsメソッドが有限の0以上でない最大区間長でConfigurationErrorを送出すること(value):
    with pytest.raises(ConfigurationError, match="0以上の有限の秒数"):
        app.validate_args(parse(f"--max-segment-seconds={value}"))


# --- main ---


@pytest.fixture
def main_dependencies():
    with (
        patch.object(app, "run") as run,
        patch.object(app, "configure_console"),
        patch("dotenv.load_dotenv") as load_dotenv,
    ):
        yield {"run": run, "load_dotenv": load_dotenv}


def test正常系_mainメソッドが正常終了で0を返すこと(main_dependencies):
    assert app.main([]) == 0


def test正常系_mainメソッドがenvファイルを読み込むこと(main_dependencies):
    app.main([])
    main_dependencies["load_dotenv"].assert_called_once_with()


def test正常系_mainメソッドが解析した引数をrunへ渡すこと(main_dependencies):
    app.main(["-s", "openai"])
    assert main_dependencies["run"].call_args.args[0].service == "openai"


def test異常系_mainメソッドが設定エラーで終了コード2を返すこと(main_dependencies):
    main_dependencies["run"].side_effect = ConfigurationError("APIキーがありません")
    assert app.main([]) == 2


@pytest.mark.parametrize("value", ["-1", "inf", "nan"])
def test異常系_mainメソッドが不正な最大区間長で終了コード2を返すこと(main_dependencies, value):
    assert app.main([f"--max-segment-seconds={value}"]) == 2


def test異常系_mainメソッドが不正な最大区間長では実行を始めないこと(main_dependencies):
    app.main(["--max-segment-seconds=nan"])
    main_dependencies["run"].assert_not_called()


def test異常系_mainメソッドが不正な最大区間長を設定エラーとして表示すること(main_dependencies, caplog):
    with caplog.at_level(logging.ERROR, logger="voicescribe"):
        app.main(["--max-segment-seconds=-1"])
    assert "設定エラー: --max-segment-seconds には0以上の有限の秒数を指定してください" in caplog.text


def test異常系_mainメソッドが実行時エラーで終了コード1を返すこと(main_dependencies):
    main_dependencies["run"].side_effect = AudioInputError("マイク入力が停止しました")
    assert app.main([]) == 1


def test正常系_mainメソッドが引数解析より先にコンソールを構成すること():
    calls = []
    args = parse()
    parser = MagicMock()
    parser.parse_args.side_effect = lambda argv: calls.append("parse_args") or args
    with (
        patch.object(app, "run"),
        patch.object(app, "configure_console", side_effect=lambda: calls.append("configure_console")),
        patch.object(app, "build_parser", return_value=parser),
        patch("dotenv.load_dotenv"),
    ):
        app.main([])
    assert calls == ["configure_console", "parse_args"]


def run_cli_in_cp1252_console(*argv):
    """日本語を表せない文字コードのコンソール（英語版Windowsの既定）で main.py を子プロセスとして実行する"""
    env = {**os.environ, "PYTHONIOENCODING": "cp1252:strict"}
    env.pop("PYTHONUTF8", None)
    return subprocess.run(
        [sys.executable, app.__file__, *argv],
        env=env,
        capture_output=True,
        timeout=60,
        check=False,
    )


@pytest.fixture(scope="module")
def cp1252_help_result():
    return run_cli_in_cp1252_console("--help")


@pytest.fixture(scope="module")
def cp1252_invalid_argument_result():
    return run_cli_in_cp1252_console("--no-such-option")


def test正常系_mainメソッドが日本語を表せないコンソールでもhelpを終了コード0で表示すること(cp1252_help_result):
    assert cp1252_help_result.returncode == 0


def test正常系_mainメソッドが日本語を表せないコンソールでもhelpに使い方を出力すること(cp1252_help_result):
    assert cp1252_help_result.stdout.decode("cp1252").startswith("usage: main.py")


def test正常系_mainメソッドが日本語を表せないコンソールでもhelpに全オプションを出力すること(cp1252_help_result):
    assert "--save-failed-audio DIR" in cp1252_help_result.stdout.decode("cp1252")


def test正常系_mainメソッドが日本語を表せないコンソールでもhelpでエラーを出力しないこと(cp1252_help_result):
    assert cp1252_help_result.stderr == b""


def test異常系_mainメソッドが日本語を表せないコンソールでも不正な引数で終了コード2を返すこと(
    cp1252_invalid_argument_result,
):
    assert cp1252_invalid_argument_result.returncode == 2


def test異常系_mainメソッドが日本語を表せないコンソールでも不正な引数を知らせること(cp1252_invalid_argument_result):
    assert "unrecognized arguments: --no-such-option" in cp1252_invalid_argument_result.stderr.decode("cp1252")


# --- run ---


@pytest.fixture
def run_dependencies():
    with (
        patch("src.infrastructure.audio_devices.SoundDeviceAudioInput") as audio_input_class,
        patch("src.infrastructure.recognition_service_factory.create_recognition_service") as create_service,
        patch("src.domain.vad_service.VADService") as vad_service_class,
        patch("src.application.audio_controller.AudioController") as controller_class,
        patch("src.infrastructure.transcript_writer.TranscriptWriter") as writer_class,
        patch.object(app, "transcribe_loop") as transcribe_loop,
    ):
        audio_input = audio_input_class.return_value
        audio_input.resolve_device.return_value = 2
        audio_input.list_input_devices.return_value = [
            InputDevice(index=1, name="Built-in Microphone", channels=1, host_api="Core Audio", is_default=True)
        ]
        yield {
            "audio_input": audio_input,
            "create_service": create_service,
            "vad_service_class": vad_service_class,
            "controller_class": controller_class,
            "writer_class": writer_class,
            "transcribe_loop": transcribe_loop,
        }


def test正常系_runメソッドが一覧表示の指定でマイクの一覧を表示すること(run_dependencies, capsys):
    app.run(parse("-l"), env={})
    assert "* 1: Built-in Microphone" in capsys.readouterr().out


def test正常系_runメソッドが一覧表示の指定では録音を始めないこと(run_dependencies):
    app.run(parse("-l"), env={})
    run_dependencies["transcribe_loop"].assert_not_called()


def test正常系_runメソッドが指定されたサービスと言語とモデルで音声認識を組み立てること(run_dependencies):
    env = {"OPENAI_API_KEY": "sk-test"}
    app.run(parse("-s", "openai", "--language", "en-US", "--model", "custom-model"), env=env)
    run_dependencies["create_service"].assert_called_once_with("openai", env, language="en-US", model="custom-model")


def test正常系_runメソッドがマイクの指定を解決すること(run_dependencies):
    app.run(parse("-d", "3", "-n", "Headset"), env={})
    run_dependencies["audio_input"].resolve_device.assert_called_once_with(device_id=3, device_name="Headset")


def test正常系_runメソッドが解決したマイクをコントローラへ渡すこと(run_dependencies):
    app.run(parse(), env={})
    assert run_dependencies["controller_class"].call_args.kwargs["device"] == 2


def test正常系_runメソッドが生成した音声認識サービスをコントローラへ渡すこと(run_dependencies):
    app.run(parse(), env={})
    stt_service = run_dependencies["controller_class"].call_args.kwargs["stt_service"]
    assert stt_service is run_dependencies["create_service"].return_value


def test正常系_runメソッドが最大区間長をコントローラへ渡すこと(run_dependencies):
    app.run(parse("--max-segment-seconds", "12"), env={})
    assert run_dependencies["controller_class"].call_args.kwargs["max_speech_duration"] == 12.0


def test正常系_runメソッドが既定では認識失敗時のコールバックを設定しないこと(run_dependencies):
    app.run(parse(), env={})
    assert run_dependencies["controller_class"].call_args.kwargs["on_recognition_error"] is None


def test正常系_runメソッドが失敗音声の保存先指定でコールバックを設定すること(run_dependencies, tmp_path):
    app.run(parse("--save-failed-audio", str(tmp_path)), env={})
    assert callable(run_dependencies["controller_class"].call_args.kwargs["on_recognition_error"])


def test正常系_runメソッドが録音前に出力先を確認すること(run_dependencies):
    app.run(parse("-o", "notes.txt"), env={})
    run_dependencies["writer_class"].return_value.prepare.assert_called_once_with()


def test正常系_runメソッドが指定された出力先でライターを生成すること(run_dependencies):
    app.run(parse("-o", "notes.txt"), env={})
    run_dependencies["writer_class"].assert_called_once_with("notes.txt")


def test正常系_runメソッドがコントローラとライターで文字起こしを開始すること(run_dependencies):
    app.run(parse(), env={})
    run_dependencies["transcribe_loop"].assert_called_once_with(
        run_dependencies["controller_class"].return_value, run_dependencies["writer_class"].return_value
    )


def test異常系_runメソッドが設定エラーのときVADモデルを読み込む前に中断すること(run_dependencies):
    run_dependencies["create_service"].side_effect = ConfigurationError("APIキーがありません")
    with pytest.raises(ConfigurationError):
        app.run(parse("-s", "openai"), env={})
    run_dependencies["vad_service_class"].assert_not_called()


def test異常系_runメソッドがマイクが見つからないとき録音を始めないこと(run_dependencies):
    run_dependencies["audio_input"].resolve_device.side_effect = ConfigurationError("マイクが見つかりません")
    with pytest.raises(ConfigurationError):
        app.run(parse("-n", "存在しないマイク"), env={})
    run_dependencies["transcribe_loop"].assert_not_called()


# --- transcribe_loop / flush_pending_speech ---


@pytest.fixture
def controller():
    controller = MagicMock()
    controller.run_forever.return_value = iter([])
    controller.flush.return_value = []
    controller.unprocessed_seconds.return_value = 0.0
    return controller


@pytest.fixture
def writer():
    writer = MagicMock()
    writer.append.return_value = True
    return writer


def interrupted_after(*results):
    """結果を返し終えたところでCtrl+Cが押された状況を模す"""
    yield from results
    raise KeyboardInterrupt


def test正常系_transcribe_loopメソッドが録音を開始すること(controller, writer):
    app.transcribe_loop(controller, writer)
    controller.start_listening.assert_called_once_with()


def test正常系_transcribe_loopメソッドが認識結果を順に書き出すこと(controller, writer):
    controller.run_forever.return_value = iter([(None, metadata("一つ目")), (None, metadata("二つ目"))])
    app.transcribe_loop(controller, writer)
    assert [call.args[0]["text"] for call in writer.append.call_args_list] == ["一つ目", "二つ目"]


def test正常系_transcribe_loopメソッドが終了時に録音を停止すること(controller, writer):
    app.transcribe_loop(controller, writer)
    controller.stop_listening.assert_called()


def test正常系_transcribe_loopメソッドがCtrl_Cを例外にせず終了すること(controller, writer):
    controller.run_forever.return_value = interrupted_after()
    assert app.transcribe_loop(controller, writer) is None


def test正常系_transcribe_loopメソッドがCtrl_C時に途中の発話を書き出すこと(controller, writer):
    controller.run_forever.return_value = interrupted_after()
    controller.flush.return_value = [(None, metadata("途中の発話"))]
    app.transcribe_loop(controller, writer)
    assert writer.append.call_args.args[0]["text"] == "途中の発話"


def test正常系_transcribe_loopメソッドがCtrl_C時に録音を止めてから途中の発話を処理すること(controller, writer):
    calls = []
    controller.run_forever.return_value = interrupted_after()
    controller.stop_listening.side_effect = lambda: calls.append("stop")
    controller.flush.side_effect = lambda: calls.append("flush") or []
    app.transcribe_loop(controller, writer)
    assert calls[:2] == ["stop", "flush"]


def test異常系_transcribe_loopメソッドがマイク停止でAudioInputErrorを送出すること(controller, writer):
    controller.run_forever.side_effect = AudioInputError("マイク入力が停止しました")
    with pytest.raises(AudioInputError):
        app.transcribe_loop(controller, writer)


def test異常系_transcribe_loopメソッドがマイク停止時にも途中の発話を書き出すこと(controller, writer):
    controller.run_forever.side_effect = AudioInputError("マイク入力が停止しました")
    controller.flush.return_value = [(None, metadata("途中の発話"))]
    with pytest.raises(AudioInputError):
        app.transcribe_loop(controller, writer)
    assert writer.append.call_args.args[0]["text"] == "途中の発話"


def test異常系_transcribe_loopメソッドが継続不能な認識失敗でも録音を停止すること(controller, writer):
    controller.run_forever.side_effect = RecognitionError("認証失敗", fatal=True)
    with pytest.raises(RecognitionError):
        app.transcribe_loop(controller, writer)
    controller.stop_listening.assert_called()


def test異常系_transcribe_loopメソッドが録音開始の失敗でも停止処理を呼ぶこと(controller, writer):
    controller.start_listening.side_effect = AudioInputError("マイクを開けません")
    with pytest.raises(AudioInputError):
        app.transcribe_loop(controller, writer)
    controller.stop_listening.assert_called()


def test異常系_flush_pending_speechメソッドが継続不能な認識失敗でRecognitionErrorを送出すること(controller, writer):
    controller.flush.side_effect = RecognitionError("認証失敗", fatal=True)
    with pytest.raises(RecognitionError, match="認証失敗"):
        app.flush_pending_speech(controller, writer)


def test異常系_flush_pending_speechメソッドが継続不能な認識失敗で書き出せなかった音声の長さを知らせること(
    controller, writer, caplog
):
    controller.flush.side_effect = RecognitionError("認証失敗", fatal=True)
    controller.unprocessed_seconds.return_value = 2.5
    with caplog.at_level(logging.ERROR, logger="voicescribe"), pytest.raises(RecognitionError):
        app.flush_pending_speech(controller, writer)
    assert "約2.5秒の音声は文字起こしされていません" in caplog.text


def test異常系_flush_pending_speechメソッドが再度のCtrl_Cを終了の妨げにしないこと(controller, writer):
    controller.flush.side_effect = KeyboardInterrupt
    assert app.flush_pending_speech(controller, writer) is None


def test異常系_flush_pending_speechメソッドが再度のCtrl_Cで書き出せなかった音声の長さを知らせること(
    controller, writer, caplog
):
    controller.flush.side_effect = KeyboardInterrupt
    controller.unprocessed_seconds.return_value = 2.5
    with caplog.at_level(logging.WARNING, logger="voicescribe"):
        app.flush_pending_speech(controller, writer)
    assert "約2.5秒の音声は文字起こしされていません" in caplog.text


def test異常系_flush_pending_speechメソッドが保存の失敗でTranscriptWriteErrorを送出すること(controller, writer):
    controller.flush.return_value = [(None, metadata("途中の発話"))]
    writer.append.side_effect = TranscriptWriteError("認識結果を保存できませんでした")
    with pytest.raises(TranscriptWriteError, match="認識結果を保存できませんでした"):
        app.flush_pending_speech(controller, writer)


def test異常系_flush_pending_speechメソッドが保存の失敗で書き出せなかった音声の長さを知らせること(
    controller, writer, caplog
):
    controller.flush.return_value = [(None, metadata("途中の発話"))]
    controller.unprocessed_seconds.return_value = 1.5
    writer.append.side_effect = TranscriptWriteError("認識結果を保存できませんでした")
    with caplog.at_level(logging.ERROR, logger="voicescribe"), pytest.raises(TranscriptWriteError):
        app.flush_pending_speech(controller, writer)
    assert "約1.5秒の音声は文字起こしされていません" in caplog.text


def test異常系_transcribe_loopメソッドがCtrl_C後の保存失敗でTranscriptWriteErrorを送出すること(controller, writer):
    controller.run_forever.return_value = interrupted_after()
    controller.flush.return_value = [(None, metadata("途中の発話"))]
    writer.append.side_effect = TranscriptWriteError("認識結果を保存できませんでした")
    with pytest.raises(TranscriptWriteError):
        app.transcribe_loop(controller, writer)


def test異常系_transcribe_loopメソッドがCtrl_C後の継続不能な認識失敗でRecognitionErrorを送出すること(
    controller, writer
):
    controller.run_forever.return_value = interrupted_after()
    controller.flush.side_effect = RecognitionError("認証失敗", fatal=True)
    with pytest.raises(RecognitionError):
        app.transcribe_loop(controller, writer)


def test異常系_transcribe_loopメソッドがマイク停止後の書き出し失敗でも元のAudioInputErrorを送出すること(
    controller, writer
):
    controller.run_forever.side_effect = AudioInputError("マイク入力が停止しました")
    controller.flush.side_effect = RecognitionError("認証失敗", fatal=True)
    with pytest.raises(AudioInputError, match="マイク入力が停止しました"):
        app.transcribe_loop(controller, writer)


def test異常系_transcribe_loopメソッドがマイク停止後の書き出し失敗を利用者へ知らせること(
    controller, writer, caplog
):
    controller.run_forever.side_effect = AudioInputError("マイク入力が停止しました")
    controller.flush.side_effect = RecognitionError("認証失敗", fatal=True)
    with caplog.at_level(logging.ERROR, logger="voicescribe"), pytest.raises(AudioInputError):
        app.transcribe_loop(controller, writer)
    assert "終了時の書き出しに失敗しました: 認証失敗" in caplog.text


def test異常系_transcribe_loopメソッドが録音中の保存失敗でTranscriptWriteErrorを送出すること(controller, writer):
    controller.run_forever.return_value = iter([(None, metadata("一つ目"))])
    writer.append.side_effect = TranscriptWriteError("認識結果を保存できませんでした")
    with pytest.raises(TranscriptWriteError):
        app.transcribe_loop(controller, writer)


def test異常系_transcribe_loopメソッドが録音中の保存失敗でも録音を停止すること(controller, writer):
    controller.run_forever.return_value = iter([(None, metadata("一つ目"))])
    writer.append.side_effect = TranscriptWriteError("認識結果を保存できませんでした")
    with pytest.raises(TranscriptWriteError):
        app.transcribe_loop(controller, writer)
    controller.stop_listening.assert_called()


# --- transcribe_loop / flush_pending_speech（実際のAudioControllerとTranscriptWriterを組み合わせる） ---

CHUNK = 512
# silence_duration=0.064秒は2チャンク分の無音で区間が終わる
ONE_UTTERANCE = [True, True, True, False, False]


@pytest.fixture
def stt_service():
    stt_service = MagicMock()
    stt_service.transcribe.return_value = "こんにちは"
    return stt_service


@pytest.fixture
def stream():
    stream = MagicMock()
    stream.active = True
    return stream


@pytest.fixture
def real_controller(stt_service, stream):
    vad_service = MagicMock()
    vad_service.required_chunk_size.return_value = CHUNK
    audio_input = MagicMock()
    audio_input.open_stream.return_value = stream
    return AudioController(
        vad_service=vad_service,
        stt_service=stt_service,
        audio_input=audio_input,
        silence_duration=0.064,
        pre_buffer_duration=0.0,
    )


@pytest.fixture
def output_file(tmp_path):
    return tmp_path / "output.txt"


@pytest.fixture
def real_writer(output_file):
    return TranscriptWriter(str(output_file))


def queue_speech(controller, speech_flags):
    """マイクから届いた体で、VADの判定結果つきのチャンクをキューへ積む"""
    controller.vad_service.is_speech.side_effect = list(speech_flags)
    for _ in speech_flags:
        controller.audio_queue.put(np.full(CHUNK, 0.5, dtype=np.float32))


def saved_texts(output_file):
    if not output_file.exists():
        return []
    return [line.split(": ", 1)[1] for line in output_file.read_text(encoding="utf-8").splitlines()]


def test異常系_flush_pending_speechメソッドが後続の区間の継続不能な失敗より前の文字起こしを保存すること(
    real_controller, real_writer, stt_service, output_file
):
    stt_service.transcribe.side_effect = ["一つ目", RecognitionError("認証失敗", fatal=True), "三つ目"]
    queue_speech(real_controller, ONE_UTTERANCE * 3)
    with pytest.raises(RecognitionError):
        app.flush_pending_speech(real_controller, real_writer)
    assert saved_texts(output_file) == ["一つ目"]


def test正常系_transcribe_loopメソッドが認識中のCtrl_Cで処理中だった発話を書き出すこと(
    real_controller, real_writer, stt_service, output_file
):
    stt_service.transcribe.side_effect = [KeyboardInterrupt, "認識中だった発話"]
    queue_speech(real_controller, ONE_UTTERANCE)
    app.transcribe_loop(real_controller, real_writer)
    assert saved_texts(output_file) == ["認識中だった発話"]


def test正常系_transcribe_loopメソッドが認識中のCtrl_Cでそれより後の音声も順に書き出すこと(
    real_controller, real_writer, stt_service, output_file
):
    stt_service.transcribe.side_effect = [KeyboardInterrupt, "認識中だった発話", "話している途中の発話"]
    queue_speech(real_controller, ONE_UTTERANCE + [True])
    app.transcribe_loop(real_controller, real_writer)
    assert saved_texts(output_file) == ["認識中だった発話", "話している途中の発話"]


def test異常系_transcribe_loopメソッドが書き出し中の再度のCtrl_Cでは再試行を繰り返さず終了すること(
    real_controller, real_writer, stt_service
):
    stt_service.transcribe.side_effect = KeyboardInterrupt
    queue_speech(real_controller, ONE_UTTERANCE)
    app.transcribe_loop(real_controller, real_writer)
    assert stt_service.transcribe.call_count == 2


def test異常系_transcribe_loopメソッドが書き出し中の再度のCtrl_Cで失われた音声の長さを知らせること(
    real_controller, real_writer, stt_service, caplog
):
    stt_service.transcribe.side_effect = KeyboardInterrupt
    queue_speech(real_controller, ONE_UTTERANCE)
    with caplog.at_level(logging.WARNING, logger="voicescribe"):
        app.transcribe_loop(real_controller, real_writer)
    assert "約0.1秒の音声は文字起こしされていません" in caplog.text


def test異常系_transcribe_loopメソッドが録音中に出力先へ書けなくなったときマイクを閉じること(
    real_controller, stream, tmp_path
):
    queue_speech(real_controller, ONE_UTTERANCE)
    with pytest.raises(TranscriptWriteError):
        app.transcribe_loop(real_controller, TranscriptWriter(str(tmp_path)))
    stream.close.assert_called_once_with()


def test異常系_mainメソッドが録音中の保存失敗を利用者へ表示すること(main_dependencies, caplog):
    main_dependencies["run"].side_effect = TranscriptWriteError("認識結果を保存できませんでした: output.txt")
    with caplog.at_level(logging.ERROR, logger="voicescribe"):
        app.main([])
    assert "エラー: 認識結果を保存できませんでした: output.txt" in caplog.text


def test異常系_mainメソッドが録音中の保存失敗で終了コード1を返すこと(main_dependencies):
    main_dependencies["run"].side_effect = TranscriptWriteError("認識結果を保存できませんでした")
    assert app.main([]) == app.EXIT_RUNTIME_ERROR


# --- main から transcribe_loop / flush_pending_speech まで（マイクと音声認識だけを差し替える） ---


class InterruptingQueue:
    """録音ループの最初の取り出しでCtrl+Cが押された状況を模し、以後は積まれた音声をそのまま渡すキュー"""

    def __init__(self, source):
        self._source = source
        self._interrupted = False

    def get(self, timeout=None):
        if not self._interrupted:
            self._interrupted = True
            raise KeyboardInterrupt
        return self._source.get(timeout=timeout)

    def __getattr__(self, name):
        return getattr(self._source, name)


@pytest.fixture
def shutdown_app(stt_service, stream):
    """Ctrl+Cの時点で未処理の発話が残っているアプリを、main() から実行できる状態にする"""
    created = []

    def create_controller(**kwargs):
        kwargs.update(silence_duration=0.064, pre_buffer_duration=0.0)
        controller = AudioController(**kwargs)
        controller.audio_queue = InterruptingQueue(controller.audio_queue)
        created.append(controller)
        return controller

    def queue_pending_speech(speech_flags):
        vad_service.is_speech.side_effect = list(speech_flags)
        audio_input.open_stream.side_effect = lambda **_kwargs: [
            created[-1].audio_queue.put(np.full(CHUNK, 0.5, dtype=np.float32)) for _ in speech_flags
        ] and stream

    with (
        patch("src.infrastructure.audio_devices.SoundDeviceAudioInput") as audio_input_class,
        patch("src.infrastructure.recognition_service_factory.create_recognition_service", return_value=stt_service),
        patch("src.domain.vad_service.VADService") as vad_service_class,
        patch("src.application.audio_controller.AudioController", side_effect=create_controller),
        patch.object(app, "configure_console"),
        patch("dotenv.load_dotenv"),
    ):
        audio_input = audio_input_class.return_value
        audio_input.resolve_device.return_value = 1
        audio_input.open_stream.return_value = stream
        vad_service = vad_service_class.return_value
        vad_service.required_chunk_size.return_value = CHUNK
        yield queue_pending_speech


def test正常系_mainメソッドがCtrl_C後の書き出しに成功したとき終了コード0を返すこと(shutdown_app, output_file):
    shutdown_app(ONE_UTTERANCE)
    assert app.main(["-o", str(output_file)]) == app.EXIT_OK


def test正常系_mainメソッドがCtrl_C後に未処理の発話を保存すること(shutdown_app, output_file):
    shutdown_app(ONE_UTTERANCE)
    app.main(["-o", str(output_file)])
    assert saved_texts(output_file) == ["こんにちは"]


def test異常系_mainメソッドがCtrl_C後の保存失敗で終了コード1を返すこと(shutdown_app, output_file):
    shutdown_app(ONE_UTTERANCE)
    with patch.object(TranscriptWriter, "append", side_effect=TranscriptWriteError("認識結果を保存できませんでした")):
        assert app.main(["-o", str(output_file)]) == app.EXIT_RUNTIME_ERROR


def test異常系_mainメソッドがCtrl_C後の保存失敗でもマイクを閉じること(shutdown_app, output_file, stream):
    shutdown_app(ONE_UTTERANCE)
    with patch.object(TranscriptWriter, "append", side_effect=TranscriptWriteError("認識結果を保存できませんでした")):
        app.main(["-o", str(output_file)])
    stream.close.assert_called_once_with()


def test異常系_mainメソッドがCtrl_C後の保存失敗を利用者へ表示すること(shutdown_app, output_file, caplog):
    shutdown_app(ONE_UTTERANCE)
    with (
        patch.object(TranscriptWriter, "append", side_effect=TranscriptWriteError("認識結果を保存できませんでした")),
        caplog.at_level(logging.ERROR, logger="voicescribe"),
    ):
        app.main(["-o", str(output_file)])
    assert "エラー: 認識結果を保存できませんでした" in caplog.text


def test異常系_mainメソッドがCtrl_C後の継続不能な認識失敗で終了コード1を返すこと(
    shutdown_app, stt_service, output_file
):
    stt_service.transcribe.side_effect = RecognitionError("認証失敗", fatal=True)
    shutdown_app(ONE_UTTERANCE)
    assert app.main(["-o", str(output_file)]) == app.EXIT_RUNTIME_ERROR


def test異常系_mainメソッドがCtrl_C後の継続不能な認識失敗でもマイクを閉じること(
    shutdown_app, stt_service, output_file, stream
):
    stt_service.transcribe.side_effect = RecognitionError("認証失敗", fatal=True)
    shutdown_app(ONE_UTTERANCE)
    app.main(["-o", str(output_file)])
    stream.close.assert_called_once_with()


def test異常系_mainメソッドがCtrl_C後の継続不能な認識失敗より前の文字起こしを保存すること(
    shutdown_app, stt_service, output_file
):
    stt_service.transcribe.side_effect = ["一つ目", RecognitionError("認証失敗", fatal=True), "三つ目"]
    shutdown_app(ONE_UTTERANCE * 3)
    app.main(["-o", str(output_file)])
    assert saved_texts(output_file) == ["一つ目"]


# --- report_transcript ---


def test正常系_report_transcriptメソッドが認識結果をライターへ渡すこと(writer):
    result = metadata()
    app.report_transcript(writer, result)
    writer.append.assert_called_once_with(result)


def test正常系_report_transcriptメソッドが認識結果をコンソールに表示すること(writer, capsys):
    app.report_transcript(writer, metadata("こんにちは"))
    assert capsys.readouterr().out == "音声認識結果: こんにちは\n"


def test正常系_report_transcriptメソッドがウェイクワードの検出を表示すること(writer, capsys):
    app.report_transcript(writer, metadata("アウラ", is_wake_word=True))
    assert capsys.readouterr().out == "ウェイクワード検出: アウラ\n"


def test正常系_report_transcriptメソッドが書き込まなかったときは表示しないこと(writer, capsys):
    writer.append.return_value = False
    app.report_transcript(writer, metadata(""))
    assert capsys.readouterr().out == ""


def test異常系_report_transcriptメソッドが保存の失敗でTranscriptWriteErrorを送出すること(writer):
    writer.append.side_effect = TranscriptWriteError("認識結果を保存できませんでした")
    with pytest.raises(TranscriptWriteError):
        app.report_transcript(writer, metadata("こんにちは"))


def test異常系_report_transcriptメソッドが保存できなかった認識結果をコンソールに残すこと(writer, capsys):
    writer.append.side_effect = TranscriptWriteError("認識結果を保存できませんでした")
    with pytest.raises(TranscriptWriteError):
        app.report_transcript(writer, metadata("こんにちは"))
    assert capsys.readouterr().out == "保存できなかった認識結果: こんにちは\n"


# --- create_failed_audio_saver ---


def test正常系_create_failed_audio_saverメソッドが失敗した発話をWAVで保存すること(tmp_path):
    save_failed_audio = app.create_failed_audio_saver(str(tmp_path / "failed"), 16000)
    save_failed_audio(np.full(1600, 0.5, dtype=np.float32), RecognitionError("失敗"))
    saved_file = next((tmp_path / "failed").glob("failed_*.wav"))
    with wave.open(str(saved_file), "rb") as wf:
        assert wf.getnframes() == 1600


def test異常系_create_failed_audio_saverメソッドが保存に失敗しても例外を出さないこと(tmp_path):
    save_failed_audio = app.create_failed_audio_saver(str(tmp_path), 16000)
    with patch("src.infrastructure.wav_encoder.write_wav", side_effect=OSError("disk full")):
        assert save_failed_audio(np.zeros(16, dtype=np.float32), RecognitionError("失敗")) is None


# --- configure_console ---


def test正常系_configure_consoleメソッドが表示できない文字で止まらない設定にすること():
    stdout = MagicMock()
    with patch.object(app.sys, "stdout", stdout), patch.object(app.logging, "basicConfig"):
        app.configure_console()
    stdout.reconfigure.assert_called_once_with(errors="replace")


def test正常系_mainメソッドが基底のVoiceScribeErrorで終了コード1を返すこと(main_dependencies):
    main_dependencies["run"].side_effect = VoiceScribeError("想定外の失敗")
    assert app.main([]) == 1
