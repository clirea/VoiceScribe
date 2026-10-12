# tests/test_mic_check.py

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import mic_check
from src.domain.errors import AudioInputError, ConfigurationError
from src.infrastructure.audio_devices import InputDevice


@pytest.fixture
def audio_input():
    audio_input = MagicMock()
    audio_input.list_input_devices.return_value = [
        InputDevice(index=1, name="Built-in Microphone", channels=1, host_api="Core Audio", is_default=True)
    ]
    audio_input.resolve_device.return_value = 1
    audio_input.describe_device.return_value = "Built-in Microphone (ID: 1)"
    return audio_input


@pytest.fixture
def write_wav():
    with patch.object(mic_check, "write_wav") as write_wav:
        yield write_wav


def recording_sleep(audio_input, blocks=2):
    """録音の待ち時間の代わりに、マイクのコールバックへ音声を届けるsleepを返す"""

    def sleep(milliseconds):
        callback = audio_input.open_stream.call_args.kwargs["callback"]
        for _ in range(blocks):
            callback(np.full((800, 1), 0.5, dtype=np.float32), 800, None, None)

    return sleep


def record(audio_input, **kwargs):
    return mic_check.test_microphone(audio_input, sleep=recording_sleep(audio_input), **kwargs)


def script(*answers):
    """メニューへの入力を順に返すread_inputを作る"""
    return MagicMock(side_effect=list(answers))


# --- list_audio_devices ---


def test正常系_list_audio_devicesメソッドがIDと名前の一覧を返すこと(audio_input):
    assert mic_check.list_audio_devices(audio_input) == [(1, "Built-in Microphone")]


def test正常系_list_audio_devicesメソッドが既定のマイクに印を付けて表示すること(audio_input, capsys):
    mic_check.list_audio_devices(audio_input)
    assert "* 1: Built-in Microphone" in capsys.readouterr().out


# --- test_microphone ---


def test正常系_test_microphoneメソッドが録音した音声を連結して返すこと(audio_input, write_wav):
    audio_data, _filename = record(audio_input)
    assert len(audio_data) == 1600


def test正常系_test_microphoneメソッドが保存したファイル名を返すこと(audio_input, write_wav):
    _audio_data, filename = record(audio_input)
    assert filename.startswith("mic_test_") and filename.endswith(".wav")


def test正常系_test_microphoneメソッドが録音をWAVへ保存すること(audio_input, write_wav):
    audio_data, filename = record(audio_input)
    write_wav.assert_called_once_with(filename, audio_data, 16000)


def test正常系_test_microphoneメソッドが指定されたマイクを解決すること(audio_input, write_wav):
    record(audio_input, device_name="Built-in")
    audio_input.resolve_device.assert_called_once_with(device_id=None, device_name="Built-in")


def test正常系_test_microphoneメソッドが解決したマイクで録音すること(audio_input, write_wav):
    record(audio_input)
    assert audio_input.open_stream.call_args.kwargs["device"] == 1


def test正常系_test_microphoneメソッドが録音後にストリームを閉じること(audio_input, write_wav):
    record(audio_input)
    audio_input.open_stream.return_value.close.assert_called_once_with()


def test正常系_test_microphoneメソッドが音声が届かないときNoneを返すこと(audio_input, write_wav):
    assert mic_check.test_microphone(audio_input, sleep=lambda milliseconds: None) is None


def test正常系_test_microphoneメソッドが音声が届かないときファイルを保存しないこと(audio_input, write_wav):
    mic_check.test_microphone(audio_input, sleep=lambda milliseconds: None)
    write_wav.assert_not_called()


def test異常系_test_microphoneメソッドが録音中の例外でもストリームを閉じること(audio_input, write_wav):
    with pytest.raises(RuntimeError):
        mic_check.test_microphone(audio_input, sleep=MagicMock(side_effect=RuntimeError("中断")))
    audio_input.open_stream.return_value.close.assert_called_once_with()


def test異常系_test_microphoneメソッドが見つからないマイクでConfigurationErrorを送出すること(audio_input, write_wav):
    audio_input.resolve_device.side_effect = ConfigurationError("マイクが見つかりません")
    with pytest.raises(ConfigurationError):
        record(audio_input, device_name="存在しないマイク")


# --- run_menu ---


@pytest.fixture
def microphone_test():
    with patch.object(mic_check, "test_microphone") as microphone_test:
        yield microphone_test


def test正常系_run_menuメソッドが終了の選択でループを抜けること(audio_input, microphone_test, capsys):
    mic_check.run_menu(audio_input, read_input=script("4"))
    assert "終了します。" in capsys.readouterr().out


def test正常系_run_menuメソッドがID指定の選択で指定IDを録音すること(audio_input, microphone_test):
    mic_check.run_menu(audio_input, read_input=script("1", "2", "4"))
    microphone_test.assert_called_once_with(audio_input, device_id=2)


def test正常系_run_menuメソッドが名前指定の選択で指定名を録音すること(audio_input, microphone_test):
    mic_check.run_menu(audio_input, read_input=script("2", "Headset", "4"))
    microphone_test.assert_called_once_with(audio_input, device_name="Headset")


def test正常系_run_menuメソッドが既定マイクの選択で指定無しの録音をすること(audio_input, microphone_test):
    mic_check.run_menu(audio_input, read_input=script("3", "4"))
    microphone_test.assert_called_once_with(audio_input)


def test正常系_run_menuメソッドが無効な選択を案内して続行すること(audio_input, microphone_test, capsys):
    mic_check.run_menu(audio_input, read_input=script("9", "4"))
    assert "無効な選択です。" in capsys.readouterr().out


def test異常系_run_menuメソッドが数値でないIDを案内して続行すること(audio_input, microphone_test, capsys):
    mic_check.run_menu(audio_input, read_input=script("1", "abc", "4"))
    assert "有効な数値を入力してください。" in capsys.readouterr().out


def test異常系_run_menuメソッドが録音の失敗を表示して続行すること(audio_input, microphone_test, capsys):
    microphone_test.side_effect = AudioInputError("マイクを開けません")
    mic_check.run_menu(audio_input, read_input=script("3", "4"))
    assert "エラー: マイクを開けません" in capsys.readouterr().out


# --- main ---


@pytest.fixture
def main_dependencies(audio_input):
    with (
        patch.object(mic_check, "SoundDeviceAudioInput", return_value=audio_input) as audio_input_class,
        patch.object(mic_check, "run_menu") as run_menu,
    ):
        yield {"audio_input_class": audio_input_class, "run_menu": run_menu}


def test正常系_mainメソッドがメニュー終了で0を返すこと(main_dependencies):
    assert mic_check.main() == 0


def test正常系_mainメソッドがメニューへオーディオ入力を渡すこと(main_dependencies, audio_input):
    mic_check.main()
    main_dependencies["run_menu"].assert_called_once_with(audio_input)


def test正常系_mainメソッドが入力の終端で0を返すこと(main_dependencies):
    main_dependencies["run_menu"].side_effect = EOFError
    assert mic_check.main() == 0


def test正常系_mainメソッドがCtrl_Cで0を返すこと(main_dependencies):
    main_dependencies["run_menu"].side_effect = KeyboardInterrupt
    assert mic_check.main() == 0


def test異常系_mainメソッドがオーディオ機能を使えないとき1を返すこと(main_dependencies):
    main_dependencies["audio_input_class"].side_effect = ConfigurationError("PortAudioがありません")
    assert mic_check.main() == 1


@pytest.fixture
def cp1252_console():
    buffer = io.BytesIO()
    with io.TextIOWrapper(buffer, encoding="cp1252", errors="strict") as console:
        yield console


def test正常系_mainメソッドが日本語を表せないコンソールでもマイク一覧を表示して終了すること(
    main_dependencies, cp1252_console
):
    with patch.object(mic_check.sys, "stdout", cp1252_console):
        result = mic_check.main()
    assert result == 0


def test異常系_mainメソッドが日本語を表せないコンソールでも入力機器エラーを終了コード1にすること(
    main_dependencies, cp1252_console
):
    main_dependencies["audio_input_class"].side_effect = AudioInputError("マイクを開けません")
    with patch.object(mic_check.sys, "stdout", cp1252_console):
        result = mic_check.main()
    assert result == 1
