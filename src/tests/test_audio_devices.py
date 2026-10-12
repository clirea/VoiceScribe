# tests/test_audio_devices.py

from unittest.mock import MagicMock, patch

import pytest

from src.domain.errors import AudioInputError, ConfigurationError
from src.infrastructure.audio_devices import (
    InputDevice,
    SoundDeviceAudioInput,
    format_input_devices,
    load_sounddevice,
)


@pytest.fixture
def sd():
    """マイク2台とスピーカー1台がある環境のsounddeviceを模したモック"""
    sd = MagicMock()
    sd.query_devices.return_value = [
        {"name": "Speakers", "hostapi": 0, "max_input_channels": 0},
        {"name": "Built-in Microphone", "hostapi": 0, "max_input_channels": 1},
        {"name": "USB Headset Mic", "hostapi": 1, "max_input_channels": 2},
    ]
    sd.query_hostapis.return_value = [{"name": "MME"}, {"name": "Windows WASAPI"}]
    sd.default.device = (1, 0)
    return sd


@pytest.fixture
def audio_input(sd):
    return SoundDeviceAudioInput(sd=sd)


def open_stream(audio_input, callback=None):
    return audio_input.open_stream(sample_rate=16000, block_size=512, device=2, callback=callback)


def test正常系_list_input_devicesメソッドが入力チャンネルの無いデバイスを除くこと(audio_input):
    assert [device.index for device in audio_input.list_input_devices()] == [1, 2]


def test正常系_list_input_devicesメソッドが既定の入力デバイスを判別すること(audio_input):
    assert [device.is_default for device in audio_input.list_input_devices()] == [True, False]


def test正常系_list_input_devicesメソッドがホストAPI名を付けること(audio_input):
    assert audio_input.list_input_devices()[1].host_api == "Windows WASAPI"


def test正常系_list_input_devicesメソッドが入力チャンネル数を返すこと(audio_input):
    assert audio_input.list_input_devices()[1].channels == 2


def test異常系_list_input_devicesメソッドが取得失敗でAudioInputErrorを送出すること(audio_input, sd):
    sd.query_devices.side_effect = RuntimeError("PortAudio error")
    with pytest.raises(AudioInputError):
        audio_input.list_input_devices()


def test正常系_resolve_deviceメソッドが指定無しのときNoneを返すこと(audio_input):
    assert audio_input.resolve_device() is None


def test正常系_resolve_deviceメソッドが名前の部分一致でデバイスIDを返すこと(audio_input):
    assert audio_input.resolve_device(device_name="headset") == 2


def test正常系_resolve_deviceメソッドが名前とIDの両方があるとき名前を優先すること(audio_input):
    assert audio_input.resolve_device(device_id=1, device_name="USB") == 2


def test正常系_resolve_deviceメソッドが存在する入力デバイスIDをそのまま返すこと(audio_input):
    assert audio_input.resolve_device(device_id=2) == 2


def test異常系_resolve_deviceメソッドが見つからない名前でConfigurationErrorを送出すること(audio_input):
    with pytest.raises(ConfigurationError):
        audio_input.resolve_device(device_name="存在しないマイク")


def test異常系_resolve_deviceメソッドが出力専用デバイスのIDでConfigurationErrorを送出すること(audio_input):
    with pytest.raises(ConfigurationError):
        audio_input.resolve_device(device_id=0)


def test異常系_resolve_deviceメソッドが存在しないIDでConfigurationErrorを送出すること(audio_input):
    with pytest.raises(ConfigurationError):
        audio_input.resolve_device(device_id=99)


def test正常系_describe_deviceメソッドが指定無しのときデフォルトマイクと返すこと(audio_input):
    assert audio_input.describe_device(None) == "デフォルトマイク"


def test正常系_describe_deviceメソッドがデバイス名とIDを返すこと(audio_input):
    assert audio_input.describe_device(2) == "USB Headset Mic (ID: 2)"


def test正常系_open_streamメソッドが指定した条件で入力ストリームを生成すること(audio_input, sd):
    callback = MagicMock()
    open_stream(audio_input, callback)
    sd.InputStream.assert_called_once_with(
        samplerate=16000, channels=1, dtype="float32", blocksize=512, callback=callback, device=2
    )


def test正常系_open_streamメソッドがストリームを開始すること(audio_input, sd):
    open_stream(audio_input)
    sd.InputStream.return_value.start.assert_called_once_with()


def test正常系_open_streamメソッドが生成したストリームを返すこと(audio_input, sd):
    assert open_stream(audio_input) is sd.InputStream.return_value


def test異常系_open_streamメソッドがストリーム生成の失敗でAudioInputErrorを送出すること(audio_input, sd):
    sd.InputStream.side_effect = RuntimeError("Invalid sample rate")
    with pytest.raises(AudioInputError):
        open_stream(audio_input)


def test異常系_open_streamメソッドが開始の失敗でAudioInputErrorを送出すること(audio_input, sd):
    sd.InputStream.return_value.start.side_effect = RuntimeError("Device unavailable")
    with pytest.raises(AudioInputError):
        open_stream(audio_input)


def test異常系_open_streamメソッドが開始に失敗したストリームを閉じること(audio_input, sd):
    sd.InputStream.return_value.start.side_effect = RuntimeError("Device unavailable")
    with pytest.raises(AudioInputError):
        open_stream(audio_input)
    sd.InputStream.return_value.close.assert_called_once_with()


def test正常系_format_input_devicesメソッドが既定のデバイスに印を付けること():
    devices = [InputDevice(index=1, name="Built-in Microphone", channels=1, host_api="Core Audio", is_default=True)]
    assert "* 1: Built-in Microphone (入力チャンネル: 1, Core Audio)" in format_input_devices(devices)


def test正常系_format_input_devicesメソッドが既定でないデバイスに印を付けないこと():
    devices = [InputDevice(index=2, name="USB Mic", channels=2, host_api="MME", is_default=False)]
    assert "  2: USB Mic (入力チャンネル: 2, MME)" in format_input_devices(devices)


def test正常系_format_input_devicesメソッドがデバイスが無いとき確認事項を案内すること():
    assert any("入力デバイスが見つかりません" in line for line in format_input_devices([]))


def test異常系_load_sounddeviceメソッドが読み込み失敗でConfigurationErrorを送出すること():
    with patch.dict("sys.modules", {"sounddevice": None}):
        with pytest.raises(ConfigurationError):
            load_sounddevice()


def test正常系_initメソッドがモジュール未指定のときsounddeviceを読み込むこと():
    loaded = MagicMock()
    with patch("src.infrastructure.audio_devices.load_sounddevice", return_value=loaded) as load:
        SoundDeviceAudioInput()
    load.assert_called_once_with()
