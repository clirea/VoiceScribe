# infrastructure/audio_devices.py

from dataclasses import dataclass

from src.domain.errors import AudioInputError, ConfigurationError


@dataclass(frozen=True)
class InputDevice:
    """入力に使えるオーディオデバイス。"""

    index: int
    name: str
    channels: int
    host_api: str
    is_default: bool


def load_sounddevice():
    """sounddeviceを読み込む。PortAudioが無い環境では理由を添えて失敗させる。"""
    try:
        import sounddevice
    except (ImportError, OSError) as e:
        raise ConfigurationError(
            f"sounddevice（PortAudio）を読み込めません: {e}\n"
            "`pip install -r requirements.txt` を実行してください。"
            "LinuxではPortAudio本体（例: libportaudio2）も必要です。"
        ) from e
    return sounddevice


def format_input_devices(devices) -> list:
    """デバイス一覧の表示用の行を返す。"""
    lines = ["利用可能なオーディオデバイス一覧:", "-" * 70]
    for device in devices:
        mark = "* " if device.is_default else "  "
        lines.append(
            f"{mark}{device.index}: {device.name} (入力チャンネル: {device.channels}, {device.host_api})"
        )
    if not devices:
        lines.append("  入力デバイスが見つかりません。マイクの接続とOSのマイク権限を確認してください。")
    lines.append("-" * 70)
    lines.append("* がデフォルトの入力デバイス")
    return lines


class SoundDeviceAudioInput:
    """sounddeviceを介したマイクの列挙・選択・入力ストリーム生成をまとめるクラス。"""

    def __init__(self, sd=None):
        self._sd = sd if sd is not None else load_sounddevice()

    def list_input_devices(self) -> list:
        """入力デバイス（マイク）の一覧を返す。"""
        try:
            devices = list(self._sd.query_devices())
            host_apis = list(self._sd.query_hostapis())
            default_index = self._sd.default.device[0]
        except Exception as e:
            raise AudioInputError(f"オーディオデバイスの一覧を取得できません: {e}") from e

        input_devices = []
        for index, device in enumerate(devices):
            if device["max_input_channels"] <= 0:
                continue
            host_api_index = device.get("hostapi", -1)
            host_api = host_apis[host_api_index]["name"] if 0 <= host_api_index < len(host_apis) else "不明"
            input_devices.append(
                InputDevice(
                    index=index,
                    name=device["name"],
                    channels=device["max_input_channels"],
                    host_api=host_api,
                    is_default=index == default_index,
                )
            )
        return input_devices

    def resolve_device(self, device_id=None, device_name=None):
        """
        ID・名前の指定から使用するデバイスIDを決める。どちらも無ければNone（OSの既定マイク）。

        意図しないマイクで録音しないよう、指定が見つからない場合は既定マイクへ切り替えずに失敗させる。
        """
        if device_id is None and device_name is None:
            return None

        devices = self.list_input_devices()

        # デバイス名が指定された場合、IDを検索（部分一致・最初の一致を採用）
        if device_name is not None:
            for device in devices:
                if device_name.lower() in device.name.lower():
                    return device.index
            raise ConfigurationError(
                f"'{device_name}' という名前のマイクが見つかりませんでした。`python main.py -l` で確認してください。"
            )

        if any(device.index == device_id for device in devices):
            return device_id
        raise ConfigurationError(
            f"ID {device_id} の入力デバイスはありません。`python main.py -l` で確認してください。"
        )

    def describe_device(self, device) -> str:
        """ログ表示用のデバイス名を返す。"""
        if device is None:
            return "デフォルトマイク"
        for candidate in self.list_input_devices():
            if candidate.index == device:
                return f"{candidate.name} (ID: {device})"
        return f"ID: {device}"

    def open_stream(self, *, sample_rate, block_size, device, callback, channels=1, dtype="float32"):
        """入力ストリームを生成して開始する。"""
        stream = None
        try:
            stream = self._sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype=dtype,
                blocksize=block_size,
                callback=callback,
                device=device,
            )
            stream.start()
        except Exception as e:
            if stream is not None:
                stream.close()
            raise AudioInputError(
                f"マイク入力を開始できません: {e}\n"
                f"マイクの接続、OSのマイク権限、デバイスが {sample_rate}Hz に対応しているかを確認してください。"
            ) from e
        return stream
