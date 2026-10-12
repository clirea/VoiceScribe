# domain/vad_service.py

import warnings

import numpy as np

from src.domain.errors import ConfigurationError, VoiceActivityDetectionError

# Silero VADが1回の判定で受け付けるサンプル数（サンプルレートごとに固定）
CHUNK_SIZES = {16000: 512, 8000: 256}


def load_silero_model():
    """silero-vadパッケージ同梱のモデルを読み込む（ネットワーク接続は不要）。"""
    try:
        from silero_vad import load_silero_vad
    except ImportError as e:
        raise ConfigurationError(
            "silero-vad / torch を読み込めません。`pip install -r requirements.txt` を実行してください。"
        ) from e
    with warnings.catch_warnings():
        # silero-vadが内部で使う torch.jit.load の非推奨警告は利用者が対処できないため表示しない
        warnings.filterwarnings("ignore", message=".*torch.jit.load.*", category=FutureWarning)
        return load_silero_vad()


class VADService:
    """
    Silero VAD を利用して音声データの音声/無音判定をするクラス。
    無音になったらファイルを切り出すなどのロジックを
    ここを通して行えるようにしているわ。
    """

    def __init__(self, threshold=0.5, model=None):
        self.threshold = threshold
        # Silero VADモデルをロード（テストでは差し替えられる）
        self.model = model if model is not None else load_silero_model()

    @staticmethod
    def required_chunk_size(sample_rate: int) -> int:
        """指定サンプルレートでVADに渡すべきチャンク長を返す。"""
        if sample_rate not in CHUNK_SIZES:
            supported = ", ".join(str(rate) for rate in sorted(CHUNK_SIZES))
            raise ConfigurationError(
                f"Silero VADが対応するサンプルレートは {supported} Hz です（指定値: {sample_rate}）。"
            )
        return CHUNK_SIZES[sample_rate]

    def reset(self):
        """ストリームをまたいで内部状態を持ち越さないよう、モデルの状態を初期化する。"""
        self.model.reset_states()

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int) -> bool:
        """
        音声チャンクが一定以上の発話確率かどうかを判定する。
        """
        if audio_chunk.size == 0:
            return False

        expected_size = self.required_chunk_size(sample_rate)
        if audio_chunk.size != expected_size:
            raise VoiceActivityDetectionError(
                f"VADへ渡すチャンクは {expected_size} サンプルである必要があります（受信: {audio_chunk.size}）。"
            )

        import torch

        try:
            tensor_audio = torch.from_numpy(np.ascontiguousarray(audio_chunk, dtype=np.float32))
            with torch.no_grad():
                prob = self.model(tensor_audio, sample_rate)
            return float(prob) >= self.threshold
        except Exception as e:
            raise VoiceActivityDetectionError(f"VADの判定に失敗しました: {e}") from e
