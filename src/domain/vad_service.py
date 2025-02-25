# domain/vad_service.py

import numpy as np
import torch

class VADService:
    """
    Silero VAD を利用して音声データの音声/無音判定をするクラス。
    無音になったらファイルを切り出すなどのロジックを
    ここを通して行えるようにしているわ。
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        # Silero VADモデルをロード
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        self.get_speech_timestamps, self.get_speech_probs, *_ = utils

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int) -> bool:
        """
        音声チャンクが一定以上の発話確率かどうかを判定する。
        """
        if audio_chunk.size == 0:
            return False

        # たまに短すぎてValueErrorが発生するから、回避してる
        try:
            tensor_audio = torch.from_numpy(audio_chunk)
            prob = self.model(tensor_audio, sample_rate)
            return prob >= self.threshold
        except ValueError:
            # Chunkが短い場合など
            return False
        except Exception as e:
            print(f"VADService: 予期せぬエラー {e}")
            return False
