# tests/test_vad_service.py

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.domain.errors import ConfigurationError, VoiceActivityDetectionError
from src.domain.vad_service import VADService, load_silero_model

torch = pytest.importorskip("torch")

SAMPLE_RATE = 16000


def audio_chunk(size=512):
    return np.zeros(size, dtype=np.float32)


@pytest.fixture
def model():
    return MagicMock(return_value=torch.tensor([[0.8]]))


@pytest.fixture
def vad_service(model):
    return VADService(threshold=0.5, model=model)


def test正常系_is_speechメソッドが閾値以上の発話確率でTrueを返すこと(vad_service):
    assert vad_service.is_speech(audio_chunk(), SAMPLE_RATE) is True


def test正常系_is_speechメソッドが閾値未満の発話確率でFalseを返すこと(vad_service, model):
    model.return_value = torch.tensor([[0.3]])
    assert vad_service.is_speech(audio_chunk(), SAMPLE_RATE) is False


def test正常系_is_speechメソッドが閾値ちょうどの発話確率でTrueを返すこと(vad_service, model):
    model.return_value = torch.tensor([[0.5]])
    assert vad_service.is_speech(audio_chunk(), SAMPLE_RATE) is True


def test正常系_is_speechメソッドが空の配列に対してFalseを返すこと(vad_service):
    assert vad_service.is_speech(np.array([], dtype=np.float32), SAMPLE_RATE) is False


def test正常系_is_speechメソッドが空の配列ではモデルを呼ばないこと(vad_service, model):
    vad_service.is_speech(np.array([], dtype=np.float32), SAMPLE_RATE)
    model.assert_not_called()


def test正常系_is_speechメソッドがモデルへfloat32のテンソルを渡すこと(vad_service, model):
    vad_service.is_speech(np.zeros(512, dtype=np.float64), SAMPLE_RATE)
    assert model.call_args.args[0].dtype == torch.float32


def test正常系_is_speechメソッドがモデルへサンプルレートを渡すこと(vad_service, model):
    vad_service.is_speech(audio_chunk(), SAMPLE_RATE)
    assert model.call_args.args[1] == SAMPLE_RATE


def test異常系_is_speechメソッドがチャンク長の不一致でVoiceActivityDetectionErrorを送出すること(vad_service):
    with pytest.raises(VoiceActivityDetectionError):
        vad_service.is_speech(audio_chunk(100), SAMPLE_RATE)


def test異常系_is_speechメソッドがモデルの失敗でVoiceActivityDetectionErrorを送出すること(vad_service, model):
    model.side_effect = RuntimeError("モデルエラー")
    with pytest.raises(VoiceActivityDetectionError):
        vad_service.is_speech(audio_chunk(), SAMPLE_RATE)


def test異常系_is_speechメソッドが未対応のサンプルレートでConfigurationErrorを送出すること(vad_service):
    with pytest.raises(ConfigurationError):
        vad_service.is_speech(audio_chunk(), 44100)


def test正常系_required_chunk_sizeメソッドが16kHzで512を返すこと():
    assert VADService.required_chunk_size(16000) == 512


def test正常系_required_chunk_sizeメソッドが8kHzで256を返すこと():
    assert VADService.required_chunk_size(8000) == 256


def test異常系_required_chunk_sizeメソッドが未対応のサンプルレートでConfigurationErrorを送出すること():
    with pytest.raises(ConfigurationError):
        VADService.required_chunk_size(48000)


def test正常系_resetメソッドがモデルの状態を初期化すること(vad_service, model):
    vad_service.reset()
    model.reset_states.assert_called_once_with()


def test正常系_initメソッドがモデル未指定のとき同梱モデルを読み込むこと():
    loaded_model = MagicMock()
    with patch("src.domain.vad_service.load_silero_model", return_value=loaded_model):
        assert VADService().model is loaded_model


def test異常系_load_silero_modelメソッドがsilero_vad未導入でConfigurationErrorを送出すること():
    with patch.dict("sys.modules", {"silero_vad": None}):
        with pytest.raises(ConfigurationError):
            load_silero_model()


def test正常系_is_speechメソッドが同梱の実モデルで無音をFalseと判定すること():
    # ネットワーク接続はconftestで遮断しており、同梱モデルだけで動くことも確かめている
    pytest.importorskip("silero_vad")
    vad_service = VADService(threshold=0.5)
    assert vad_service.is_speech(audio_chunk(), SAMPLE_RATE) is False
