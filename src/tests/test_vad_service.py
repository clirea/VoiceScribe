
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from domain.vad_service import VADService

@pytest.fixture
def vad_service():
    return VADService(threshold=0.5)

def test正常系_is_speechメソッドが音声を検出すること(vad_service):
    # 音声らしきデータを生成(サイン波)
    sample_rate = 16000
    duration = 0.1  # 100ms
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_chunk = np.sin(2 * np.pi * 440 * t)  # 440Hzのサイン波
    
    # モデルの戻り値をモック
    with patch.object(vad_service.model, '__call__', return_value=torch.tensor(0.8)):
        result = vad_service.is_speech(audio_chunk, sample_rate)
        assert result == True

def test正常系_is_speechメソッドが無音を検出すること(vad_service):
    # 無音データ(ゼロ配列)を生成
    audio_chunk = np.zeros(16000)
    sample_rate = 16000
    
    # モデルの戻り値をモック
    with patch.object(vad_service.model, '__call__', return_value=torch.tensor(0.3)):
        result = vad_service.is_speech(audio_chunk, sample_rate)
        assert result == False

def test異常系_is_speechメソッドが空の配列に対してFalseを返すこと(vad_service):
    audio_chunk = np.array([])
    sample_rate = 16000
    result = vad_service.is_speech(audio_chunk, sample_rate)
    assert result == False

def test異常系_is_speechメソッドがValueErrorを適切に処理すること(vad_service):
    audio_chunk = np.array([0.1, 0.2])  # 極端に短いチャンク
    sample_rate = 16000
    
    # モデルがValueErrorを投げるようにモック
    with patch.object(vad_service.model, '__call__', side_effect=ValueError):
        result = vad_service.is_speech(audio_chunk, sample_rate)
        assert result == False

def test異常系_is_speechメソッドが予期せぬエラーを適切に処理すること(vad_service):
    audio_chunk = np.array([0.1, 0.2, 0.3])
    sample_rate = 16000
    
    # モデルが予期せぬエラーを投げるようにモック
    with patch.object(vad_service.model, '__call__', side_effect=Exception("予期せぬエラー")):
        result = vad_service.is_speech(audio_chunk, sample_rate)
        assert result == False
