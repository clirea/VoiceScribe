# tests/test_audio_controller.py

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from application.audio_controller import AudioController

@pytest.fixture
def controller():
    return AudioController()

def test正常系_run_foreverで無音を検知してファイル保存すること(controller):
    fake_chunk = np.zeros((512, 1), dtype=np.float32)

    # キューに何回か無音チャンクをput
    for _ in range(5):
        controller.audio_queue.put(fake_chunk)

    with patch.object(controller, 'save_to_wav') as mock_save:
        with patch.object(controller.stt_service, 'transcribe', return_value=""):
            controller.is_running = True
            # run_foreverを少しだけ走らせる
            with patch('time.sleep', side_effect=Exception("StopLoop")):
                try:
                    controller.run_forever()
                except Exception as e:
                    # time.sleepへのパッチでループ強制脱出
                    pass

        assert mock_save.called  # ファイル保存メソッドが呼び出されたか

def test正常系_wakeword検出時にファイル名にwake_wordが含まれること(controller):
    fake_chunk = np.ones((512, 1), dtype=np.float32)  # ある程度振幅あり
    controller.audio_queue.put(fake_chunk)

    # Wakeword検出するようにモック
    with patch.object(controller.stt_service, 'transcribe', return_value="アウラです"), \
         patch.object(controller.wakeword_detector, 'detect', return_value=True), \
         patch.object(controller, 'save_to_wav') as mock_save:
        controller.is_running = True
        with patch('time.sleep', side_effect=Exception("StopLoop")):
            try:
                controller.run_forever()
            except:
                pass

        # save_to_wavが呼ばれた際の引数を確認
        assert mock_save.called
        args, kwargs = mock_save.call_args
        filename = args[1]
        assert "wake_word" in filename
