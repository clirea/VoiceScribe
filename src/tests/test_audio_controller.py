# tests/test_audio_controller.py

import logging
import re
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from src.application.audio_controller import AudioController
from src.domain.errors import AudioInputError, ConfigurationError, RecognitionError

CHUNK = 512
# silence_duration=0.064秒は2チャンク分の無音で区間が終わる
SPEECH_THEN_SILENCE = [True, True, True, False, False]


def chunk(value=0.5):
    return np.full(CHUNK, value, dtype=np.float32)


@pytest.fixture
def vad_service():
    vad_service = MagicMock()
    vad_service.required_chunk_size.return_value = CHUNK
    vad_service.is_speech.return_value = False
    return vad_service


@pytest.fixture
def stt_service():
    stt_service = MagicMock()
    stt_service.transcribe.return_value = "こんにちは"
    return stt_service


@pytest.fixture
def audio_input():
    audio_input = MagicMock()
    audio_input.open_stream.return_value.active = True
    return audio_input


@pytest.fixture
def make_controller(vad_service, stt_service, audio_input):
    def factory(**overrides):
        options = {
            "vad_service": vad_service,
            "stt_service": stt_service,
            "audio_input": audio_input,
            "silence_duration": 0.064,
            "pre_buffer_duration": 0.0,
            "max_speech_duration": 0,
        }
        options.update(overrides)
        return AudioController(**options)

    return factory


@pytest.fixture
def controller(make_controller):
    return make_controller()


def queue_chunks(controller, speech_flags, value=0.5):
    """VADの判定結果つきでチャンクをキューへ積む"""
    controller.vad_service.is_speech.side_effect = list(speech_flags)
    for _ in speech_flags:
        controller.audio_queue.put(chunk(value))


def run(controller, speech_flags, value=0.5):
    """積んだチャンクを処理し終えたら停止するようにして、run_foreverの結果を集める"""
    flags = list(speech_flags)

    def is_speech(mono_chunk, sample_rate):
        flag = flags.pop(0)
        if not flags:
            controller.is_running = False
        return flag

    controller.vad_service.is_speech.side_effect = is_speech
    for _ in speech_flags:
        controller.audio_queue.put(chunk(value))
    controller.is_running = True
    return list(controller.run_forever())


# --- __init__ ---


def test異常系_initメソッドがVADと合わないブロックサイズでConfigurationErrorを送出すること(make_controller):
    with pytest.raises(ConfigurationError):
        make_controller(block_size=1024)


def test正常系_initメソッドが保持秒数からキューの上限を決めること(make_controller):
    controller = make_controller(max_queue_seconds=60.0)
    assert controller.audio_queue.maxsize == 1875


# --- run_forever ---


def test正常系_run_foreverメソッドが発話後の無音で認識結果を1件返すこと(controller):
    assert len(run(controller, SPEECH_THEN_SILENCE)) == 1


def test正常系_run_foreverメソッドが認識テキストをメタデータに入れること(controller):
    _audio, metadata = run(controller, SPEECH_THEN_SILENCE)[0]
    assert metadata["text"] == "こんにちは"


def test正常系_run_foreverメソッドが切り出した発話の音声を返すこと(controller):
    audio, _metadata = run(controller, SPEECH_THEN_SILENCE)[0]
    assert len(audio) == 3 * CHUNK


def test正常系_run_foreverメソッドが切り出した発話を音声認識へ渡すこと(controller, stt_service):
    run(controller, SPEECH_THEN_SILENCE)
    assert len(stt_service.transcribe.call_args.args[0]) == 3 * CHUNK


def test正常系_run_foreverメソッドがサンプルレートを音声認識へ渡すこと(controller, stt_service):
    run(controller, SPEECH_THEN_SILENCE)
    assert stt_service.transcribe.call_args.args[1] == 16000


def test正常系_run_foreverメソッドがウェイクワードを含む発話を検出すること(controller, stt_service):
    stt_service.transcribe.return_value = "アウラ、聞こえる？"
    _audio, metadata = run(controller, SPEECH_THEN_SILENCE)[0]
    assert metadata["is_wake_word"] is True


def test正常系_run_foreverメソッドがウェイクワードの無い発話を通常の認識として扱うこと(controller):
    _audio, metadata = run(controller, SPEECH_THEN_SILENCE)[0]
    assert metadata["is_wake_word"] is False


def test正常系_run_foreverメソッドがタイムスタンプを従来の書式で付けること(controller):
    _audio, metadata = run(controller, SPEECH_THEN_SILENCE)[0]
    assert re.fullmatch(r"\d{8}_\d{6}", metadata["timestamp"])


def test正常系_run_foreverメソッドが無音だけのとき音声認識を呼ばないこと(controller, stt_service):
    run(controller, [False] * 10)
    stt_service.transcribe.assert_not_called()


def test正常系_run_foreverメソッドが連続する2つの発話を別々に返すこと(controller):
    assert len(run(controller, SPEECH_THEN_SILENCE + SPEECH_THEN_SILENCE)) == 2


def test正常系_run_foreverメソッドが聞き取れなかった発話を結果に含めないこと(controller, stt_service):
    stt_service.transcribe.return_value = ""
    assert run(controller, SPEECH_THEN_SILENCE) == []


def test正常系_run_foreverメソッドが空白だけの認識結果を結果に含めないこと(controller, stt_service):
    stt_service.transcribe.return_value = "  \n"
    assert run(controller, SPEECH_THEN_SILENCE) == []


def test正常系_run_foreverメソッドが既定ではVADが発話と判定した小さな声も音声認識へ送ること(controller, stt_service):
    run(controller, SPEECH_THEN_SILENCE, value=0.001)
    stt_service.transcribe.assert_called_once()


def test正常系_run_foreverメソッドがmin_amplitude指定時に振幅の小さい区間を音声認識へ送らないこと(
    make_controller, stt_service
):
    run(make_controller(min_amplitude=0.01), SPEECH_THEN_SILENCE, value=0.001)
    stt_service.transcribe.assert_not_called()


def test正常系_run_foreverメソッドが認識に成功したときコールバックを呼ぶこと(make_controller):
    on_speech_detected = MagicMock()
    controller = make_controller(on_speech_detected=on_speech_detected)
    audio, metadata = run(controller, SPEECH_THEN_SILENCE)[0]
    on_speech_detected.assert_called_once_with(audio, metadata)


def test正常系_run_foreverメソッドが一時的な認識失敗の発話を結果に含めないこと(controller, stt_service):
    stt_service.transcribe.side_effect = RecognitionError("一時的な失敗")
    assert run(controller, SPEECH_THEN_SILENCE) == []


def test正常系_run_foreverメソッドが一時的な認識失敗の後も次の発話を処理すること(controller, stt_service):
    stt_service.transcribe.side_effect = [RecognitionError("一時的な失敗"), "二つ目の発話"]
    results = run(controller, SPEECH_THEN_SILENCE + SPEECH_THEN_SILENCE)
    assert [metadata["text"] for _audio, metadata in results] == ["二つ目の発話"]


def test正常系_run_foreverメソッドが認識失敗時に音声と例外をコールバックへ渡すこと(make_controller, stt_service):
    error = RecognitionError("一時的な失敗")
    stt_service.transcribe.side_effect = error
    on_recognition_error = MagicMock()
    controller = make_controller(on_recognition_error=on_recognition_error)
    run(controller, SPEECH_THEN_SILENCE)
    assert on_recognition_error.call_args.args[1] is error


def test異常系_run_foreverメソッドが継続不能な認識失敗でRecognitionErrorを送出すること(controller, stt_service):
    stt_service.transcribe.side_effect = RecognitionError("認証失敗", fatal=True)
    with pytest.raises(RecognitionError):
        run(controller, SPEECH_THEN_SILENCE)


def test異常系_run_foreverメソッドが継続不能な認識失敗でもコールバックを呼ぶこと(make_controller, stt_service):
    stt_service.transcribe.side_effect = RecognitionError("認証失敗", fatal=True)
    on_recognition_error = MagicMock()
    controller = make_controller(on_recognition_error=on_recognition_error)
    with pytest.raises(RecognitionError):
        run(controller, SPEECH_THEN_SILENCE)
    on_recognition_error.assert_called_once()


def test異常系_run_foreverメソッドがマイク入力の停止でAudioInputErrorを送出すること(controller):
    controller.stream = MagicMock(active=False)
    controller.is_running = True
    with pytest.raises(AudioInputError):
        next(controller.run_forever())


def test正常系_run_foreverメソッドが音声待ちの間もマイクが有効なら待ち続けること(controller):
    def stop_after_check():
        controller.is_running = False
        return True

    stream = MagicMock()
    type(stream).active = PropertyMock(side_effect=stop_after_check)
    controller.stream = stream
    controller.is_running = True
    assert list(controller.run_forever()) == []


def test正常系_run_foreverメソッドが停止状態では何も返さないこと(controller):
    controller.audio_queue.put(chunk())
    assert list(controller.run_forever()) == []


# --- flush ---


def test正常系_flushメソッドが切り出し途中の発話を認識して返すこと(controller):
    run(controller, [True, True])
    assert [metadata["text"] for _audio, metadata in controller.flush()] == ["こんにちは"]


def test正常系_flushメソッドがキューに残った音声も処理すること(controller, stt_service):
    queue_chunks(controller, [True, True, True])
    list(controller.flush())
    assert len(stt_service.transcribe.call_args.args[0]) == 3 * CHUNK


def test正常系_flushメソッドがキューの中で確定した発話と途中の発話の両方を返すこと(controller):
    queue_chunks(controller, SPEECH_THEN_SILENCE + [True])
    assert len(list(controller.flush())) == 2


def test正常系_flushメソッドが発話が無いとき何も返さないこと(controller):
    assert list(controller.flush()) == []


def test正常系_flushメソッドが後続の区間が失敗する前に先行の結果を返すこと(controller, stt_service):
    stt_service.transcribe.side_effect = ["一つ目", RecognitionError("認証失敗", fatal=True)]
    queue_chunks(controller, SPEECH_THEN_SILENCE + [True])
    _audio, metadata = next(controller.flush())
    assert metadata["text"] == "一つ目"


def test異常系_flushメソッドが先行の結果を返した後で継続不能な失敗をRecognitionErrorとして送出すること(
    controller, stt_service
):
    stt_service.transcribe.side_effect = ["一つ目", RecognitionError("認証失敗", fatal=True)]
    queue_chunks(controller, SPEECH_THEN_SILENCE + [True])
    results = controller.flush()
    next(results)
    with pytest.raises(RecognitionError):
        next(results)


def test正常系_flushメソッドが認識を中断された区間をもう一度認識して返すこと(controller, stt_service):
    stt_service.transcribe.side_effect = [KeyboardInterrupt, "中断された発話"]
    with pytest.raises(KeyboardInterrupt):
        run(controller, SPEECH_THEN_SILENCE)
    assert [metadata["text"] for _audio, metadata in controller.flush()] == ["中断された発話"]


def test正常系_flushメソッドが中断された区間と同じ音声を認識へ渡すこと(controller, stt_service):
    stt_service.transcribe.side_effect = [KeyboardInterrupt, "中断された発話"]
    with pytest.raises(KeyboardInterrupt):
        run(controller, SPEECH_THEN_SILENCE)
    list(controller.flush())
    first_audio, second_audio = (call.args[0] for call in stt_service.transcribe.call_args_list)
    assert second_audio is first_audio


def test正常系_flushメソッドが中断された区間をそれより後の音声より先に返すこと(controller, stt_service):
    stt_service.transcribe.side_effect = [KeyboardInterrupt, "中断された発話", "後の発話"]
    with pytest.raises(KeyboardInterrupt):
        run(controller, SPEECH_THEN_SILENCE)
    queue_chunks(controller, [True])
    assert [metadata["text"] for _audio, metadata in controller.flush()] == ["中断された発話", "後の発話"]


def test正常系_flushメソッドが再認識に成功した区間を二重に認識しないこと(controller, stt_service):
    stt_service.transcribe.side_effect = [KeyboardInterrupt, "中断された発話"]
    with pytest.raises(KeyboardInterrupt):
        run(controller, SPEECH_THEN_SILENCE)
    list(controller.flush())
    list(controller.flush())
    assert stt_service.transcribe.call_count == 2


def test正常系_flushメソッドが認識に失敗した区間を再認識しないこと(controller, stt_service):
    stt_service.transcribe.side_effect = RecognitionError("一時的な失敗")
    run(controller, SPEECH_THEN_SILENCE)
    list(controller.flush())
    assert stt_service.transcribe.call_count == 1


def test異常系_flushメソッドが再認識中の再度の中断でKeyboardInterruptを送出すること(controller, stt_service):
    stt_service.transcribe.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run(controller, SPEECH_THEN_SILENCE)
    with pytest.raises(KeyboardInterrupt):
        list(controller.flush())


# --- unprocessed_seconds ---


def test正常系_unprocessed_secondsメソッドが未処理の音声が無いとき0を返すこと(controller):
    assert controller.unprocessed_seconds() == 0


def test正常系_unprocessed_secondsメソッドがキューに残った音声の長さを返すこと(controller):
    queue_chunks(controller, [True, True])
    assert controller.unprocessed_seconds() == pytest.approx(2 * CHUNK / 16000)


def test正常系_unprocessed_secondsメソッドが切り出し途中の発話の長さを返すこと(controller):
    run(controller, [True, True, True])
    assert controller.unprocessed_seconds() == pytest.approx(3 * CHUNK / 16000)


def test正常系_unprocessed_secondsメソッドが認識を中断された区間の長さを返すこと(controller, stt_service):
    stt_service.transcribe.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run(controller, SPEECH_THEN_SILENCE)
    assert controller.unprocessed_seconds() == pytest.approx(3 * CHUNK / 16000)


def test正常系_unprocessed_secondsメソッドがflushで全て処理した後に0を返すこと(controller):
    queue_chunks(controller, [True, True])
    list(controller.flush())
    assert controller.unprocessed_seconds() == 0


# --- process_speech_segment ---


def test正常系_process_speech_segmentメソッドが音声が無いときNoneを返すこと(controller):
    assert controller.process_speech_segment(None) is None


def test正常系_process_speech_segmentメソッドが音声が無いとき音声認識を呼ばないこと(controller, stt_service):
    controller.process_speech_segment(np.array([], dtype=np.float32))
    stt_service.transcribe.assert_not_called()


def test正常系_process_speech_segmentメソッドがサンプルレートをメタデータに入れること(controller):
    _audio, metadata = controller.process_speech_segment(chunk())
    assert metadata["sample_rate"] == 16000


def test正常系_process_speech_segmentメソッドがチャンネル数をメタデータに入れること(controller):
    _audio, metadata = controller.process_speech_segment(chunk())
    assert metadata["channels"] == 1


def test正常系_process_speech_segmentメソッドが既定では小さな音声も認識へ送ること(controller, stt_service):
    controller.process_speech_segment(chunk(0.001))
    stt_service.transcribe.assert_called_once()


def test正常系_process_speech_segmentメソッドがmin_amplitude未満の音声に対してNoneを返すこと(make_controller):
    assert make_controller(min_amplitude=0.01).process_speech_segment(chunk(0.001)) is None


# --- _audio_callback ---


def indata(value=0.5):
    return np.full((CHUNK, 1), value, dtype=np.float32)


def test正常系_audio_callbackメソッドが録音中の音声をモノラルでキューへ積むこと(controller):
    controller.is_running = True
    controller._audio_callback(indata(), CHUNK, None, None)
    assert controller.audio_queue.get_nowait().shape == (CHUNK,)


def test正常系_audio_callbackメソッドが音声を複製して積むこと(controller):
    controller.is_running = True
    buffer = indata(0.5)
    controller._audio_callback(buffer, CHUNK, None, None)
    buffer[:] = 0.0
    assert controller.audio_queue.get_nowait()[0] == pytest.approx(0.5)


def test正常系_audio_callbackメソッドが停止中の音声を積まないこと(controller):
    controller._audio_callback(indata(), CHUNK, None, None)
    assert controller.audio_queue.empty()


def test正常系_audio_callbackメソッドがキュー満杯のとき例外を出さずに取りこぼしを数えること(make_controller):
    controller = make_controller(max_queue_seconds=0.032)
    controller.is_running = True
    controller._audio_callback(indata(), CHUNK, None, None)
    controller._audio_callback(indata(), CHUNK, None, None)
    assert controller._dropped_chunks == 1


def test正常系_audio_callbackメソッドが入力オーバーフローを数えること(controller):
    controller.is_running = True
    controller._audio_callback(indata(), CHUNK, None, MagicMock(input_overflow=True))
    assert controller._input_overflows == 1


# --- _report_input_problems ---


def test正常系_report_input_problemsメソッドが取りこぼしを警告すること(controller, caplog):
    controller._dropped_chunks = 10
    with caplog.at_level(logging.WARNING):
        controller._report_input_problems()
    assert "音声を取りこぼしました" in caplog.text


def test正常系_report_input_problemsメソッドが警告後に件数を戻すこと(controller):
    controller._dropped_chunks = 10
    controller._report_input_problems()
    assert controller._dropped_chunks == 0


def test正常系_report_input_problemsメソッドが取りこぼしが無いとき警告しないこと(controller, caplog):
    with caplog.at_level(logging.WARNING):
        controller._report_input_problems()
    assert caplog.text == ""


def test正常系_report_input_problemsメソッドが短い間隔では警告を繰り返さないこと(controller, caplog):
    controller._dropped_chunks = 10
    controller._report_input_problems()
    caplog.clear()
    controller._dropped_chunks = 10
    with caplog.at_level(logging.WARNING):
        controller._report_input_problems()
    assert caplog.text == ""


# --- start_listening / stop_listening ---


def test正常系_start_listeningメソッドが指定した条件で入力ストリームを開くこと(make_controller, audio_input):
    controller = make_controller(device=3)
    controller.start_listening()
    audio_input.open_stream.assert_called_once_with(
        sample_rate=16000, block_size=CHUNK, device=3, callback=controller._audio_callback
    )


def test正常系_start_listeningメソッドが録音中の状態にすること(controller):
    controller.start_listening()
    assert controller.is_running is True


def test正常系_start_listeningメソッドがVADの状態を初期化すること(controller, vad_service):
    controller.start_listening()
    vad_service.reset.assert_called_once_with()


def test正常系_start_listeningメソッドが開始済みのとき二重にストリームを開かないこと(controller, audio_input):
    controller.start_listening()
    controller.start_listening()
    audio_input.open_stream.assert_called_once()


def test異常系_start_listeningメソッドがマイクを開けないときAudioInputErrorを送出すること(controller, audio_input):
    audio_input.open_stream.side_effect = AudioInputError("マイクを開けません")
    with pytest.raises(AudioInputError):
        controller.start_listening()


def test異常系_start_listeningメソッドがマイクを開けないとき停止状態に戻すこと(controller, audio_input):
    audio_input.open_stream.side_effect = AudioInputError("マイクを開けません")
    with pytest.raises(AudioInputError):
        controller.start_listening()
    assert controller.is_running is False


def test正常系_stop_listeningメソッドがストリームを停止すること(controller, audio_input):
    controller.start_listening()
    controller.stop_listening()
    audio_input.open_stream.return_value.stop.assert_called_once_with()


def test正常系_stop_listeningメソッドがストリームを閉じること(controller, audio_input):
    controller.start_listening()
    controller.stop_listening()
    audio_input.open_stream.return_value.close.assert_called_once_with()


def test正常系_stop_listeningメソッドが停止状態にすること(controller):
    controller.start_listening()
    controller.stop_listening()
    assert controller.is_running is False


def test正常系_stop_listeningメソッドが二度呼ばれてもストリームを一度だけ閉じること(controller, audio_input):
    controller.start_listening()
    controller.stop_listening()
    controller.stop_listening()
    audio_input.open_stream.return_value.close.assert_called_once_with()


def test異常系_stop_listeningメソッドが停止に失敗してもストリームを閉じること(controller, audio_input):
    audio_input.open_stream.return_value.stop.side_effect = RuntimeError("device lost")
    controller.start_listening()
    controller.stop_listening()
    audio_input.open_stream.return_value.close.assert_called_once_with()
