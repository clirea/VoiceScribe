# tests/test_transcript_writer.py

import pytest

from src.domain.errors import ConfigurationError, TranscriptWriteError
from src.infrastructure.transcript_writer import TranscriptWriter


def metadata(text="こんにちは", is_wake_word=False):
    return {"timestamp": "20261012_101530", "is_wake_word": is_wake_word, "text": text}


@pytest.fixture
def output_file(tmp_path):
    return tmp_path / "output.txt"


@pytest.fixture
def writer(output_file):
    return TranscriptWriter(str(output_file))


def test正常系_appendメソッドがタイムスタンプ付きの1行を書き込むこと(writer, output_file):
    writer.append(metadata())
    assert output_file.read_text(encoding="utf-8") == "[20261012_101530] 音声認識: こんにちは\n"


def test正常系_appendメソッドがウェイクワードの行に専用の接頭辞を付けること(writer, output_file):
    writer.append(metadata("アウラ", is_wake_word=True))
    assert output_file.read_text(encoding="utf-8") == "[20261012_101530] ウェイクワード: アウラ\n"


def test正常系_appendメソッドが既存の内容を残して追記すること(writer, output_file):
    output_file.write_text("既存の行\n", encoding="utf-8")
    writer.append(metadata())
    assert output_file.read_text(encoding="utf-8").splitlines()[0] == "既存の行"


def test正常系_appendメソッドが改行を含むテキストを1行にまとめること(writer, output_file):
    writer.append(metadata("一行目\n二行目"))
    assert output_file.read_text(encoding="utf-8") == "[20261012_101530] 音声認識: 一行目 二行目\n"


def test正常系_appendメソッドが書き込んだときTrueを返すこと(writer):
    assert writer.append(metadata()) is True


def test正常系_appendメソッドが空文字のときFalseを返すこと(writer):
    assert writer.append(metadata("")) is False


def test正常系_appendメソッドが空白だけのテキストを書き込まないこと(writer, output_file):
    writer.append(metadata("  \n"))
    assert not output_file.exists()


def test正常系_appendメソッドがメタデータ無しのときFalseを返すこと(writer):
    assert writer.append(None) is False


def test正常系_prepareメソッドが出力ディレクトリを作成すること(tmp_path):
    TranscriptWriter(str(tmp_path / "notes" / "daily" / "output.txt")).prepare()
    assert (tmp_path / "notes" / "daily").is_dir()


def test正常系_prepareメソッドが既存の内容を変更しないこと(writer, output_file):
    output_file.write_text("既存の行\n", encoding="utf-8")
    writer.prepare()
    assert output_file.read_text(encoding="utf-8") == "既存の行\n"


def test異常系_prepareメソッドが書き込めない出力先でConfigurationErrorを送出すること(tmp_path):
    with pytest.raises(ConfigurationError):
        TranscriptWriter(str(tmp_path)).prepare()



def test異常系_appendメソッドが書き込めなくなった出力先でTranscriptWriteErrorを送出すること(tmp_path):
    with pytest.raises(TranscriptWriteError):
        TranscriptWriter(str(tmp_path)).append(metadata())


def test異常系_appendメソッドが保存できなかった出力先をエラーに含めること(tmp_path):
    with pytest.raises(TranscriptWriteError, match="認識結果を保存できませんでした"):
        TranscriptWriter(str(tmp_path)).append(metadata())
