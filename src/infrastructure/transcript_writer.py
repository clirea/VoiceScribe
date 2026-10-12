# infrastructure/transcript_writer.py

import os

from src.domain.errors import ConfigurationError, TranscriptWriteError


class TranscriptWriter:
    """音声認識結果をテキストファイルに追記するクラス。"""

    def __init__(self, output_file="output.txt"):
        self.output_file = output_file

    def prepare(self):
        """録音を始める前に、出力先へ書き込めることを確かめる（既存の内容は変更しない）。"""
        try:
            # 出力ディレクトリがなければ作成
            output_dir = os.path.dirname(self.output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(self.output_file, "a", encoding="utf-8"):
                pass
        except OSError as e:
            raise ConfigurationError(f"出力ファイルに書き込めません: {self.output_file} ({e})") from e

    def append(self, metadata) -> bool:
        """
        タイムスタンプと認識テキストを1行で追記する。書くものが無ければFalseを返す。
        録音中に出力先へ書けなくなった場合はTranscriptWriteErrorを送出する。
        """
        text = (metadata or {}).get("text")
        if not text or not text.strip():
            return False

        prefix = "ウェイクワード" if metadata["is_wake_word"] else "音声認識"
        # 1発話を1行に保つ
        single_line_text = " ".join(text.split())

        try:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(f"[{metadata['timestamp']}] {prefix}: {single_line_text}\n")
        except OSError as e:
            raise TranscriptWriteError(f"認識結果を保存できませんでした: {self.output_file} ({e})") from e
        return True
