# domain/errors.py


class VoiceScribeError(Exception):
    """VoiceScribeが利用者へ理由を伝えて終了・継続を判断するための基底例外。"""


class ConfigurationError(VoiceScribeError):
    """APIキー・マイク指定・依存ライブラリなど、設定の不備を表す。"""


class AudioInputError(VoiceScribeError):
    """マイク入力を開始・継続できないことを表す。"""


class VoiceActivityDetectionError(VoiceScribeError):
    """VADモデルで音声判定ができないことを表す。"""


class TranscriptWriteError(VoiceScribeError):
    """認識結果を出力ファイルへ保存できないことを表す。"""


class RecognitionError(VoiceScribeError):
    """
    音声認識の失敗を表す。

    fatal=True は認証エラーなど、再試行しても回復しない失敗を示す。
    """

    def __init__(self, message: str, *, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal
