# infrastructure/whisper_api_support.py

from src.domain.errors import RecognitionError

# 再試行しても回復しないHTTPステータス（認証失敗・権限なし・モデルなし）
FATAL_STATUS_CODES = frozenset({401, 403, 404})


def to_whisper_language(language):
    """"ja-JP" のようなBCP-47タグを、Whisper系APIが受け付けるISO-639-1コード（"ja"）に変換する。"""
    if not language:
        return None
    return language.replace("_", "-").split("-")[0].lower()


def to_recognition_error(service_label: str, error: Exception) -> RecognitionError:
    """OpenAI互換SDKの例外を、継続可否つきのRecognitionErrorに変換する。"""
    fatal = getattr(error, "status_code", None) in FATAL_STATUS_CODES
    return RecognitionError(f"{service_label}の音声認識に失敗しました: {error}", fatal=fatal)
