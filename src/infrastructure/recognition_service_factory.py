# infrastructure/recognition_service_factory.py

import importlib.util
import json
import logging
import os
import re

from src.domain.errors import ConfigurationError
from src.infrastructure.google_cloud_recognition_service import INSTALL_HINT, GoogleCloudRecognitionService
from src.infrastructure.groq_recognition_service import GroqRecognitionService
from src.infrastructure.openai_recognition_service import OpenAIRecognitionService
from src.infrastructure.speech_recognition_service import SpeechRecognitionService

logger = logging.getLogger(__name__)

SERVICE_NAMES = ("google", "openai", "groq")

GOOGLE_CREDENTIALS_ENV = "GOOGLE_CLOUD_CREDENTIALS_JSON"
LEGACY_GOOGLE_CREDENTIALS_ENV = "GOOGLE_API_KEY"

# .env.example をそのままコピーした値（your_xxx_here）は未設定として扱う
_PLACEHOLDER_PATTERN = re.compile(r"^your_.*_here$", re.IGNORECASE)


def read_setting(env, name):
    """環境変数を読み、空・プレースホルダならNoneを返す。"""
    value = (env.get(name) or "").strip()
    if not value or _PLACEHOLDER_PATTERN.match(value):
        return None
    return value


def create_recognition_service(service_name, env=None, language=None, model=None):
    """
    指定された音声認識サービスを生成する。

    指定と異なるサービスへ音声を送らないよう、APIキーが無い場合は
    他サービスへ切り替えずにConfigurationErrorを送出する。
    """
    env = os.environ if env is None else env
    service_name = (service_name or "google").lower()

    if service_name == "openai":
        api_key = _require_api_key(env, "OPENAI_API_KEY", "openai")
        logger.info("OpenAI APIを使用した音声認識サービスを初期化します")
        return OpenAIRecognitionService(api_key=api_key, language=language, model=model)

    if service_name == "groq":
        api_key = _require_api_key(env, "GROQ_API_KEY", "groq")
        logger.info("Groq APIを使用した音声認識サービスを初期化します")
        return GroqRecognitionService(api_key=api_key, language=language, model=model)

    if service_name == "google":
        if model:
            raise ConfigurationError("--model は openai / groq でのみ指定できます。")
        credentials = _resolve_google_credentials(env)
        if credentials is None:
            logger.info("Google Speech Recognition無料枠を使用します（リクエスト数に制限あり）")
            return SpeechRecognitionService(language=language)
        logger.info("Google Cloud Speech-to-Text（サービスアカウント）を使用します")
        return GoogleCloudRecognitionService.from_service_account(language=language, **credentials)

    raise ConfigurationError(
        f"未対応の音声認識サービスです: {service_name}（指定できる値: {', '.join(SERVICE_NAMES)}）"
    )


def _require_api_key(env, name, service_name):
    api_key = read_setting(env, name)
    if api_key is None:
        raise ConfigurationError(
            f"-s {service_name} には環境変数 {name} が必要です。.env に設定するか、-s google を指定してください。"
        )
    return api_key


def _resolve_google_credentials(env):
    """
    Google Cloudのサービスアカウント認証情報を、from_service_account へ渡す引数として返す。

    未設定ならNone（無料枠を使う）。値はJSONファイルのパスのほか、
    旧 GOOGLE_API_KEY で使われていたJSON文字列そのものも受け付ける。
    """
    name = GOOGLE_CREDENTIALS_ENV
    value = read_setting(env, name)
    if value is None:
        name = LEGACY_GOOGLE_CREDENTIALS_ENV
        value = read_setting(env, name)
        if value is None:
            return None
        logger.warning(
            "%s は旧名称です。%s にサービスアカウントJSONのパスを設定する形へ移行してください。",
            LEGACY_GOOGLE_CREDENTIALS_ENV,
            GOOGLE_CREDENTIALS_ENV,
        )

    # 値は秘密情報でありうるため、エラーの文面には含めない
    if value.startswith("{"):
        try:
            credentials = {"credentials_info": json.loads(value)}
        except ValueError:
            raise ConfigurationError(f"{name} のJSON文字列を解釈できません。") from None
    elif os.path.isfile(value):
        credentials = {"credentials_path": value}
    else:
        raise ConfigurationError(
            f"{name} はサービスアカウントJSONのファイルパスとしてもJSON文字列としても解釈できません。"
            f"無料枠を使う場合は {name} を削除してください。"
        )

    if not _is_google_cloud_speech_installed():
        raise ConfigurationError(f"Google Cloud Speech-to-Text には追加パッケージが必要です。{INSTALL_HINT}")
    return credentials


def _is_google_cloud_speech_installed():
    try:
        return importlib.util.find_spec("google.cloud.speech") is not None
    except ModuleNotFoundError:
        return False
