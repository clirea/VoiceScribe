# tests/test_recognition_service_factory.py

import logging
from unittest.mock import patch

import pytest

from src.domain.errors import ConfigurationError
from src.infrastructure import recognition_service_factory as factory
from src.infrastructure.recognition_service_factory import create_recognition_service, read_setting


@pytest.fixture
def service_classes():
    """各サービスクラスを差し替え、SDKのクライアントを生成しないようにする"""
    with (
        patch.object(factory, "OpenAIRecognitionService") as openai_service,
        patch.object(factory, "GroqRecognitionService") as groq_service,
        patch.object(factory, "SpeechRecognitionService") as google_service,
        patch.object(factory, "GoogleCloudRecognitionService") as google_cloud_service,
    ):
        yield {
            "openai": openai_service,
            "groq": groq_service,
            "google": google_service,
            "google_cloud": google_cloud_service,
        }


@pytest.fixture
def credentials_file(tmp_path):
    path = tmp_path / "service_account.json"
    path.write_text("{}", encoding="utf-8")
    return str(path)


@pytest.fixture
def google_cloud_installed():
    with patch.object(factory, "_is_google_cloud_speech_installed", return_value=True):
        yield


def test正常系_create_recognition_serviceメソッドがopenai指定でOpenAIのサービスを返すこと(service_classes):
    service = create_recognition_service("openai", {"OPENAI_API_KEY": "sk-test"})
    assert service is service_classes["openai"].return_value


def test正常系_create_recognition_serviceメソッドがOpenAIへキーと言語とモデルを渡すこと(service_classes):
    create_recognition_service("openai", {"OPENAI_API_KEY": "sk-test"}, language="ja-JP", model="custom-model")
    service_classes["openai"].assert_called_once_with(api_key="sk-test", language="ja-JP", model="custom-model")


def test正常系_create_recognition_serviceメソッドがgroq指定でGroqのサービスを返すこと(service_classes):
    service = create_recognition_service("groq", {"GROQ_API_KEY": "gsk-test"})
    assert service is service_classes["groq"].return_value


def test正常系_create_recognition_serviceメソッドがGroqへキーと言語とモデルを渡すこと(service_classes):
    create_recognition_service("groq", {"GROQ_API_KEY": "gsk-test"}, language="en-US", model="custom-model")
    service_classes["groq"].assert_called_once_with(api_key="gsk-test", language="en-US", model="custom-model")


def test正常系_create_recognition_serviceメソッドがサービス名の大文字小文字を区別しないこと(service_classes):
    service = create_recognition_service("OpenAI", {"OPENAI_API_KEY": "sk-test"})
    assert service is service_classes["openai"].return_value


@pytest.mark.parametrize(
    "service_name, env",
    [
        pytest.param("openai", {}, id="openai-未設定"),
        pytest.param("openai", {"OPENAI_API_KEY": "  "}, id="openai-空白"),
        pytest.param("openai", {"OPENAI_API_KEY": "your_openai_api_key_here"}, id="openai-プレースホルダ"),
        pytest.param("groq", {}, id="groq-未設定"),
        pytest.param("groq", {"GROQ_API_KEY": "your_groq_api_key_here"}, id="groq-プレースホルダ"),
    ],
)
def test異常系_create_recognition_serviceメソッドがAPIキー無しでConfigurationErrorを送出すること(
    service_classes, service_name, env
):
    with pytest.raises(ConfigurationError):
        create_recognition_service(service_name, env)


@pytest.mark.parametrize("service_name", ["openai", "groq"])
def test異常系_create_recognition_serviceメソッドがAPIキー無しでもGoogleへ切り替えないこと(
    service_classes, service_name
):
    with pytest.raises(ConfigurationError):
        create_recognition_service(service_name, {})
    service_classes["google"].assert_not_called()


def test正常系_create_recognition_serviceメソッドがgoogle指定で無料枠のサービスを生成すること(service_classes):
    create_recognition_service("google", {})
    service_classes["google"].assert_called_once_with(language=None)


def test正常系_create_recognition_serviceメソッドがサービス名未指定のときgoogleを使うこと(service_classes):
    service = create_recognition_service(None, {})
    assert service is service_classes["google"].return_value


def test正常系_create_recognition_serviceメソッドがGoogleへ言語を渡すこと(service_classes):
    create_recognition_service("google", {}, language="en-US")
    service_classes["google"].assert_called_once_with(language="en-US")


def test正常系_create_recognition_serviceメソッドがサンプルのままのGOOGLE_API_KEYを無視すること(service_classes):
    env = {"GOOGLE_API_KEY": "your_google_cloud_speech_api_credentials_json_here"}
    create_recognition_service("google", env)
    service_classes["google"].assert_called_once_with(language=None)


def test正常系_create_recognition_serviceメソッドが認証情報のパスでGoogle_Cloudのサービスを返すこと(
    service_classes, credentials_file, google_cloud_installed
):
    service = create_recognition_service("google", {"GOOGLE_CLOUD_CREDENTIALS_JSON": credentials_file})
    assert service is service_classes["google_cloud"].from_service_account.return_value


def test正常系_create_recognition_serviceメソッドが認証情報のパスと言語をGoogle_Cloudへ渡すこと(
    service_classes, credentials_file, google_cloud_installed
):
    create_recognition_service("google", {"GOOGLE_CLOUD_CREDENTIALS_JSON": credentials_file}, language="en-US")
    service_classes["google_cloud"].from_service_account.assert_called_once_with(
        language="en-US", credentials_path=credentials_file
    )


def test正常系_create_recognition_serviceメソッドが認証情報のあるとき無料枠のサービスを生成しないこと(
    service_classes, credentials_file, google_cloud_installed
):
    create_recognition_service("google", {"GOOGLE_CLOUD_CREDENTIALS_JSON": credentials_file})
    service_classes["google"].assert_not_called()


def test正常系_create_recognition_serviceメソッドが旧名称GOOGLE_API_KEYのファイルパスを受け付けること(
    service_classes, credentials_file, google_cloud_installed
):
    create_recognition_service("google", {"GOOGLE_API_KEY": credentials_file})
    service_classes["google_cloud"].from_service_account.assert_called_once_with(
        language=None, credentials_path=credentials_file
    )


def test正常系_create_recognition_serviceメソッドが旧名称GOOGLE_API_KEYのJSON文字列を受け付けること(
    service_classes, google_cloud_installed
):
    env = {"GOOGLE_API_KEY": ' {"type": "service_account", "project_id": "demo"} '}
    create_recognition_service("google", env)
    service_classes["google_cloud"].from_service_account.assert_called_once_with(
        language=None, credentials_info={"type": "service_account", "project_id": "demo"}
    )


def test正常系_create_recognition_serviceメソッドが新名称のJSON文字列も受け付けること(
    service_classes, google_cloud_installed
):
    create_recognition_service("google", {"GOOGLE_CLOUD_CREDENTIALS_JSON": '{"type": "service_account"}'})
    service_classes["google_cloud"].from_service_account.assert_called_once_with(
        language=None, credentials_info={"type": "service_account"}
    )


def test正常系_create_recognition_serviceメソッドが旧名称の使用時に新名称への移行を促すこと(
    service_classes, credentials_file, google_cloud_installed, caplog
):
    with caplog.at_level(logging.WARNING, logger=factory.__name__):
        create_recognition_service("google", {"GOOGLE_API_KEY": credentials_file})
    assert "GOOGLE_CLOUD_CREDENTIALS_JSON" in caplog.text


def test正常系_create_recognition_serviceメソッドが新旧両方あるとき新名称を優先すること(
    service_classes, credentials_file, google_cloud_installed
):
    env = {"GOOGLE_CLOUD_CREDENTIALS_JSON": credentials_file, "GOOGLE_API_KEY": "legacy-value"}
    create_recognition_service("google", env)
    service_classes["google_cloud"].from_service_account.assert_called_once_with(
        language=None, credentials_path=credentials_file
    )


def test異常系_create_recognition_serviceメソッドが存在しない認証情報パスでConfigurationErrorを送出すること(
    service_classes, google_cloud_installed
):
    with pytest.raises(ConfigurationError):
        create_recognition_service("google", {"GOOGLE_CLOUD_CREDENTIALS_JSON": "missing/service_account.json"})


def test異常系_create_recognition_serviceメソッドが壊れたJSON文字列のGOOGLE_API_KEYでConfigurationErrorを送出すること(
    service_classes, google_cloud_installed
):
    with pytest.raises(ConfigurationError):
        create_recognition_service("google", {"GOOGLE_API_KEY": '{"type": "service_account", "private_key": '})


@pytest.mark.parametrize(
    "value",
    [
        pytest.param('{"private_key": "SECRET-MARKER", ', id="壊れたJSON文字列"),
        pytest.param("SECRET-MARKER", id="パスでもJSONでもない値"),
    ],
)
def test異常系_create_recognition_serviceメソッドが解釈できない認証情報の値をエラーに含めないこと(
    service_classes, google_cloud_installed, value
):
    with pytest.raises(ConfigurationError) as error:
        create_recognition_service("google", {"GOOGLE_API_KEY": value})
    assert "SECRET-MARKER" not in str(error.value)


def test異常系_create_recognition_serviceメソッドが壊れたJSON文字列の解析例外を連鎖させないこと(
    service_classes, google_cloud_installed
):
    with pytest.raises(ConfigurationError) as error:
        create_recognition_service("google", {"GOOGLE_API_KEY": '{"private_key": "SECRET-MARKER", '})
    assert error.value.__cause__ is None


def test異常系_create_recognition_serviceメソッドが認証情報の読み込み失敗でも無料枠へ切り替えないこと(
    service_classes, google_cloud_installed
):
    service_classes["google_cloud"].from_service_account.side_effect = ConfigurationError("読み込めません")
    with pytest.raises(ConfigurationError):
        create_recognition_service("google", {"GOOGLE_API_KEY": '{"type": "service_account"}'})
    service_classes["google"].assert_not_called()


def test異常系_create_recognition_serviceメソッドがgoogle_cloud_speech未導入でConfigurationErrorを送出すること(
    service_classes, credentials_file
):
    with patch.object(factory, "_is_google_cloud_speech_installed", return_value=False):
        with pytest.raises(ConfigurationError):
            create_recognition_service("google", {"GOOGLE_CLOUD_CREDENTIALS_JSON": credentials_file})


def test異常系_create_recognition_serviceメソッドがgoogleへのモデル指定でConfigurationErrorを送出すること(
    service_classes,
):
    with pytest.raises(ConfigurationError):
        create_recognition_service("google", {}, model="whisper-1")


def test異常系_create_recognition_serviceメソッドが未対応のサービス名でConfigurationErrorを送出すること(
    service_classes,
):
    with pytest.raises(ConfigurationError):
        create_recognition_service("azure", {})


def test正常系_create_recognition_serviceメソッドがenv未指定のとき環境変数を読むこと(service_classes, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-environ")
    create_recognition_service("openai")
    service_classes["openai"].assert_called_once_with(api_key="sk-from-environ", language=None, model=None)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("sk-test", "sk-test"),
        ("  sk-test  ", "sk-test"),
        ("", None),
        ("   ", None),
        ("your_openai_api_key_here", None),
        ("YOUR_GROQ_API_KEY_HERE", None),
    ],
)
def test正常系_read_settingメソッドが空とプレースホルダを未設定として扱うこと(value, expected):
    assert read_setting({"NAME": value}, "NAME") == expected


def test正常系_read_settingメソッドが存在しない名前に対してNoneを返すこと():
    assert read_setting({}, "NAME") is None


def test正常系_is_google_cloud_speech_installedメソッドが未導入のときFalseを返すこと():
    with patch.object(factory.importlib.util, "find_spec", side_effect=ModuleNotFoundError):
        assert factory._is_google_cloud_speech_installed() is False


def test正常系_is_google_cloud_speech_installedメソッドが導入済みのときTrueを返すこと():
    with patch.object(factory.importlib.util, "find_spec", return_value=object()):
        assert factory._is_google_cloud_speech_installed() is True
