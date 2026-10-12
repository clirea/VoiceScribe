# infrastructure/google_cloud_recognition_service.py

import numpy as np

from src.domain.errors import ConfigurationError, RecognitionError
from src.infrastructure.wav_encoder import to_pcm16

DEFAULT_LANGUAGE = "ja-JP"
INSTALL_HINT = "`pip install -r requirements-google-cloud.txt` を実行してください。"


class GoogleCloudRecognitionService:
    """
    公式SDK（google-cloud-speech）でGoogle Cloud Speech-to-Text V1の同期認識を行うクラス。

    同期認識が受け付ける音声は1回あたり約60秒までのため、長い発話は --max-segment-seconds で区切る。
    """

    def __init__(self, language=None, client=None, timeout=30.0):
        try:
            from google.api_core import exceptions as api_exceptions
            from google.auth import exceptions as auth_exceptions
            from google.cloud import speech
        except ImportError as e:
            raise ConfigurationError(f"google-cloud-speech を読み込めません。{INSTALL_HINT}") from e

        self._speech = speech
        self._api_exceptions = api_exceptions
        self._auth_exceptions = auth_exceptions
        self.language = language or DEFAULT_LANGUAGE
        self.timeout = timeout
        self.client = client if client is not None else self._create_default_client()

    @classmethod
    def from_service_account(cls, credentials_path=None, credentials_info=None, language=None, timeout=30.0):
        """サービスアカウントJSONのファイルパス、または読み込み済みの内容（dict）からサービスを生成する。"""
        try:
            from google.cloud import speech
        except ImportError as e:
            raise ConfigurationError(f"google-cloud-speech を読み込めません。{INSTALL_HINT}") from e

        # 認証情報の中身が例外の文面や連鎖に残らないよう、元の例外は引き継がない
        try:
            if credentials_info is not None:
                client = speech.SpeechClient.from_service_account_info(credentials_info)
            else:
                client = speech.SpeechClient.from_service_account_file(credentials_path)
        except Exception as e:
            raise ConfigurationError(
                "Google Cloudのサービスアカウント認証情報を読み込めません"
                f"（{type(e).__name__}）。サービスアカウントキーのJSONであることを確認してください。"
            ) from None
        return cls(language=language, client=client, timeout=timeout)

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        NumPy配列を16bit PCMに変換して音声認識を行う。
        聞き取れなかった場合は空文字を返し、APIの失敗はRecognitionErrorにする。
        """
        if audio_data is None or audio_data.size == 0:
            return ""

        speech = self._speech
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=self.language,
        )
        audio = speech.RecognitionAudio(content=to_pcm16(audio_data))

        try:
            # SDK既定の再試行は期限が5000秒あるため無効にし、1回の呼び出しをtimeoutで打ち切る
            response = self.client.recognize(config=config, audio=audio, timeout=self.timeout, retry=None)
        except self._api_exceptions.GoogleAPICallError as e:
            raise self._to_recognition_error(e) from e
        except self._auth_exceptions.GoogleAuthError as e:
            # 認証系の例外は文面に応答内容を含みうるため、種類だけを伝える
            raise RecognitionError(
                f"Google Cloudの認証に失敗しました（{type(e).__name__}）。サービスアカウントキーを確認してください。",
                fatal=True,
            ) from None
        except Exception as e:
            raise RecognitionError(f"Google Cloud音声認識で予期せぬエラーが発生しました: {e}") from e

        transcripts = [result.alternatives[0].transcript.strip() for result in response.results if result.alternatives]
        return " ".join(transcripts)

    def _create_default_client(self):
        try:
            return self._speech.SpeechClient()
        except self._auth_exceptions.GoogleAuthError as e:
            raise ConfigurationError(
                f"Google Cloudの認証情報が見つかりません（{type(e).__name__}）。"
                "GOOGLE_CLOUD_CREDENTIALS_JSON にサービスアカウントJSONのパスを設定してください。"
            ) from None

    def _to_recognition_error(self, error) -> RecognitionError:
        """認証・権限・対象なしは再試行しても回復しないため、継続不能として扱う。"""
        fatal_errors = (
            self._api_exceptions.Unauthenticated,
            self._api_exceptions.PermissionDenied,
            self._api_exceptions.NotFound,
        )
        return RecognitionError(
            f"Google Cloud音声認識へのリクエストに失敗しました: {error}", fatal=isinstance(error, fatal_errors)
        )
