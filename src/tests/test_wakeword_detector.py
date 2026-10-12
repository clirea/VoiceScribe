# tests/test_wakeword_detector.py

import pytest

from src.domain.wakeword_detector import WakewordDetector


@pytest.fixture
def detector():
    return WakewordDetector()


def test正常系_detectメソッドがデフォルトのウェイクワードを検出すること(detector):
    # デフォルトのウェイクワード "アウラ" を含むテキスト
    assert detector.detect("こんにちは、アウラです") is True


def test正常系_detectメソッドが大文字小文字を区別せずウェイクワードを検出すること(detector):
    # 大文字小文字が混在したウェイクワード
    assert detector.detect("Hello, AURA here") is True


def test正常系_detectメソッドがカスタムウェイクワードを検出すること():
    # カスタムウェイクワードでの初期化
    custom_detector = WakewordDetector(wakewords=["フリーレン"])
    assert custom_detector.detect("フリーレンが現れた") is True


def test正常系_detectメソッドがカスタム指定時にデフォルトのウェイクワードを検出しないこと():
    custom_detector = WakewordDetector(wakewords=["フリーレン"])
    assert custom_detector.detect("アウラです") is False


def test異常系_detectメソッドが空文字列に対してFalseを返すこと(detector):
    assert detector.detect("") is False


def test異常系_detectメソッドがNoneに対してFalseを返すこと(detector):
    assert detector.detect(None) is False


def test正常系_detectメソッドがウェイクワードを含まない文字列に対してFalseを返すこと(detector):
    assert detector.detect("こんにちは、私は魔族です") is False


def test正常系_detectメソッドが複数のウェイクワードのいずれかを検出すること(detector):
    # デフォルトの "あうら" を含むテキスト
    assert detector.detect("あうらさんこんにちは") is True
