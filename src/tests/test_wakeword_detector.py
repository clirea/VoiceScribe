import pytest
from domain.wakeword_detector import WakewordDetector

@pytest.fixture
def detector():
    return WakewordDetector()

def test正常系_detectメソッドがデフォルトのウェイクワードを検出すること(detector):
    # デフォルトのウェイクワード "アウラ" を含むテキスト
    result = detector.detect("こんにちは、アウラです")
    assert result == True

def test正常系_detectメソッドが大文字小文字を区別せずウェイクワードを検出すること(detector):
    # 大文字小文字が混在したウェイクワード
    result = detector.detect("Hello, AURA here")
    assert result == True

def test正常系_detectメソッドがカスタムウェイクワードを検出すること():
    # カスタムウェイクワードでの初期化
    custom_detector = WakewordDetector(wakewords=["フリーレン"])
    result = custom_detector.detect("フリーレンが現れた")
    assert result == True

def test異常系_detectメソッドが空文字列に対してFalseを返すこと(detector):
    result = detector.detect("")
    assert result == False

def test異常系_detectメソッドがNoneに対してFalseを返すこと(detector):
    result = detector.detect(None)
    assert result == False

def test正常系_detectメソッドがウェイクワードを含まない文字列に対してFalseを返すこと(detector):
    result = detector.detect("こんにちは、私は魔族です")
    assert result == False

def test正常系_detectメソッドが複数のウェイクワードのいずれかを検出すること(detector):
    # デフォルトの "あうら" を含むテキスト
    result = detector.detect("あうらさんこんにちは")
    assert result == True
