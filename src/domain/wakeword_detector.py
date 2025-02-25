# domain/wakeword_detector.py

class WakewordDetector:
    """
    ウェイクワード（例：アウラ）が含まれるかどうかを
    文字列解析で判定するクラス。
    """

    def __init__(self, wakewords=None):
        # デフォルトのウェイクワードを設定
        self.wakewords = wakewords if wakewords else ["アウラ", "あうら", "aura"]

    def detect(self, recognized_text: str) -> bool:
        """
        音声認識結果にウェイクワードが含まれるかどうかを返す。
        """
        if not recognized_text:
            return False
        for w in self.wakewords:
            if w.lower() in recognized_text.lower():
                return True
        return False
