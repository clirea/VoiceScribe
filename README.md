# VoiceScribe

音声認識を使用してリアルタイムで音声をテキストに変換するPythonアプリケーションです。ウェイクワード検出と音声認識機能を備えています。

## プロジェクトの概要と目的

VoiceScribeは、リアルタイムの音声認識技術を活用して、話された言葉を自動的にテキストに変換するアプリケーションです。このプロジェクトは以下の目的で開発されました：

- 音声データのリアルタイム処理と認識の実装方法を学ぶ
- クリーンアーキテクチャに基づいたPythonアプリケーションの設計方法を実践する
- 最新の音声認識APIを活用した実用的なアプリケーションの開発

このアプリケーションは、会議の議事録作成、講義のノート取り、アクセシビリティ向上など、様々な用途に活用できます。

## 技術的な詳細

VoiceScribeは以下の技術を使用しています：

- **音声処理**: NumPy, SoundDeviceを使用した音声データの取得と処理
- **音声活動検出(VAD)**: Silero VADを使用して発話区間を検出
- **音声認識**: 複数の音声認識サービスをサポート
  - Google Speech Recognition（デフォルト、無料枠あり）
  - OpenAI Whisperモデル（APIキー必要）
  - Groq API（APIキー必要）
- **アーキテクチャ**: クリーンアーキテクチャに基づいた層分け設計
  - ドメイン層: 音声検出、ウェイクワード検出などのコアロジック
  - アプリケーション層: 音声コントローラーなどのユースケース実装
  - インフラストラクチャ層: 外部APIとの連携

## 音声認識サービスについて

VoiceScribeは複数の音声認識サービスをサポートしています：

1. **Google Speech Recognition（デフォルト）**:
   - 無料枠を使用（APIキー不要）
   - 1分間に20回までのリクエスト制限あり
   - Google Cloud Speech APIキーを設定することで制限なしで使用可能

2. **OpenAI Whisper**:
   - 高精度な音声認識が可能
   - OpenAI APIキーが必要（有料）

3. **Groq API**:
   - Whisper Large V3 Turboモデルを使用
   - Groq APIキーが必要（有料）

## 学習目的と活用方法

このプロジェクトは以下のような学習目的に最適です：

- **音声処理技術の学習**: リアルタイム音声処理のパイプラインを理解する
- **クリーンアーキテクチャの実践**: 関心の分離と依存性の逆転を実践的に学ぶ
- **APIインテグレーション**: 外部APIを効果的に活用する方法を学ぶ
- **Pythonプログラミング**: 非同期処理、コールバック、イベント駆動プログラミングを学ぶ

実際の活用例：
- オンライン会議の自動文字起こし
- 講義や講演の記録
- 音声コマンドによるアプリケーション制御
- 音声データの収集と分析

## 機能

- リアルタイム音声認識
- 認識結果のテキストファイル出力
- 複数のマイクデバイスのサポート
- ウェイクワード検出※このPJTでは使用していません
- 複数の音声認識サービスの切り替え

## 必要条件

- Python 3.8以上
- 必要なライブラリ（requirements.txtに記載）

## インストール方法

1. リポジトリをクローンします
```bash
git clone https://github.com/yourusername/VoiceScribe.git
cd VoiceScribe
```

2. 必要なライブラリをインストールします
```bash
pip install -r requirements.txt
```

3. 環境設定ファイルを準備します
```bash
cp .env.example .env
```
`.env`ファイルを編集して、必要なAPIキーを設定してください。
- OpenAI API Key（Whisperモデルを使用する場合）
- Groq API Key（Groqの音声認識を使用する場合）
- Google Cloud Speech API Credentials JSON（Google Cloud Speech APIを使用する場合）

## 使用方法

### 基本的な使い方
```bash
python main.py
```

### 出力ファイルを指定する
```bash
python main.py -o path/to/output.txt
```

### 特定のマイクデバイスを使用する
```bash
# デバイスIDで指定
python main.py -d 1

# デバイス名で指定（部分一致）
python main.py -n "マイク名"
```

### 音声認識サービスを指定する
```bash
# Google Speech Recognition（デフォルト）
python main.py -s google

# OpenAI Whisper
python main.py -s openai

# Groq API
python main.py -s groq
```

### 複数のオプションを同時に使用する

VoiceScribeでは、複数のコマンドラインオプションを同時に指定することができます。以下に一般的な使用例を示します：

```bash
# 特定のマイクと音声認識サービスを指定
python main.py -d 1 -s openai

# マイク名と出力ファイルを指定
python main.py -n "Headset" -o "meeting_notes.txt"

# すべてのオプションを組み合わせる
python main.py -n "マイク名" -s groq -o "output/transcript.txt"
```

### 実用的な使用例

以下に、実際のシナリオに基づいた使用例を示します：

1. **会議の議事録作成**（高精度が必要な場合）:
```bash
python main.py -s openai -o "meetings/2024-02-26_board_meeting.txt"
```

2. **講義の録音**（特定のマイクを使用）:
```bash
python main.py -n "Blue Yeti" -o "lectures/physics_101.txt"
```

3. **日常的な使用**（無料枠で十分な場合）:
```bash
python main.py -o "notes/daily_notes.txt"
```

### 利用可能なマイクデバイスを表示する
```bash
python main.py -l
```

### マイクのテスト
```bash
python mic_check.py
```

## プロジェクト構造

```
VoiceScribe/
├── main.py              # メインアプリケーション
├── mic_check.py         # マイクテスト用スクリプト
├── requirements.txt     # 必要なライブラリ
├── .env                 # 環境設定ファイル（APIキーなど）
├── .env.example         # 環境設定ファイルのサンプル
└── src/                 # ソースコード
    ├── application/     # アプリケーション層
    │   └── audio_controller.py  # 音声処理コントローラー
    ├── domain/          # ドメイン層
    │   ├── vad_service.py       # 音声活動検出サービス
    │   └── wakeword_detector.py # ウェイクワード検出
    ├── infrastructure/  # インフラストラクチャ層
    │   ├── speech_recognition_service.py  # Google音声認識サービス
    │   ├── openai_recognition_service.py  # OpenAI音声認識実装
    │   └── groq_recognition_service.py    # Groq音声認識実装
    └── tests/           # テストコード
```

## 拡張方法

VoiceScribeは拡張性を考慮して設計されています。以下のような拡張が可能です：

1. **新しい音声認識サービスの追加**:
   - `src/infrastructure`に新しい認識サービスクラスを追加
   - `speech_recognition_service.py`のインターフェースを実装

2. **ウェイクワード検出の有効化**:
   - `src/domain/wakeword_detector.py`を拡張
   - 特定のキーワードに反応するロジックを実装

3. **UIの追加**:
   - PyQtやTkinterを使用したGUIの実装
   - ウェブインターフェースの追加（Flask/FastAPIなど）

## トラブルシューティング

- **マイクが認識されない場合**: `mic_check.py`を実行して、利用可能なマイクデバイスを確認
- **音声認識が動作しない場合**: `.env`ファイルのAPIキーが正しく設定されているか確認
- **パフォーマンスの問題**: ブロックサイズやサンプルレートを調整して最適化
- **Google Speech Recognitionのリクエスト制限**: 無料枠では1分間に20回までのリクエスト制限があります。頻繁に使用する場合はGoogle Cloud Speech APIキーを設定するか、他のサービスを使用してください。

## ライセンス

[MIT License](LICENSE)

## 貢献方法

1. このリポジトリをフォークします
2. 機能ブランチを作成します (`git checkout -b feature/amazing-feature`)
3. 変更をコミットします (`git commit -m 'Add some amazing feature'`)
4. ブランチにプッシュします (`git push origin feature/amazing-feature`)
5. プルリクエストを作成します 
