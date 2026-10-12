# VoiceScribe

音声認識を使用してリアルタイムで音声をテキストに変換するPythonアプリケーションです。マイクの音声から発話区間を切り出し、音声認識サービスで文字起こしした結果をテキストファイルへ追記します。

## プロジェクトの概要と目的

VoiceScribeは、リアルタイムの音声認識技術を活用して、話された言葉を自動的にテキストに変換するアプリケーションです。このプロジェクトは以下の目的で開発されました：

- 音声データのリアルタイム処理と認識の実装方法を学ぶ
- クリーンアーキテクチャに基づいたPythonアプリケーションの設計方法を実践する
- 最新の音声認識APIを活用した実用的なアプリケーションの開発

会議の議事録作成、講義のノート取りなどに利用できます。

## 機能

- リアルタイム音声認識（発話の終わりを検出して、発話ごとに1行ずつ出力）
- 認識結果のテキストファイル出力（追記。既存の内容は消しません）
- マイクデバイスの一覧表示と、ID・名前による選択
- 音声認識サービスの切り替え（Google / OpenAI / Groq）
- Ctrl+Cで終了したとき、認識中だった発話と、話している途中の発話も認識して書き出す（書き出し中にもう一度Ctrl+Cを押すと、残りは書き出さずに終了します）
- 音声認識に失敗した発話をWAVで残す（`--save-failed-audio` を指定したときのみ）
- ウェイクワード検出：認識結果に「アウラ」「あうら」「aura」が含まれる行は、接頭辞が `ウェイクワード` になります（それ以外の動作は変わりません）

## 処理の流れ

1. マイクから 16kHz・モノラルの音声を512サンプルずつ取り込む
2. [Silero VAD](https://github.com/snakers4/silero-vad) で発話かどうかを判定する（モデルは `silero-vad` パッケージに同梱され、起動時のダウンロードはありません）
3. 無音が1.0秒続くか、発話が `--max-segment-seconds`（既定30秒）に達したら、発話区間を切り出す
4. 切り出した音声を選択した音声認識サービスへ送り、テキストを受け取る
5. `[タイムスタンプ] 音声認識: テキスト` の形式で出力ファイルへ追記する

## 音声認識サービスについて

音声は、`-s` で選んだサービスへインターネット経由で送信されます。選んだサービスが使えない場合に別のサービスへ自動で切り替えることはありません。

| `-s` | サービス | 必要な設定 | 補足 |
|---|---|---|---|
| `google`（既定） | Google Speech Recognition | なし | [SpeechRecognition](https://github.com/Uberi/speech_recognition) ライブラリの既定キーを使う非公式の利用方法です。リクエスト数に制限があり、Google側の都合で使えなくなることがあります。音声はHTTPSで送信します。 |
| `google` | Google Cloud Speech-to-Text | `GOOGLE_CLOUD_CREDENTIALS_JSON` と `pip install -r requirements-google-cloud.txt` | サービスアカウントJSONのパスを設定したときに使われます（有料）。公式SDK（`google-cloud-speech`）のV1同期認識を使い、1回の認識は30秒で打ち切ります。同期認識は1回あたり約60秒までの音声しか受け付けないため、`--max-segment-seconds` は60未満にしてください。 |
| `openai` | OpenAI API | `OPENAI_API_KEY` | 既定モデルは `whisper-1`（有料）。`--model` で変更できます。 |
| `groq` | Groq API | `GROQ_API_KEY` | 既定モデルは `whisper-large-v3-turbo`。`--model` で変更できます。 |

認識言語は `--language` で指定します（例: `ja-JP`、`en-US`）。指定しない場合、`google` は日本語（`ja-JP`）、`openai` / `groq` はAPI側の自動判定です。

## 必要条件

- Python 3.10 以上（3.10 / 3.12 / 3.13 でテストを実行済み）
- OS
  - macOS 14 以降（Apple Silicon）：テストとマイク一覧表示を確認済み
  - Windows 10 / 11（64bit）：依存ライブラリがインストール可能な組み合わせであることのみ確認済みで、実機での動作は未確認です
  - Linux / Intel Mac：未確認です。LinuxではPortAudio本体（例: `sudo apt install libportaudio2`）が別途必要です
- マイクと、音声認識サービスへ接続できるネットワーク
- GPUは不要です（VADはCPUで動作します）

## インストール方法

1. リポジトリをクローンします
```bash
git clone https://github.com/clirea/VoiceScribe.git
cd VoiceScribe
```

2. 仮想環境を作成し、必要なライブラリをインストールします
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```
```bash
python -m pip install -r requirements.txt
```
PyTorchを含むため、初回は数百MBのダウンロードがあります。

Google Cloud Speech-to-Text（サービスアカウント）を使う場合だけ、追加でインストールします。
```bash
python -m pip install -r requirements-google-cloud.txt
```

3. OpenAI / Groq / Google Cloud を使う場合だけ、環境設定ファイルを準備します
```bash
# macOS / Linux
cp .env.example .env

# Windows
copy .env.example .env
```
`.env` を編集し、使うサービスの行の `#` を外して値を設定してください。既定の `google`（無料枠）だけを使う場合、`.env` は不要です。

## 使用方法

### 基本的な使い方
```bash
python main.py
```
`Ctrl+C` で終了します。認識結果は `output.txt` に追記されます。

macOSでは、初回実行時にターミナルへのマイクのアクセス許可を求められます。

### 出力ファイルを指定する
```bash
python main.py -o path/to/output.txt
```

### 利用可能なマイクデバイスを表示する
```bash
python main.py -l
```
`*` が付いているものがOSの既定の入力デバイスです。Windowsでは同じマイクがホストAPI（MME、WASAPIなど）ごとに複数表示されます。

### 特定のマイクデバイスを使用する
```bash
# デバイスIDで指定
python main.py -d 1

# デバイス名で指定（部分一致。複数一致した場合は一覧で最初のもの）
python main.py -n "マイク名"
```
指定したマイクが見つからない場合は、既定のマイクに切り替えずにエラーで終了します。

### 音声認識サービスを指定する
```bash
# Google Speech Recognition（デフォルト）
python main.py -s google

# OpenAI API
python main.py -s openai

# Groq API
python main.py -s groq
```

### オプション一覧

| オプション | 内容 | 既定値 |
|---|---|---|
| `-o`, `--output` | 出力ファイルのパス | `output.txt` |
| `-d`, `--device` | 使用するマイクデバイスのID | OSの既定マイク |
| `-n`, `--name` | 使用するマイクデバイスの名前（部分一致） | なし |
| `-l`, `--list` | マイクデバイスの一覧を表示して終了 | - |
| `-s`, `--service` | 音声認識サービス（`google` / `openai` / `groq`） | `google` |
| `--language` | 認識言語（例: `ja-JP`） | `google` は `ja-JP`、他は自動判定 |
| `--model` | `openai` / `groq` で使うモデル名 | `whisper-1` / `whisper-large-v3-turbo` |
| `--max-segment-seconds` | 無音が来なくても発話を区切る秒数（`0` で無制限。負数・`inf`・`nan` は設定エラー） | `30` |
| `--save-failed-audio DIR` | 音声認識に失敗した発話をWAVで保存するディレクトリ | 保存しない |

### 使用例
```bash
# 会議の議事録作成（OpenAI APIを使用し、失敗した発話は後で確認できるよう保存）
python main.py -s openai -o "meetings/board_meeting.txt" --save-failed-audio "meetings/failed"

# 講義の記録（特定のマイクを使用）
python main.py -n "Blue Yeti" -o "lectures/physics_101.txt"

# 英語の音声をGroqで認識
python main.py -s groq --language en-US -o "notes/english.txt"
```

### 出力形式
```
[20261012_101530] 音声認識: こんにちは、今日の会議を始めます
[20261012_101542] ウェイクワード: アウラ、議事録を取って
```
聞き取れなかった発話や、認識に失敗した発話は出力ファイルに書き込まれません（失敗はコンソールに表示されます）。

### 終了時の書き出し
`Ctrl+C` を押すと録音を止め、認識の途中だった発話、まだ認識に送っていない音声、話している途中の発話を順に認識して書き出してから終了します。認識の途中だった発話の再認識は1回だけです。書き出しがすべて終われば終了コードは `0` です。

- 書き出し中にもう一度 `Ctrl+C` を押した場合は、そこまでに認識できた分だけを保存し、書き出せなかった音声のおおよその秒数を表示して終了します。利用者による打ち切りなので終了コードは `0` です。
- 書き出し中に、出力ファイルへの保存失敗や、認証エラーなど継続できない認識失敗が起きた場合は、そこまでに保存できた分を残し、書き出せなかった音声のおおよその秒数とエラーを表示して、終了コード `1` で終了します。どちらの場合もマイクは閉じられます。

### マイクのテスト
```bash
python mic_check.py
```
マイクを選んで3秒間録音し、カレントディレクトリに `mic_test_日時.wav` を保存します。

### 終了コード

| コード | 意味 |
|---|---|
| `0` | 正常終了（`Ctrl+C` による終了を含む。終了時の書き出しを再度の `Ctrl+C` で打ち切った場合も `0`） |
| `1` | 実行中のエラー（マイクの切断、APIキーの認証失敗、出力ファイルへの保存失敗など。`Ctrl+C` 後の終了時の書き出し中に起きた場合を含む） |
| `2` | 設定の不備（APIキー未設定、マイクが見つからない、開始時に出力先へ書き込めない、引数の値が不正など） |

## トラブルシューティング

- **`設定エラー: -s openai には環境変数 OPENAI_API_KEY が必要です`**: `.env` の該当行の `#` を外し、実際のキーを設定してください。`your_xxx_here` のままの値は未設定として扱われます。
- **マイクが認識されない場合**: `python main.py -l` または `python mic_check.py` で入力デバイスを確認してください。macOSでは「システム設定 > プライバシーとセキュリティ > マイク」でターミナルを許可してください。
- **`マイク入力を開始できません`**: マイクが16kHzに対応していない可能性があります。Windowsでは `python main.py -l` に表示される別のホストAPI（MMEなど）の同じマイクをIDで指定すると動作する場合があります。
- **`音声を取りこぼしました`**: 音声認識の応答待ちが長く、溜められる音声（60秒）を超えました。ネットワークの状態を確認するか、応答の速いサービスを使用してください。
- **Google（無料枠）で認識に失敗し続ける場合**: リクエスト数の制限に達しているか、既定キーが使えなくなっている可能性があります。`openai` / `groq` か Google Cloud Speech-to-Text を使用してください。
- **Apple Silicon Macで Google の認識が `Bad CPU type` などで失敗する場合**: SpeechRecognitionに同梱のFLACエンコーダがIntel用のため、Rosetta 2 か `brew install flac` が必要です。
- **`エラー: 認識結果を保存できませんでした`**: 録音中に出力ファイルへ書けなくなりました（ディスクの空き不足、保存先の取り外しなど）。録音は停止し、保存できなかった認識結果はコンソールに表示されます。
- **小さな声が記録されない／雑音が記録される場合**: 発話かどうかはSilero VADだけで判定しています（振幅による足切りは既定では行いません）。
- **以前の `.env` で `GOOGLE_API_KEY` を設定していた場合**: サービスアカウントJSONの内容（JSON文字列）またはファイルパスとして、従来どおり Google Cloud Speech-to-Text に使われます（`requirements-google-cloud.txt` のインストールが必要です）。今後は `GOOGLE_CLOUD_CREDENTIALS_JSON` にJSONファイルのパスを設定する形を推奨します。無料枠を使う場合は `GOOGLE_API_KEY` の行を削除してください。

## プロジェクト構造

```
VoiceScribe/
├── main.py              # メインアプリケーション（引数の解釈と依存の組み立て）
├── mic_check.py         # マイクテスト用スクリプト
├── requirements.txt     # 実行に必要なライブラリ
├── requirements-google-cloud.txt # Google Cloud Speech-to-Text を使う場合の追加ライブラリ
├── requirements-dev.txt # テスト・lint用のライブラリ
├── pyproject.toml       # pytest / ruff の設定
├── .env.example         # 環境設定ファイルのサンプル
└── src/                 # ソースコード
    ├── application/     # アプリケーション層
    │   └── audio_controller.py  # 録音・VAD・音声認識の流れを制御
    ├── domain/          # ドメイン層
    │   ├── errors.py            # 例外の定義
    │   ├── speech_recognizer.py # 音声認識サービスのインターフェース
    │   ├── speech_segmenter.py  # 発話区間の切り出し
    │   ├── vad_service.py       # 音声活動検出サービス
    │   └── wakeword_detector.py # ウェイクワード検出
    ├── infrastructure/  # インフラストラクチャ層
    │   ├── audio_devices.py               # マイクの列挙・選択・入力ストリーム
    │   ├── recognition_service_factory.py # 設定から音声認識サービスを生成
    │   ├── speech_recognition_service.py  # Google音声認識サービス（無料枠）
    │   ├── google_cloud_recognition_service.py # Google Cloud Speech-to-Text（公式SDK）
    │   ├── openai_recognition_service.py  # OpenAI音声認識実装
    │   ├── groq_recognition_service.py    # Groq音声認識実装
    │   ├── whisper_api_support.py         # OpenAI / Groq 共通の言語・エラー変換
    │   ├── transcript_writer.py           # 認識結果のファイル出力
    │   └── wav_encoder.py                 # WAVへの変換
    └── tests/           # テストコード
```

## 検証方法

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest
```

テストはマイク・音声認識API・ネットワークを使いません（テスト中の外部接続は `src/tests/conftest.py` で遮断しています）。Google の2経路は、実際のライブラリのHTTP / gRPC送信だけを差し替えて、送信先・リクエストの内容・タイムアウトを確認しています。実際のサービスでの認識結果は確認していません。`.github/workflows/ci.yml` は同じ手順をWindows / macOSで実行します。

## 拡張方法

1. **新しい音声認識サービスの追加**:
   - `src/infrastructure` に `transcribe(audio_data, sample_rate) -> str` を持つクラスを追加（`src/domain/speech_recognizer.py` のインターフェース）
   - 失敗時は `RecognitionError` を送出し、聞き取れなかった場合は空文字を返す
   - `recognition_service_factory.py` にサービス名を登録

2. **ウェイクワードの変更**:
   - `AudioController` に `WakewordDetector(wakewords=[...])` を渡す

## ライセンス

[MIT License](LICENSE)

## 貢献方法

1. このリポジトリをフォークします
2. 機能ブランチを作成します (`git checkout -b feature/amazing-feature`)
3. 変更をコミットします (`git commit -m 'Add some amazing feature'`)
4. ブランチにプッシュします (`git push origin feature/amazing-feature`)
5. プルリクエストを作成します
