# VoiceScribe

音声認識を使用してリアルタイムで音声をテキストに変換するPythonアプリケーションです。ウェイクワード検出と音声認識機能を備えています。

## 機能

- リアルタイム音声認識
- ウェイクワード検出
- 認識結果のテキストファイル出力
- 複数のマイクデバイスのサポート

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
    ├── domain/          # ドメイン層
    ├── infrastructure/  # インフラストラクチャ層
    └── tests/           # テストコード
```

## ライセンス

[MIT License](LICENSE)

## 貢献方法

1. このリポジトリをフォークします
2. 機能ブランチを作成します (`git checkout -b feature/amazing-feature`)
3. 変更をコミットします (`git commit -m 'Add some amazing feature'`)
4. ブランチにプッシュします (`git push origin feature/amazing-feature`)
5. プルリクエストを作成します 