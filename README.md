# SFC-LLM

Keio SFCの授業シラバスを検索し、LLMで自然言語回答するチャットアプリケーションです。  
Milvus によるベクトル検索（RAG）を使って関連情報を取得し、Meta Llama 3 モデルで応答を生成します。

---

## 🔧 プロジェクト構成

```
.
├── src/                     # メインアプリケーションコード
│   ├── __init__.py          # Pythonパッケージ化
│   ├── chat_server.py       # FastAPI アプリケーション本体
│   ├── config.py            # 設定ファイル
│   ├── milvus_search.py     # Milvus による検索処理
│   ├── prompts.py           # LLMプロンプトテンプレート
│   ├── test_chat.py         # 動作確認用クライアントスクリプト
│   └── test_collections.py  # コレクション存在確認
├── utils/                   # ユーティリティ関数
│   ├── __init__.py          # Pythonパッケージ化
│   ├── embedding.py         # 埋め込みベクトル生成ユーティリティ
│   └── llm.py               # LLM関連ユーティリティ
├── scripts/                 # 実行スクリプト
│   ├── __init__.py          # Pythonパッケージ化
│   ├── init_syllabus_collection.py  # Milvus コレクション作成＋データ投入
│   ├── test_search_performance.py   # 検索性能評価スクリプト
│   └── start_server.py              # サーバー起動補助スクリプト
├── csvs/                    # データファイル
│   └── sfc_syllabus.csv     # 授業データ
└── models/                  # モデルファイル
    └── mixtral/             # LLMモデルファイル
```

---

## 🚀 起動手順

### 1. 環境変数の設定（必要に応じて）

```bash
# HuggingFace APIトークン（必須）
export HUGGINGFACE_TOKEN="your_token_here"

# モデルディレクトリのカスタマイズ（任意）
export SFC_LLM_MODEL_DIR="/path/to/model/directory"

# 埋め込みモデルのデバイス設定（任意、デフォルトはCPU）
export SFC_LLM_EMBEDDING_DEVICE="cuda:0"

# APIサーバーのホスト・ポート設定（任意）
export SFC_LLM_API_HOST="0.0.0.0"
export SFC_LLM_API_PORT="8001"
```

### 2. 依存パッケージのインストール（初回のみ）

```bash
# Poetry を使用する場合
poetry install

# または直接 pip を使用する場合
pip install -r requirements.txt
```

### 3. Milvus にコレクション作成（初回のみ）

```bash
python scripts/init_syllabus_collection.py
```

### 4. FastAPI サーバー起動

```bash
# 便利な起動スクリプトを使用（推奨）
python scripts/start_server.py

# または直接起動
python -m src.chat_server

# または uvicorn を使用
uvicorn src.chat_server:app --host 0.0.0.0 --port 8001
```

### 5. 動作確認クライアントの実行

```bash
# デフォルトクエリを使用
python src/test_chat.py

# カスタムクエリを指定
python src/test_chat.py "プログラミングの授業を教えてください"

# 詳細情報を表示
python src/test_chat.py -v
```

### 6. 検索性能のテスト

```bash
# 検索性能テストスクリプトを実行
python scripts/test_search_performance.py
```

### 注意事項: Pythonモジュールのインポートについて

スクリプトを直接実行する際に「ModuleNotFoundError: No module named 'src'」などのエラーが発生する場合は、以下のいずれかの方法で解決できます：

1. **モジュールとして実行する**:
   ```bash
   python -m src.test_chat "プログラミングの授業を教えてください"
   ```

2. **PYTHONPATH環境変数を設定する**:
   ```bash
   PYTHONPATH=. python src/test_chat.py "プログラミングの授業を教えてください"
   ```

3. **既に修正済みのスクリプトを使用する**:
   各スクリプトファイルには、Pythonパスを自動的に設定するコードが追加されています。

---

## 📦 使用技術

- **LLM**: Meta Llama 3 70B Instruct（HuggingFace）
- **検索**: Milvus（ローカルDBモード）
- **埋め込み**: SentenceTransformer (`all-MiniLM-L6-v2`)
- **APIサーバー**: FastAPI

---

## 🔍 主な機能

- **最適化された検索**: 授業概要フィールドに特化したインデックスによる高速検索
- **RAG判定**: 質問内容に応じて自動的にRAGが必要かどうかを判断
- **キャッシュ機能**: モデルや埋め込み関数のキャッシュによるパフォーマンス向上
- **エラーハンドリング**: 堅牢なエラー処理と詳細なログ出力
- **設定の柔軟性**: 環境変数による各種設定のカスタマイズ
- **パフォーマンステスト**: 検索性能を評価するためのテストスクリプト

### 検索性能の最適化

検索性能を向上させるために以下の最適化を行っています：

1. **フィールド長の最適化**: 各フィールドの最大長を必要最小限に設定
2. **インデックス戦略**: 「授業概要」フィールドに特化したインデックスを作成
3. **検索アルゴリズム**: 最も関連性の高い情報を含む「授業概要」フィールドを優先的に検索

---

## 🤝 コントリビューション

このプロジェクトへの貢献を歓迎します。以下の手順で開発に参加できます：

1. リポジトリをフォークする
2. 機能追加やバグ修正のためのブランチを作成する (`git checkout -b feature/amazing-feature`)
3. 変更をコミットする (`git commit -m 'Add some amazing feature'`)
4. ブランチをプッシュする (`git push origin feature/amazing-feature`)
5. プルリクエストを作成する

### 開発ガイドライン

- コードは[PEP 8](https://pep8.org/)スタイルガイドに従ってください
- 新しい機能には適切なテストを追加してください
- ドキュメントを更新してください
- コミットメッセージは明確で説明的にしてください

---

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。
