# sfc-llm

Keio SFCの授業シラバスを検索し、LLMで自然言語回答するチャットアプリケーションです。  
Milvus によるベクトル検索（RAG）を使って関連情報を取得し、Meta Llama 3 モデルで応答を生成します。

---

## 🔧 構成

```
.
├── src/
│   ├── chat_server.py       # FastAPI アプリケーション本体
│   ├── test_chat.py         # 動作確認用クライアントスクリプト
│   ├── milvus_search.py     # Milvus による検索処理
│   ├── test_collections.py  # コレクション存在確認
├── scripts/
│   └── init_syllabus_collection.py  # Milvus コレクション作成＋データ投入
├── csvs/
│   └── sfc_syllabus.csv     # 授業データ
```

---

## 🚀 起動手順

### 1. 依存パッケージのインストール（初回のみ）

```bash
poetry install
```

### 2. Milvus にコレクション作成（初回のみ）

```bash
poetry run python scripts/init_syllabus_collection.py
```

### 3. FastAPI サーバー起動

```bash
poetry run uvicorn src.chat_server:app --host 0.0.0.0 --port 8001
```

### 4. 動作確認クライアントの実行

```bash
poetry run python src/test_chat.py
```

---

## 📦 使用技術

- **LLM**: Meta Llama 3 70B Instruct（HuggingFace）
- **検索**: Milvus（ローカルDBモード）
- **執算化**: SentenceTransformer (`all-MiniLM-L6-v2`)
- **APIサーバー**: FastAPI

