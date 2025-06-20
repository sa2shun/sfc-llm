# SFC-LLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

**先進的なVision Language Model (VLM) 対応 SFC シラバス検索チャットボット**

Keio SFCの授業シラバスを検索し、LLMで自然言語回答するチャットアプリケーションです。  
最新のVLM技術により、画像とテキストを同時処理し、高精度な授業情報提供を実現します。

## ✨ 主な特徴

- 🤖 **複数のAIモデル対応**: LLM、VLM、vLLMによる高速推論
- 🖼️ **マルチモーダル処理**: テキスト + 画像の同時解析
- 🔍 **高精度検索**: Milvusベクトルデータベースによる意味的検索
- 🚀 **パフォーマンス最適化**: vLLMで10-20倍高速化
- 🐳 **Docker対応**: 開発から本番まで一貫した環境
- 🔐 **セキュリティ**: API認証とレート制限
- 📊 **監視機能**: ヘルスチェックとメトリクス

## 🏗️ アーキテクチャ

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Server │    │   AI Engines    │
│                 │    │              │    │                 │
│ • Web UI        │◄──►│ • FastAPI    │◄──►│ • LLM (Llama3)  │
│ • CLI Client    │    │ • Auth       │    │ • VLM (LLaVA)   │
│ • REST API      │    │ • Rate Limit │    │ • vLLM Engine   │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐    ┌─────────────────┐
                    │ Vector Database  │    │ Knowledge Base  │
                    │                  │    │                 │
                    │ • Milvus        │◄──►│ • SFC Syllabus  │
                    │ • Embeddings    │    │ • Course Data   │
                    │ • Semantic      │    │ • Metadata      │
                    │   Search        │    │                 │
                    └──────────────────┘    └─────────────────┘
```

## 📁 プロジェクト構成

```
sfc-llm/
├── 🚀 src/                          # コアアプリケーション
│   ├── chat_server.py               # 標準チャットサーバー
│   ├── vlm_chat_server.py          # VLM対応サーバー
│   ├── enhanced_chat_server.py     # 拡張機能サーバー
│   ├── config.py                   # 設定管理（型安全）
│   ├── milvus_search.py            # ベクトル検索エンジン
│   └── prompts.py                  # プロンプトテンプレート
├── 🔧 utils/                        # ユーティリティ
│   ├── llm.py                      # LLM推論エンジン
│   ├── vlm_engine.py               # VLMエンジン
│   ├── vllm_engine.py              # 高速推論エンジン
│   ├── embedding.py                # 埋め込み生成
│   ├── multimodal.py               # マルチモーダル処理
│   └── course_planner.py           # 履修プランニング
├── 📜 scripts/                      # 実行スクリプト
│   ├── start_server.py             # サーバー起動
│   ├── start_vlm_server.py         # VLMサーバー起動
│   └── init_syllabus_collection.py # DB初期化
├── 🧪 tests/                       # テストスイート
│   ├── test_config.py              # 設定テスト
│   ├── test_embedding.py           # 埋め込みテスト
│   ├── test_integration.py         # 統合テスト
│   └── run_tests.py                # テストランナー
├── 📚 docs/                        # ドキュメント
│   └── PERFORMANCE_GUIDE.md        # パフォーマンスガイド
├── 🐳 Docker関連
│   ├── Dockerfile                  # マルチステージビルド
│   └── docker-compose.yml          # オーケストレーション
└── 📊 データ・設定
    ├── csvs/sfc_syllabus.csv       # シラバスデータ
    ├── models/                     # AIモデル
    └── cache/                      # キャッシュ
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
export SFC_LLM_API_PORT="9001"
```

### 2. 依存パッケージのインストール（初回のみ）

```bash
# Poetry を使用する場合
poetry install

# または直接 pip を使用する場合
pip install -r requirements.txt
```

### 3. Milvus にコレクション作成（初回のみ）

研究会を除外したSFCシラバスデータベースを作成します：

```bash
python scripts/init_syllabus_collection.py
```

このスクリプトは以下の処理を行います：
- 元のCSVデータから研究会を含む授業を除外
- SFC学部・研究科の授業のみを対象
- 埋め込みベクトルを生成してMilvusに保存

### 4. FastAPI サーバー起動

```bash
# 便利な起動スクリプトを使用（推奨）
python scripts/start_server.py  # デフォルトはローカルモード (127.0.0.1)

# グローバルモードで起動（外部からアクセス可能）
python scripts/start_server.py --global

# 認証なしで起動
python scripts/start_server.py --no-auth

# グローバルモードかつ認証なしで起動
python scripts/start_server.py --global --no-auth

# または直接起動
python -m src.chat_server

# または uvicorn を使用
uvicorn src.chat_server:app --host 0.0.0.0 --port 9001
```

### 認証設定

サーバーはデフォルトでパスワード認証が有効になっています。以下の環境変数で設定を変更できます：

```bash
# 認証パスワードの設定（デフォルト: "kawallmshima"）
export SFC_LLM_API_PASSWORD="your_password_here"

# 認証の有効/無効切り替え（デフォルト: true）
export SFC_LLM_API_REQUIRE_AUTH="false"  # 認証を無効化
```

外部からAPIにアクセスする場合は、リクエストヘッダーに `X-API-Key` を設定する必要があります：

```bash
# curlの例
curl -H "X-API-Key: kawallmshima" \
     -H "Content-Type: application/json" \
     -d '{"user_input":"プログラミングの授業を教えてください"}' \
     http://your-server:9001/chat
```

### 5. 動作確認クライアントの実行

```bash
# デフォルトクエリを使用
python src/test_chat.py

# カスタムクエリを指定
python src/test_chat.py "プログラミングの授業を教えてください"

# 詳細情報を表示
python src/test_chat.py -v

# カスタムAPIキーを指定
python src/test_chat.py -k "your_api_key" "プログラミングの授業を教えてください"
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

- **研究会除外フィルタ**: データベース作成時に研究会を含む授業を自動的に除外
- **最適化された検索**: 授業概要フィールドに特化したインデックスによる高速検索
- **RAG判定**: 質問内容に応じて自動的にRAGが必要かどうかを判断
- **キャッシュ機能**: モデルや埋め込み関数のキャッシュによるパフォーマンス向上
- **エラーハンドリング**: 堅牢なエラー処理と詳細なログ出力
- **設定の柔軟性**: 環境変数による各種設定のカスタマイズ
- **パフォーマンステスト**: 検索性能を評価するためのテストスクリプト
- **API認証**: 外部公開時のセキュリティ確保のためのAPIキー認証
- **ローカル/グローバルモード**: 用途に応じたサーバー起動モードの切り替え

### 検索性能の最適化

検索性能を向上させるために以下の最適化を行っています：

1. **フィールド長の最適化**: 各フィールドの最大長を必要最小限に設定
2. **インデックス戦略**: 「授業概要」フィールドに特化したインデックスを作成
3. **検索アルゴリズム**: 最も関連性の高い情報を含む「授業概要」フィールドを優先的に検索

---

## 🔧 Docker を使用した起動

### Docker Compose での起動（推奨）

```bash
# 環境変数ファイルを作成
cp .env.example .env
# .envファイルでHUGGINGFACE_TOKENを設定

# 全サービスを起動（Milvus + SFC-LLM）
docker-compose up -d

# データベース初期化
docker-compose exec sfc-llm python scripts/init_syllabus_collection.py

# 動作確認
docker-compose exec sfc-llm python src/test_chat.py

# ログ確認
docker-compose logs sfc-llm

# サービス停止
docker-compose down

# 完全リセット（データも削除）
docker-compose down -v
```

### Docker の利点

- **簡単セットアップ**: Milvus データベースも含めて一括起動
- **環境分離**: ホスト環境への影響なし
- **再現性**: 異なる環境での一貫した動作
- **スケーラビリティ**: 容易な水平スケール

---

## ⚡ 性能最適化ガイド

### 現在の性能特性

- **レスポンス時間**: 5-10秒（通常のクエリ）
- **メモリ使用量**: 8-16GB（70Bモデル使用時）
- **同時処理**: 1-2リクエスト

### 推奨最適化手法

#### 1. LLMモデル最適化（最重要）

```python
# utils/llm.py での実装例
@lru_cache(maxsize=1)
def get_model() -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.float16,  # Half精度で高速化
        load_in_8bit=True,         # 8bit量子化でメモリ50%削減
        use_cache=True,            # KVキャッシュ有効化
        token=HF_TOKEN
    )
    
    # PyTorch 2.0 コンパイル最適化
    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode="reduce-overhead")
    
    return model
```

**期待効果**: メモリ使用量50%削減、推論速度20-30%向上

#### 2. 検索システム最適化

```python
# src/milvus_search.py での実装例
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def search_syllabus_cached(query_hash: str, top_k: int):
    """検索結果をキャッシュして高速化"""
    return _search_syllabus_internal(query_hash, top_k)

def search_syllabus(query: str, top_k=RAG_TOP_K):
    query_hash = hashlib.md5(query.encode()).hexdigest()
    return search_syllabus_cached(query_hash, top_k)
```

**期待効果**: 重複クエリのレスポンス時間90%削減

#### 3. 埋め込み生成キャッシュ

```python
# utils/embedding.py での実装例
class EmbeddingCache:
    def __init__(self, cache_dir="./cache/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}
    
    def get_embedding(self, text: str):
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        # キャッシュから読み込みまたは新規生成
```

**期待効果**: 埋め込み計算時間80%削減

#### 4. 非同期API処理

```python
# src/chat_server.py での実装例
from concurrent.futures import ThreadPoolExecutor
import asyncio

executor = ThreadPoolExecutor(max_workers=4)

@app.post("/chat")
async def chat(req: ChatRequest):
    loop = asyncio.get_event_loop()
    
    # CPU集約的な処理を別スレッドで実行
    result = await loop.run_in_executor(
        executor, 
        generate_response, 
        prompt
    )
    return result
```

**期待効果**: 同時処理能力3-5倍向上

### 性能監視

```bash
# メモリ使用量監視
docker stats sfc-llm

# レスポンス時間測定
python scripts/test_search_performance.py

# GPU使用率確認（GPU使用時）
nvidia-smi
```

### ハードウェア推奨仕様

#### 最小構成
- **CPU**: 4コア以上
- **メモリ**: 16GB以上
- **ストレージ**: 50GB以上

#### 推奨構成
- **CPU**: 8コア以上
- **メモリ**: 32GB以上
- **GPU**: NVIDIA GPU 12GB VRAM以上（RTX 3060 Ti/4070以上）
- **ストレージ**: SSD 100GB以上

#### 本格運用構成
- **CPU**: 16コア以上
- **メモリ**: 64GB以上
- **GPU**: NVIDIA A100/H100または複数GPU
- **ストレージ**: NVMe SSD 200GB以上

### 実装優先度

1. **高優先度**（即効性あり）
   - モデル量子化とGPUメモリ最適化
   - 検索結果キャッシュ
   - API非同期処理

2. **中優先度**
   - 埋め込みキャッシュ
   - バッチ処理対応
   - レスポンス圧縮

3. **低優先度**（長期的改善）
   - Redis外部キャッシュ
   - 高度なレート制限
   - 分散処理対応

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
