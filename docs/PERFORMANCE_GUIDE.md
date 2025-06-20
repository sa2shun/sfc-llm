# SFC-LLM パフォーマンス最適化ガイド

このガイドでは、SFC-LLMアプリケーションのパフォーマンスを最大化するための設定と最適化手法を説明します。

## 📊 現在のパフォーマンス特性

### ベースライン性能
- **レスポンス時間**: 5-15秒（通常のクエリ）
- **メモリ使用量**: 8-32GB（70Bモデル使用時）
- **同時処理**: 1-3リクエスト
- **スループット**: 10-20 tokens/sec

### ボトルネック分析
1. **LLM推論**: 全体の70-80%
2. **埋め込み生成**: 10-15%
3. **ベクトル検索**: 5-10%
4. **その他処理**: 5%

## 🚀 最適化手法

### 1. モデル最適化（最重要・効果大）

#### vLLM使用（推奨）
```bash
# vLLMを有効化
export SFC_LLM_USE_VLLM="true"
export SFC_LLM_VLLM_GPU_MEMORY="0.9"
export SFC_LLM_VLLM_TENSOR_PARALLEL="1"

# vLLMサーバー起動
python scripts/start_server.py --no-auth
```

**期待効果**: 推論速度10-20倍向上

#### 8bit量子化
```bash
# 8bit量子化を有効化（メモリ削減）
export SFC_LLM_ENABLE_8BIT="true"
```

**期待効果**: メモリ使用量50%削減、速度10-20%向上

#### PyTorch 2.0コンパイル
```bash
# モデルコンパイルを有効化
export SFC_LLM_ENABLE_MODEL_COMPILE="true"
```

**期待効果**: 推論速度15-30%向上

### 2. 埋め込み最適化

#### GPU使用
```bash
# 埋め込み生成にGPUを使用
export SFC_LLM_EMBEDDING_DEVICE="cuda"
```

**期待効果**: 埋め込み生成3-5倍高速化

#### キャッシュサイズ調整
```bash
# キャッシュサイズを増加
export SFC_LLM_EMBEDDING_CACHE_SIZE="500"
export SFC_LLM_SEARCH_CACHE_SIZE="200"
```

**期待効果**: キャッシュヒット率向上、レスポンス時間90%削減（キャッシュヒット時）

### 3. 検索最適化

#### Milvus設定最適化
```python
# src/milvus_search.py での最適化例
SEARCH_PARAMS = {
    "metric_type": "COSINE",
    "params": {
        "nprobe": 16,  # 検索精度vs速度のバランス
        "radius": 0.1,  # 類似度閾値
        "range_filter": 0.8
    }
}
```

#### インデックス最適化
```python
# 高速検索用インデックス設定
INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
```

### 4. サーバー最適化

#### 非同期処理
```bash
# ワーカー数を調整
export SFC_LLM_WORKERS="4"
export SFC_LLM_MAX_CONCURRENT="8"
```

#### リクエスト圧縮
```bash
# レスポンス圧縮を有効化
export SFC_LLM_ENABLE_COMPRESSION="true"
```

## 🔧 推奨設定組み合わせ

### 開発環境（CPU中心）
```bash
export SFC_LLM_EMBEDDING_DEVICE="cpu"
export SFC_LLM_EMBEDDING_CACHE_SIZE="100"
export SFC_LLM_MAX_NEW_TOKENS="256"
export SFC_LLM_ENABLE_MODEL_COMPILE="false"
```

### 本番環境（GPU利用）
```bash
export SFC_LLM_USE_VLLM="true"
export SFC_LLM_EMBEDDING_DEVICE="cuda"
export SFC_LLM_VLLM_GPU_MEMORY="0.85"
export SFC_LLM_EMBEDDING_CACHE_SIZE="1000"
export SFC_LLM_SEARCH_CACHE_SIZE="500"
export SFC_LLM_ENABLE_MODEL_COMPILE="true"
export SFC_LLM_MAX_NEW_TOKENS="512"
```

### 高負荷環境（複数GPU）
```bash
export SFC_LLM_USE_VLLM="true"
export SFC_LLM_VLLM_TENSOR_PARALLEL="2"
export SFC_LLM_VLLM_GPU_MEMORY="0.9"
export SFC_LLM_WORKERS="8"
export SFC_LLM_MAX_CONCURRENT="16"
```

## 📈 パフォーマンス監視

### ベンチマークスクリプト
```bash
# 検索性能テスト
python scripts/test_search_performance.py

# 統合性能テスト
python tests/run_tests.py --type integration

# カスタムベンチマーク
python scripts/benchmark.py --queries 100 --concurrent 4
```

### メトリクス監視
```bash
# システムリソース監視
htop
nvidia-smi  # GPU使用時

# アプリケーションメトリクス
curl http://localhost:9001/metrics

# ログ監視
tail -f server.log | grep "Response time"
```

## 🖥️ ハードウェア推奨仕様

### 最小構成（開発用）
- **CPU**: 4コア以上（Intel i5/AMD Ryzen 5以上）
- **メモリ**: 16GB以上
- **ストレージ**: SSD 50GB以上
- **GPU**: なし（CPU推論）

**期待性能**: 15-30秒/リクエスト

### 推奨構成（本番用）
- **CPU**: 8コア以上（Intel i7/AMD Ryzen 7以上）
- **メモリ**: 32GB以上
- **ストレージ**: NVMe SSD 100GB以上
- **GPU**: NVIDIA RTX 4070以上（12GB VRAM）

**期待性能**: 3-8秒/リクエスト

### 高性能構成（エンタープライズ）
- **CPU**: 16コア以上（Intel Xeon/AMD EPYC）
- **メモリ**: 64GB以上
- **ストレージ**: NVMe SSD RAID 200GB以上
- **GPU**: NVIDIA A100/H100または複数GPU

**期待性能**: 1-3秒/リクエスト、高同時処理

## 🎛️ 詳細チューニング

### vLLM設定
```python
# utils/vllm_engine.py での詳細設定
VLLM_CONFIG = {
    "model": HF_MODEL_ID,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.85,
    "max_num_batched_tokens": 4096,
    "max_num_seqs": 8,
    "block_size": 16,
    "swap_space": 4  # GB
}
```

### モデル量子化
```python
# utils/llm.py での量子化設定
QUANTIZATION_CONFIG = {
    "load_in_8bit": True,
    "llm_int8_threshold": 6.0,
    "llm_int8_has_fp16_weight": False
}
```

### キャッシュ戦略
```python
# utils/embedding.py での詳細キャッシュ設定
CACHE_CONFIG = {
    "embedding_cache_size": 1000,
    "cache_ttl": 3600,  # 1時間
    "persistent_cache": True,
    "cache_compression": True
}
```

## 🔍 トラブルシューティング

### よくある問題と解決方法

#### 1. メモリ不足エラー
```bash
# 解決方法
export SFC_LLM_MAX_NEW_TOKENS="256"  # トークン数削減
export SFC_LLM_ENABLE_8BIT="true"    # 量子化有効
export SFC_LLM_VLLM_GPU_MEMORY="0.7" # GPU使用量削減
```

#### 2. 推論速度が遅い
```bash
# 確認項目
nvidia-smi  # GPU使用率確認
htop        # CPU使用率確認

# 最適化
export SFC_LLM_USE_VLLM="true"
export SFC_LLM_ENABLE_MODEL_COMPILE="true"
```

#### 3. 同時接続エラー
```bash
# 解決方法
export SFC_LLM_MAX_CONCURRENT="4"   # 同時接続数削減
export SFC_LLM_WORKERS="2"          # ワーカー数調整
```

### パフォーマンス測定コマンド
```bash
# レスポンス時間測定
time python src/test_chat.py "プログラミングについて"

# 同時接続テスト
ab -n 10 -c 2 -p test_data.json -T application/json http://localhost:9001/chat

# メモリ使用量監視
python -c "
import psutil
import time
while True:
    print(f'Memory: {psutil.virtual_memory().percent}%')
    time.sleep(5)
"
```

## 📝 パフォーマンス最適化チェックリスト

### 必須項目
- [ ] HuggingFace Tokenの設定
- [ ] 適切なデバイス選択（CPU/GPU）
- [ ] メモリ使用量の監視
- [ ] 基本的なキャッシュ設定

### 推奨項目
- [ ] vLLM有効化
- [ ] 8bit量子化設定
- [ ] PyTorch 2.0コンパイル
- [ ] 埋め込みキャッシュ最適化
- [ ] 検索インデックス調整

### 高度な最適化
- [ ] 複数GPU設定
- [ ] カスタムモデル量子化
- [ ] Redis外部キャッシュ
- [ ] ロードバランサー設定
- [ ] モニタリングダッシュボード構築

## 🚀 期待される改善結果

| 最適化手法 | レスポンス時間改善 | メモリ削減 | スループット向上 |
|------------|-------------------|-----------|------------------|
| vLLM使用 | 10-20倍 | - | 10-20倍 |
| 8bit量子化 | 10-20% | 50% | 10-15% |
| GPU埋め込み | 3-5倍（埋め込み） | - | 20-30% |
| キャッシュ最適化 | 90%（ヒット時） | - | 2-3倍 |
| 全体最適化 | 20-50倍 | 50% | 30-100倍 |

正しく最適化されたシステムでは、開発環境でも3-8秒、本番環境では1-3秒でのレスポンスが期待できます。