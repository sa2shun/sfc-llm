# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Japanese language RAG (Retrieval-Augmented Generation) application** for **Keio SFC (Shonan Fujisawa Campus) course syllabus search**. The system allows users to ask questions about SFC courses in natural Japanese and receive intelligent responses based on the actual syllabus data.

### What This Application Does

- **Syllabus Search**: Users can search for SFC courses using natural language queries in Japanese
- **Intelligent Q&A**: Ask questions like "プログラミングの授業を教えてください" (Tell me about programming classes)
- **RAG-powered Responses**: Combines vector search with LLM generation for accurate, contextual answers
- **Course Information**: Provides details about course content, objectives, schedules, and requirements
- **Smart Routing**: Automatically determines when to use syllabus data vs. general knowledge

### Target Use Cases

- **Students**: Finding relevant courses, understanding course content and requirements
- **Faculty**: Course discovery and academic planning assistance  
- **Academic Advisors**: Helping students with course selection
- **Researchers**: Analyzing course offerings and curriculum structure

## 🏗️ プロジェクト概要

### アーキテクチャ
- **コアシステム**: FastAPI + Milvus + LLM (Meta Llama 3)
- **VLM対応**: LLaVA, Qwen2-VL による画像解析
- **高速化**: vLLM による推論速度10-20倍向上
- **データベース**: ベクトル検索による意味的類似性マッチング
- **キャッシュ**: 埋め込み・検索結果の多層キャッシュ
- **セキュリティ**: API認証、非rootユーザー実行

### 主要コンポーネント
- `src/chat_server.py`: 標準チャットサーバー
- `src/vlm_chat_server.py`: VLM対応マルチモーダルサーバー  
- `src/enhanced_chat_server.py`: 履修プランニング機能付きサーバー
- `utils/vlm_engine.py`: Vision Language Model エンジン
- `utils/vllm_engine.py`: 高速推論エンジン
- `tests/`: 統合テストスイート

## Essential Commands

### Server Management
```bash
# Start server (local mode, default port 9001)
python scripts/start_server.py

# Start server with external access
python scripts/start_server.py --global

# Start server without authentication
python scripts/start_server.py --no-auth

# Start VLM-enhanced server (Vision Language Model)
python scripts/start_vlm_server.py
python scripts/start_vlm_server.py --vlm-model llava-mistral-7b --no-auth

# Start enhanced server with advanced features
python -m src.enhanced_chat_server
uvicorn src.enhanced_chat_server:app --host 0.0.0.0 --port 9001

# Start VLM server directly
python -m src.vlm_chat_server
uvicorn src.vlm_chat_server:app --host 0.0.0.0 --port 9001

# Start standard server
python -m src.chat_server
uvicorn src.chat_server:app --host 0.0.0.0 --port 9001

# Poetry-managed execution
poetry run python scripts/start_server.py
poetry run python scripts/start_vlm_server.py
```

### Database Initialization
```bash
# Initialize Milvus collection with syllabus data (required on first run)
python scripts/init_syllabus_collection.py
```

### Testing and Validation
```bash
# Test chat functionality with SFC syllabus queries
python src/test_chat.py
python src/test_chat.py "プログラミングの授業を教えてください"
python src/test_chat.py "データサイエンスに関する科目はありますか"
python src/test_chat.py "英語で行われる授業を探しています"

# Test VLM functionality (Vision Language Model)
python src/test_vlm_chat.py "このシラバス画像について教えて" -i syllabus.jpg
python src/test_vlm_chat.py --list-models
python src/test_vlm_chat.py "画像の授業内容を分析して" -i course_image.png --upload

# Test enhanced features
python src/test_chat.py "データサイエンスの履修プランを教えてください"
python src/test_chat.py "プログラミング初心者におすすめの科目は？"

# Test search performance
python scripts/test_search_performance.py

# Test collections
python src/test_collections.py

# Comprehensive test suite
python tests/run_tests.py --type all
python tests/run_tests.py --type unit
python tests/run_tests.py --type integration
```

### Dependency Management
```bash
# Primary method (Poetry - recommended)
poetry install                  # Install all dependencies from pyproject.toml
poetry install --no-dev        # Install production dependencies only
poetry shell                   # Activate virtual environment

# Alternative method (pip)
pip install -r requirements.txt

# Environment setup (installs Poetry if needed)
./setup.sh

# Poetry environment management
poetry add <package>            # Add new dependency
poetry remove <package>         # Remove dependency
poetry show                     # List installed packages
poetry env info                # Show environment information
```

### Development Tools
```bash
# Jupyter notebook for development and analysis
jupyter lab                     # Start JupyterLab
jupyter notebook               # Start classic Jupyter Notebook

# Performance monitoring
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

### Docker Commands
```bash
# Build and start all services (recommended)
docker-compose up -d

# Build and start with logs
docker-compose up --build

# Stop all services
docker-compose down

# Stop and remove volumes (clean reset)
docker-compose down -v

# View logs
docker-compose logs sfc-llm
docker-compose logs milvus

# Initialize database in Docker
docker-compose exec sfc-llm python scripts/init_syllabus_collection.py

# Test chat in Docker
docker-compose exec sfc-llm python src/test_chat.py

# Individual service management
docker-compose up -d milvus      # Start only Milvus
docker-compose up -d sfc-llm     # Start only the app

# Clean rebuild
docker-compose down --volumes --remove-orphans
docker-compose build --no-cache
```

## Architecture Overview

This is a Japanese language RAG (Retrieval-Augmented Generation) system for **Keio SFC (Shonan Fujisawa Campus) course syllabus search** using:

- **FastAPI server** (`src/chat_server.py`) - Main API endpoint with authentication
- **Milvus vector database** - Local file-based storage for course embeddings
- **Meta Llama 3 70B** - Text generation model via HuggingFace
- **SentenceTransformer** - Embedding generation for semantic search

### Core Components

1. **Search System** (`src/milvus_search.py`) - Optimized vector search focused on course summary field
2. **LLM Interface** (`utils/llm.py`) - Model loading and response generation with caching
3. **Embedding Service** (`utils/embedding.py`) - Text vectorization for search queries
4. **RAG Decision Logic** (`src/prompts.py`) - Determines when to use retrieval vs direct response

### Key Design Patterns

- **SFC-specific optimization**: Designed specifically for Japanese academic content and SFC course structure
- **Field-optimized search**: Prioritizes course summary field (60% weight) over goals (30%) and schedule (10%) for optimal SFC syllabus search
- **Smart RAG routing**: Automatically decides whether to use SFC syllabus data vs. general knowledge based on query content
- **Japanese language support**: Optimized for natural Japanese queries about academic courses
- **Model caching**: LLM and embedding models are cached for performance
- **Flexible authentication**: Can run with/without API key validation

## Environment Variables

Required:
- `HUGGINGFACE_TOKEN` - For model access

Optional configuration:
- `SFC_LLM_MODEL_DIR` - Custom model directory path
- `SFC_LLM_EMBEDDING_DEVICE` - GPU/CPU selection for embeddings (`cuda` or `cpu`)
- `SFC_LLM_API_HOST`/`SFC_LLM_API_PORT` - Server binding (default: `0.0.0.0:9001`)
- `SFC_LLM_API_PASSWORD` - Authentication key (default: `kawallmshima`)
- `SFC_LLM_API_REQUIRE_AUTH` - Enable/disable authentication (default: `true`)

### Environment Setup Example
```bash
# Required
export HUGGINGFACE_TOKEN="your_token_here"

# Optional customizations
export SFC_LLM_MODEL_DIR="/custom/model/path"
export SFC_LLM_EMBEDDING_DEVICE="cuda"      # Use GPU for embeddings
export SFC_LLM_API_HOST="127.0.0.1"        # Local access only
export SFC_LLM_API_PORT="8001"             # Custom port
export SFC_LLM_API_PASSWORD="custom_pass"   # Custom password
export SFC_LLM_API_REQUIRE_AUTH="false"    # Disable authentication

# Performance optimization settings
export SFC_LLM_MAX_NEW_TOKENS="512"        # Maximum tokens to generate
export SFC_LLM_TEMPERATURE="0.7"           # Sampling temperature
export SFC_LLM_TOP_P="0.9"                 # Nucleus sampling parameter
export SFC_LLM_ENABLE_MODEL_COMPILE="true" # Enable PyTorch 2.0 compile
export SFC_LLM_EMBEDDING_CACHE_SIZE="100"  # Embedding cache size
export SFC_LLM_SEARCH_CACHE_SIZE="100"     # Search result cache size

# Logging settings
export SFC_LLM_LOG_LEVEL="INFO"            # Log level (DEBUG, INFO, WARNING, ERROR)

# vLLM settings (for ultra-fast inference)
export SFC_LLM_USE_VLLM="true"             # Enable vLLM engine
export SFC_LLM_VLLM_GPU_MEMORY="0.85"      # GPU memory utilization
export SFC_LLM_VLLM_TENSOR_PARALLEL="1"    # Number of GPUs for tensor parallelism
```

## Development Notes

### Package Management
- **Primary**: Poetry (pyproject.toml) for dependency management
- **Fallback**: pip (requirements.txt) for compatibility
- Use `poetry shell` to activate the virtual environment
- Use `./setup.sh` for automated environment setup

### Module Import Resolution
- All Python modules support direct execution with automatic path resolution
- When running scripts directly, they automatically add project root to Python path
- Can also use module execution: `python -m src.chat_server`

### File Structure
- **Data**: `csvs/sfc_syllabus.csv` contains **Keio SFC course syllabus data** with course summaries, objectives, and schedules
- **Models**: `models/mixtral/` directory for LLM model files  
- **Scripts**: `scripts/` contains utilities for setup and testing
- **Application**: `src/` contains main application code
- **Utilities**: `utils/` contains reusable utility functions

### Key Configuration
- Main config in `src/config.py` with environment variable overrides
- Default port changed from 8001 to 9001 to avoid conflicts
- Milvus uses local file-based storage (not client-server mode)
- Authentication enabled by default with configurable password