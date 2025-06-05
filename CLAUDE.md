# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Server Management
```bash
# Start server (local mode, default port 9001)
python scripts/start_server.py

# Start server with external access
python scripts/start_server.py --global

# Start server without authentication
python scripts/start_server.py --no-auth

# Direct server startup
python -m src.chat_server
uvicorn src.chat_server:app --host 0.0.0.0 --port 9001
```

### Database Initialization
```bash
# Initialize Milvus collection with syllabus data (required on first run)
python scripts/init_syllabus_collection.py
```

### Testing and Validation
```bash
# Test chat functionality
python src/test_chat.py
python src/test_chat.py "プログラミングの授業を教えてください"

# Test search performance
python scripts/test_search_performance.py

# Test collections
python src/test_collections.py
```

### Dependency Management
```bash
# Install dependencies
pip install -r requirements.txt
poetry install  # if using Poetry

# Environment setup
./setup.sh
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
```

## Architecture Overview

This is a Japanese language RAG (Retrieval-Augmented Generation) system for Keio SFC course syllabus search using:

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

- **Field-optimized search**: Prioritizes course summary field (60% weight) over goals (30%) and schedule (10%)
- **Smart RAG routing**: Automatically decides whether to use retrieval based on query content
- **Model caching**: LLM and embedding models are cached for performance
- **Flexible authentication**: Can run with/without API key validation

## Environment Variables

Required:
- `HUGGINGFACE_TOKEN` - For model access

Optional configuration:
- `SFC_LLM_MODEL_DIR` - Custom model directory path
- `SFC_LLM_EMBEDDING_DEVICE` - GPU/CPU selection for embeddings
- `SFC_LLM_API_HOST`/`SFC_LLM_API_PORT` - Server binding
- `SFC_LLM_API_PASSWORD` - Authentication key
- `SFC_LLM_API_REQUIRE_AUTH` - Enable/disable authentication

## File Structure Notes

- Data files in `csvs/sfc_syllabus.csv`
- Model files in `models/mixtral/` directory
- All Python modules support direct execution with automatic path resolution
- Scripts in `scripts/` are utilities for setup and testing
- Main application code in `src/` with utilities in `utils/`