"""
Configuration settings for the SFC-LLM application.

This module centralizes all configuration parameters with environment variable support
and type safety. All settings are documented and have sensible defaults.
"""
import os
from pathlib import Path
from typing import List, Tuple

# =============================================================================
# Project Structure
# =============================================================================

PROJECT_ROOT: Path = Path(__file__).parent.parent.absolute()
DATA_DIR: Path = PROJECT_ROOT / "csvs"
MODELS_DIR: Path = PROJECT_ROOT / "models"
CACHE_DIR: Path = PROJECT_ROOT / "cache"

# =============================================================================
# Database Configuration
# =============================================================================

MILVUS_DB_NAME: str = "sfc_syllabus.db"
MILVUS_COLLECTION_NAME: str = "sfc_syllabus_collection"

# =============================================================================
# Large Language Model Configuration
# =============================================================================

# Primary LLM model configuration
HF_MODEL_ID: str = "meta-llama/Meta-Llama-3-70B-Instruct"
LOCAL_MODEL_DIR: str = os.environ.get(
    "SFC_LLM_MODEL_DIR", 
    f"/raid/{os.environ.get('USER', 'default')}/meta-llama_Llama-3.1-70B-Instruct"
)
HF_TOKEN: str = os.environ.get("HUGGINGFACE_TOKEN", "")

# Embedding model configuration
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE: str = os.environ.get("SFC_LLM_EMBEDDING_DEVICE", "cpu")

# VLM (Vision Language Model) configuration
VLM_MODEL: str = os.environ.get("SFC_LLM_VLM_MODEL", "llava-hf/llava-v1.6-mistral-7b-hf")

# =============================================================================
# API Server Configuration
# =============================================================================

API_HOST: str = os.environ.get("SFC_LLM_API_HOST", "0.0.0.0")
API_PORT: int = int(os.environ.get("SFC_LLM_API_PORT", "9001"))
API_PASSWORD: str = os.environ.get("SFC_LLM_API_PASSWORD", "kawallmshima")
API_REQUIRE_AUTH: bool = os.environ.get("SFC_LLM_API_REQUIRE_AUTH", "true").lower() == "true"

# Valid ports for the application
VALID_PORTS: List[int] = [8001, 9001, 9002, 9003]

# =============================================================================
# RAG (Retrieval-Augmented Generation) Settings
# =============================================================================

RAG_TOP_K: int = 5
VECTOR_SEARCH_FIELDS: List[str] = ["summary", "goals", "schedule"]
VECTOR_SEARCH_WEIGHTS: List[float] = [0.6, 0.3, 0.1]  # Weights for each field

# =============================================================================
# LLM Generation Parameters
# =============================================================================

MAX_NEW_TOKENS: int = int(os.environ.get("SFC_LLM_MAX_NEW_TOKENS", "512"))
TEMPERATURE: float = float(os.environ.get("SFC_LLM_TEMPERATURE", "0.7"))
TOP_P: float = float(os.environ.get("SFC_LLM_TOP_P", "0.9"))

# =============================================================================
# Performance Optimization Settings
# =============================================================================

# General performance settings
ENABLE_MODEL_COMPILE: bool = os.environ.get("SFC_LLM_ENABLE_MODEL_COMPILE", "true").lower() == "true"
EMBEDDING_CACHE_SIZE: int = int(os.environ.get("SFC_LLM_EMBEDDING_CACHE_SIZE", "100"))
SEARCH_CACHE_SIZE: int = int(os.environ.get("SFC_LLM_SEARCH_CACHE_SIZE", "100"))

# vLLM-specific settings
USE_VLLM: bool = os.environ.get("SFC_LLM_USE_VLLM", "false").lower() == "true"
VLLM_GPU_MEMORY_UTILIZATION: float = float(os.environ.get("SFC_LLM_VLLM_GPU_MEMORY", "0.85"))
VLLM_TENSOR_PARALLEL_SIZE: int = int(os.environ.get("SFC_LLM_VLLM_TENSOR_PARALLEL", "1"))

# =============================================================================
# Logging Configuration
# =============================================================================

LOG_LEVEL: str = os.environ.get("SFC_LLM_LOG_LEVEL", "INFO").upper()
LOG_FORMAT: str = os.environ.get(
    "SFC_LLM_LOG_FORMAT", 
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# =============================================================================
# Validation Functions
# =============================================================================

def validate_config() -> bool:
    """
    Validate configuration settings.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    # Validate required environment variables
    if not HF_TOKEN:
        print("Warning: HUGGINGFACE_TOKEN not set")
        return False
    
    # Validate API port
    if API_PORT not in VALID_PORTS:
        print(f"Warning: API_PORT {API_PORT} not in valid ports {VALID_PORTS}")
    
    # Validate directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    return True

def get_config_summary() -> dict:
    """
    Get a summary of current configuration.
    
    Returns:
        dict: Configuration summary
    """
    return {
        "model": {
            "llm_model": HF_MODEL_ID,
            "embedding_model": EMBEDDING_MODEL,
            "vlm_model": VLM_MODEL,
            "device": EMBEDDING_DEVICE
        },
        "api": {
            "host": API_HOST,
            "port": API_PORT,
            "auth_enabled": API_REQUIRE_AUTH
        },
        "performance": {
            "use_vllm": USE_VLLM,
            "model_compile": ENABLE_MODEL_COMPILE,
            "cache_sizes": {
                "embedding": EMBEDDING_CACHE_SIZE,
                "search": SEARCH_CACHE_SIZE
            }
        }
    }
