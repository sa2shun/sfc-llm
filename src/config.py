"""
Configuration settings for the SFC-LLM application.
"""
import os
from pathlib import Path
import getpass

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "csvs"
MODELS_DIR = PROJECT_ROOT / "models"

# Database settings
MILVUS_DB_NAME = "sfc_syllabus.db"

# LLM Model settings
HF_MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
LOCAL_MODEL_DIR = os.environ.get(
    "SFC_LLM_MODEL_DIR", 
    f"/raid/{os.environ.get('USER', 'default')}/meta-llama_Llama-3.1-70B-Instruct"
)
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# Embedding model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = os.environ.get("SFC_LLM_EMBEDDING_DEVICE", "cpu")

# API settings
API_HOST = os.environ.get("SFC_LLM_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("SFC_LLM_API_PORT", "8001"))
API_PASSWORD = os.environ.get("SFC_LLM_API_PASSWORD", "kawallmshima")
API_REQUIRE_AUTH = os.environ.get("SFC_LLM_API_REQUIRE_AUTH", "true").lower() == "true"

# RAG settings
RAG_TOP_K = 5
VECTOR_SEARCH_FIELDS = ["summary", "goals", "schedule"]
VECTOR_SEARCH_WEIGHTS = [0.6, 0.3, 0.1]  # Weights for each field

# LLM generation settings
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
