"""
Embedding utilities for vector search.
"""
from sentence_transformers import SentenceTransformer
import logging
import hashlib
import pickle
import os
from pathlib import Path
from functools import lru_cache
from typing import List, Any, Optional
import numpy as np
from src.config import EMBEDDING_MODEL, EMBEDDING_DEVICE

logger = logging.getLogger(__name__)

# Cache directory for embeddings
CACHE_DIR = Path("./cache/embeddings")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class EmbeddingFunction:
    """Wrapper for SentenceTransformer to match expected interface."""
    
    def __init__(self, model_name: str, device: str):
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
    
    def encode_queries(self, queries: List[str]) -> List[List[float]]:
        """Encode queries."""
        embeddings = self.model.encode(queries)
        return embeddings.tolist()
    
    def encode_documents(self, documents: List[str]) -> List[List[float]]:
        """Encode documents.""" 
        embeddings = self.model.encode(documents)
        return embeddings.tolist()

@lru_cache(maxsize=1)
def get_embedding_fn() -> EmbeddingFunction:
    """
    Get the embedding function with caching.
    
    Returns:
        An embedding function wrapper
    """
    logger.info(f"Initializing embedding model {EMBEDDING_MODEL} on {EMBEDDING_DEVICE}")
    return EmbeddingFunction(EMBEDDING_MODEL, EMBEDDING_DEVICE)

def _get_cache_path(text: str) -> Path:
    """Get cache file path for a text."""
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    return CACHE_DIR / f"{text_hash}.pkl"

def _load_cached_embedding(text: str) -> Optional[List[List[float]]]:
    """Load embedding from cache if available."""
    cache_path = _get_cache_path(text)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached embedding: {e}")
            # Remove corrupted cache file
            cache_path.unlink(missing_ok=True)
    return None

def _save_cached_embedding(text: str, embedding: List[List[float]]) -> None:
    """Save embedding to cache."""
    cache_path = _get_cache_path(text)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)
    except Exception as e:
        logger.warning(f"Failed to save embedding to cache: {e}")

def encode_query(query: str) -> List[List[float]]:
    """
    Encode a single query string with caching.
    
    Args:
        query: The query text to encode
        
    Returns:
        The query embedding vector
    """
    # Try to load from cache first
    cached_embedding = _load_cached_embedding(query)
    if cached_embedding is not None:
        logger.debug(f"Using cached embedding for query: {query[:50]}...")
        return cached_embedding
    
    # Generate new embedding
    embedding_fn = get_embedding_fn()
    embedding = embedding_fn.encode_queries([query])
    
    # Save to cache
    _save_cached_embedding(query, embedding)
    logger.debug(f"Generated and cached new embedding for query: {query[:50]}...")
    
    return embedding

def encode_documents(documents: List[str]) -> List[List[float]]:
    """
    Encode a list of document strings.
    
    Args:
        documents: List of document texts to encode
        
    Returns:
        List of document embedding vectors
    """
    embedding_fn = get_embedding_fn()
    return embedding_fn.encode_documents(documents)
