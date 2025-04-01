"""
Embedding utilities for vector search.
"""
from pymilvus import model
import logging
from functools import lru_cache
from typing import List, Any
from src.config import EMBEDDING_MODEL, EMBEDDING_DEVICE

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_embedding_fn() -> model.dense.SentenceTransformerEmbeddingFunction:
    """
    Get the embedding function with caching.
    
    Returns:
        A SentenceTransformer embedding function
    """
    logger.info(f"Initializing embedding model {EMBEDDING_MODEL} on {EMBEDDING_DEVICE}")
    return model.dense.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device=EMBEDDING_DEVICE
    )

def encode_query(query: str) -> List[List[float]]:
    """
    Encode a single query string.
    
    Args:
        query: The query text to encode
        
    Returns:
        The query embedding vector
    """
    embedding_fn = get_embedding_fn()
    return embedding_fn.encode_queries([query])

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
