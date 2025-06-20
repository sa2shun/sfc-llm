"""
Milvus vector database search functionality.
"""
from pymilvus import MilvusClient
import logging
import hashlib
from functools import lru_cache
from typing import List, Dict, Any, Optional

from utils.embedding import encode_query
from src.config import (
    MILVUS_DB_NAME, 
    RAG_TOP_K, 
    VECTOR_SEARCH_FIELDS, 
    VECTOR_SEARCH_WEIGHTS,
    SEARCH_CACHE_SIZE
)

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_milvus_client() -> MilvusClient:
    """Get a cached Milvus client."""
    logger.info(f"Connecting to Milvus database: {MILVUS_DB_NAME}")
    return MilvusClient(MILVUS_DB_NAME)


def search_syllabus(query: str, top_k: int = RAG_TOP_K) -> List[Dict[str, Any]]:
    """
    Search the syllabus collection for relevant documents with caching.
    
    Args:
        query: The search query
        top_k: Number of results to return
        
    Returns:
        List of matching documents with their metadata
    """
    # Create hash of query for caching
    query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
    
    # Try to get from cache first
    try:
        return _search_syllabus_internal(query_hash, query, top_k)
    except Exception as e:
        logger.error(f"Search error for query '{query[:50]}...': {str(e)}")
        return []

@lru_cache(maxsize=SEARCH_CACHE_SIZE)
def _search_syllabus_internal(query_hash: str, query: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Internal search function with caching.
    
    Args:
        query_hash: MD5 hash of the query (for cache key)
        query: The actual search query
        top_k: Number of results to return
        
    Returns:
        List of matching documents with their metadata
    """
    client = get_milvus_client()
    vector = encode_query(query)
    
    # Output fields to retrieve
    output_fields = [
        "subject_name", 
        "faculty",
        "category", 
        "credits",
        "year",
        "semester",
        "delivery_mode",
        "language",
        "english_support",
        "selection",
        "giga",
        "url"
    ]
    
    # Optimized search with better parameters
    results = client.search(
        collection_name="sfc_syllabus_collection",
        anns_field="summary",  # Focus on summary field for best performance
        data=vector,
        limit=top_k,
        search_params={
            "metric_type": "COSINE",
            "params": {"nprobe": 10}  # Better accuracy vs speed tradeoff
        },
        output_fields=output_fields
    )
    
    # Process results more efficiently
    processed_results = []
    for hit in results[0]:
        # Extract entity data to the top level for easier access
        if "entity" in hit:
            hit.update(hit["entity"])
        
        hit["_weight"] = hit.get("distance", 0.0)
        hit["_matched_field"] = "summary"
        processed_results.append(hit)
    
    logger.info(f"Found {len(processed_results)} results for query hash {query_hash[:8]}...")
    return processed_results
