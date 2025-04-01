"""
Milvus vector database search functionality.
"""
from pymilvus import MilvusClient
import logging
from functools import lru_cache
from typing import List, Dict, Any

from utils.embedding import encode_query
from src.config import (
    MILVUS_DB_NAME, 
    RAG_TOP_K, 
    VECTOR_SEARCH_FIELDS, 
    VECTOR_SEARCH_WEIGHTS
)

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_milvus_client():
    """Get a cached Milvus client."""
    logger.info(f"Connecting to Milvus database: {MILVUS_DB_NAME}")
    return MilvusClient(MILVUS_DB_NAME)

def search_syllabus(query: str, top_k=RAG_TOP_K) -> List[Dict[str, Any]]:
    """
    Search the syllabus collection for relevant documents.
    
    Args:
        query: The search query
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
    
    # Optimized search approach - prioritize summary field first
    # This matches the approach in the notebook that showed better performance
    results = client.search(
        collection_name="sfc_syllabus_collection",
        anns_field="summary",  # Focus on summary field which has the index
        data=vector,
        limit=top_k,
        search_params={"metric_type": "COSINE"},
        output_fields=output_fields
    )
    
    # Process results
    processed_results = []
    for hit in results[0]:
        # Extract entity data to the top level for easier access
        if "entity" in hit:
            for key, value in hit["entity"].items():
                hit[key] = value
        
        hit["_weight"] = hit.get("distance", 0.0)
        hit["_matched_field"] = "summary"
        processed_results.append(hit)
    
    return processed_results
