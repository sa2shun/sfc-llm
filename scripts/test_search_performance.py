#!/usr/bin/env python
"""
Test script to evaluate the performance of the Milvus search.
"""
import os
import sys
import time
import logging

# Add the project root to the Python path when run directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.milvus_search import search_syllabus
from utils.embedding import get_embedding_fn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_search_performance(queries):
    """
    Test the search performance with a list of queries.
    
    Args:
        queries: List of query strings to test
    """
    logger.info("Starting search performance test")
    
    for query in queries:
        logger.info(f"Testing query: '{query}'")
        
        # Measure search time
        start_time = time.time()
        results = search_syllabus(query)
        elapsed_time = time.time() - start_time
        
        logger.info(f"Found {len(results)} results in {elapsed_time:.4f} seconds")
        
        # Print top 3 results
        logger.info("Top results:")
        for i, hit in enumerate(results[:3]):
            subject = hit.get('subject_name', '[No Subject]')
            score = hit.get('_weight', 0)
            logger.info(f"  {i+1}. {subject} (Score: {score:.4f})")
        
        logger.info("-" * 50)

def main():
    """Run the search performance test."""
    # Test queries
    queries = [
        "プログラミングの授業",
        "データサイエンス",
        "人工知能",
        "英語で行われる授業",
        "微分積分"
    ]
    
    test_search_performance(queries)

if __name__ == "__main__":
    main()
