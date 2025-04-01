#!/usr/bin/env python
"""
Test script to check Milvus collections.
"""
import logging
import os
import sys
from pymilvus import MilvusClient

# Add the project root to the Python path when run directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.config import MILVUS_DB_NAME

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Check Milvus collections and their status."""
    logger.info(f"Connecting to Milvus database: {MILVUS_DB_NAME}")
    client = MilvusClient(MILVUS_DB_NAME)
    
    collections = client.list_collections()
    logger.info(f"Found {len(collections)} collections: {collections}")
    
    for collection in collections:
        logger.info(f"Checking collection: {collection}")
        
        # Get collection statistics
        stats = client.get_collection_stats(collection)
        row_count = stats.get("row_count", 0)
        logger.info(f"  • Row count: {row_count}")
        
        # Get collection schema
        try:
            schema = client.describe_collection(collection)
            if 'fields' in schema:
                logger.info(f"  • Fields: {[field.get('name', 'unknown') for field in schema['fields']]}")
            else:
                logger.info(f"  • Schema structure: {schema.keys()}")
        except Exception as e:
            logger.error(f"  • Error getting schema: {e}")
        
        # Check indexes
        indexes = client.list_indexes(collection)
        logger.info(f"  • Indexes: {indexes}")

if __name__ == "__main__":
    main()
