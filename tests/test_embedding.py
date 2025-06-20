#!/usr/bin/env python3
"""
Tests for embedding utilities.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.embedding import EmbeddingFunction, get_embedding_fn, encode_query


class TestEmbedding(unittest.TestCase):
    """Test embedding functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_texts = [
            "プログラミングの基礎を学ぶ授業です",
            "データサイエンスの応用について",
            "This is an English text for testing"
        ]
    
    def test_embedding_function_initialization(self):
        """Test EmbeddingFunction initialization."""
        embedding_fn = EmbeddingFunction("all-MiniLM-L6-v2", "cpu")
        
        self.assertIsNotNone(embedding_fn.model)
        self.assertEqual(embedding_fn.dim, 384)  # Expected dimension for all-MiniLM-L6-v2
    
    def test_encode_queries(self):
        """Test query encoding."""
        embedding_fn = get_embedding_fn()
        
        # Test single query
        result = embedding_fn.encode_queries(["プログラミングを学びたい"])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 384)
        
        # Test multiple queries
        result = embedding_fn.encode_queries(self.test_texts)
        self.assertEqual(len(result), 3)
        for embedding in result:
            self.assertEqual(len(embedding), 384)
    
    def test_encode_documents(self):
        """Test document encoding."""
        embedding_fn = get_embedding_fn()
        
        result = embedding_fn.encode_documents(self.test_texts)
        self.assertEqual(len(result), 3)
        for embedding in result:
            self.assertEqual(len(embedding), 384)
    
    def test_encode_query_function(self):
        """Test standalone encode_query function."""
        query = "プログラミングの授業について"
        result = encode_query(query)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 384)
    
    def test_embedding_consistency(self):
        """Test that same text produces same embedding."""
        text = "同じテキストは同じ埋め込みを生成するべき"
        
        embedding1 = encode_query(text)
        embedding2 = encode_query(text)
        
        # Should be exactly the same (from cache or deterministic)
        self.assertEqual(embedding1, embedding2)
    
    def test_get_embedding_fn_caching(self):
        """Test that get_embedding_fn returns cached instance."""
        fn1 = get_embedding_fn()
        fn2 = get_embedding_fn()
        
        # Should be the same instance due to lru_cache
        self.assertIs(fn1, fn2)


if __name__ == "__main__":
    unittest.main()