#!/usr/bin/env python3
"""
Tests for configuration module.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import validate_config, get_config_summary, PROJECT_ROOT, DATA_DIR


class TestConfig(unittest.TestCase):
    """Test configuration functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = os.environ.copy()
        
    def tearDown(self):
        """Clean up test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_project_paths(self):
        """Test that project paths are correctly set."""
        self.assertTrue(PROJECT_ROOT.exists())
        self.assertTrue(PROJECT_ROOT.is_dir())
        self.assertEqual(PROJECT_ROOT.name, "sfc-llm")
        
    def test_data_directory_creation(self):
        """Test that data directory is created."""
        # This will create the directory if it doesn't exist
        validate_config()
        self.assertTrue(DATA_DIR.exists())
        self.assertTrue(DATA_DIR.is_dir())
    
    def test_config_summary(self):
        """Test configuration summary generation."""
        summary = get_config_summary()
        
        # Check required keys
        self.assertIn("model", summary)
        self.assertIn("api", summary)
        self.assertIn("performance", summary)
        
        # Check model configuration
        model_config = summary["model"]
        self.assertIn("llm_model", model_config)
        self.assertIn("embedding_model", model_config)
        self.assertIn("vlm_model", model_config)
        self.assertIn("device", model_config)
        
    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        # Set custom environment variables
        os.environ["SFC_LLM_API_PORT"] = "8888"
        os.environ["SFC_LLM_API_HOST"] = "custom.host"
        os.environ["SFC_LLM_TEMPERATURE"] = "0.5"
        
        # Re-import config to pick up new values
        import importlib
        from src import config
        importlib.reload(config)
        
        self.assertEqual(config.API_PORT, 8888)
        self.assertEqual(config.API_HOST, "custom.host")
        self.assertEqual(config.TEMPERATURE, 0.5)
    
    def test_validate_config_without_token(self):
        """Test configuration validation without HuggingFace token."""
        # Remove HF token if present
        if "HUGGINGFACE_TOKEN" in os.environ:
            del os.environ["HUGGINGFACE_TOKEN"]
        
        # Re-import config
        import importlib
        from src import config
        importlib.reload(config)
        
        # Validation should return False without token
        self.assertFalse(config.validate_config())


if __name__ == "__main__":
    unittest.main()