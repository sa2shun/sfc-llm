#!/usr/bin/env python3
"""
Integration tests for SFC-LLM application.
"""
import os
import sys
import time
import unittest
import requests
import threading
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import API_HOST, API_PORT


class TestIntegration(unittest.TestCase):
    """Integration tests that require a running server."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class - start server if needed."""
        cls.server_url = f"http://{API_HOST}:{API_PORT}"
        cls.server_process = None
        
        # Check if server is already running
        try:
            response = requests.get(f"{cls.server_url}/health", timeout=2)
            if response.status_code == 200:
                cls.server_running = True
                print("Server already running")
                return
        except:
            pass
        
        # Start server for testing
        cls.server_running = False
        print("Starting server for integration tests...")
        
        # Set environment for test server
        env = os.environ.copy()
        env["SFC_LLM_API_REQUIRE_AUTH"] = "false"
        env["SFC_LLM_LOG_LEVEL"] = "WARNING"
        
        try:
            cls.server_process = subprocess.Popen(
                [sys.executable, "-m", "src.chat_server"],
                cwd=str(project_root),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for server to start
            for _ in range(30):  # Wait up to 30 seconds
                try:
                    response = requests.get(f"{cls.server_url}/health", timeout=1)
                    if response.status_code == 200:
                        cls.server_running = True
                        print("Test server started successfully")
                        break
                except:
                    time.sleep(1)
            
            if not cls.server_running:
                raise Exception("Failed to start test server")
                
        except Exception as e:
            print(f"Could not start server for integration tests: {e}")
            cls.server_running = False
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test class - stop server if we started it."""
        if cls.server_process:
            cls.server_process.terminate()
            cls.server_process.wait(timeout=10)
            print("Test server stopped")
    
    def test_server_health(self):
        """Test server health endpoint."""
        if not self.server_running:
            self.skipTest("Server not running")
        
        response = requests.get(f"{self.server_url}/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("status", data)
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        if not self.server_running:
            self.skipTest("Server not running")
        
        response = requests.get(f"{self.server_url}/")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("SFC-LLM", data["message"])
    
    def test_chat_endpoint_simple(self):
        """Test chat endpoint with simple query."""
        if not self.server_running:
            self.skipTest("Server not running")
        
        payload = {
            "user_input": "Hello",
            "include_search_info": False
        }
        
        response = requests.post(
            f"{self.server_url}/chat",
            json=payload,
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response", data)
        self.assertIsInstance(data["response"], str)
        self.assertTrue(len(data["response"]) > 0)
    
    def test_chat_endpoint_japanese(self):
        """Test chat endpoint with Japanese query."""
        if not self.server_running:
            self.skipTest("Server not running")
        
        payload = {
            "user_input": "プログラミングについて教えてください",
            "include_search_info": True
        }
        
        response = requests.post(
            f"{self.server_url}/chat",
            json=payload,
            timeout=60  # Japanese processing might take longer
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response", data)
        self.assertIsInstance(data["response"], str)
        
        # Should include search info for this query
        if data.get("search_results"):
            self.assertIsInstance(data["search_results"], list)
    
    def test_chat_endpoint_invalid_input(self):
        """Test chat endpoint with invalid input."""
        if not self.server_running:
            self.skipTest("Server not running")
        
        # Test empty input
        payload = {"user_input": ""}
        response = requests.post(f"{self.server_url}/chat", json=payload)
        # Should handle gracefully (either 400 or return a response)
        self.assertIn(response.status_code, [200, 400, 422])
        
        # Test missing input
        payload = {}
        response = requests.post(f"{self.server_url}/chat", json=payload)
        self.assertIn(response.status_code, [400, 422])


class TestVLMIntegration(unittest.TestCase):
    """Integration tests for VLM functionality."""
    
    def test_vlm_server_endpoints(self):
        """Test VLM server endpoints exist."""
        # This test doesn't require a running VLM server
        # It just checks that the endpoints are properly defined
        
        from src.vlm_chat_server import app
        
        # Check that key endpoints exist
        routes = [route.path for route in app.routes]
        
        expected_routes = [
            "/",
            "/chat", 
            "/multimodal-chat",
            "/upload-chat",
            "/vlm/models",
            "/vlm/switch-model",
            "/health"
        ]
        
        for route in expected_routes:
            self.assertIn(route, routes)


if __name__ == "__main__":
    # Set up test environment
    os.environ["SFC_LLM_API_REQUIRE_AUTH"] = "false"
    os.environ["SFC_LLM_LOG_LEVEL"] = "WARNING"
    
    unittest.main(verbosity=2)