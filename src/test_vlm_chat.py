#!/usr/bin/env python3
"""
Test client for VLM-enhanced SFC chat server.
"""
import os
import sys
import argparse
import json
import base64
from pathlib import Path
from typing import Optional

# Add the project root to the Python path
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

try:
    import requests
    from PIL import Image
    import io
except ImportError as e:
    print(f"Required dependency missing: {e}")
    print("Install with: pip install requests pillow")
    sys.exit(1)

from src.config import API_HOST, API_PORT, API_PASSWORD, API_REQUIRE_AUTH

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded}"

def test_text_chat(
    query: str, 
    server_url: str, 
    api_key: Optional[str] = None,
    verbose: bool = False
) -> None:
    """Test text-only chat."""
    print(f"🔍 Text Query: {query}")
    print("Thinking..." + " " * 10, end="\r")
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    
    payload = {
        "user_input": query,
        "include_search_info": verbose
    }
    
    try:
        response = requests.post(
            f"{server_url}/chat",
            json=payload,
            headers=headers,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Response:")
            print(result["response"])
            
            if verbose and result.get("search_results"):
                print("\n📚 Search Results:")
                for i, result_item in enumerate(result["search_results"][:3], 1):
                    print(f"  {i}. {result_item.get('subject_name', 'N/A')}")
                    print(f"     Score: {result_item.get('distance', 'N/A'):.3f}")
            
            if result.get("model_info"):
                print(f"\n🤖 Model Info: {result['model_info']}")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out. The server might be processing or starting up.")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API server.")
        print(f"Make sure the server is running with: python scripts/start_vlm_server.py")
        print(f"Server address: {server_url}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_multimodal_chat(
    query: str,
    image_path: str,
    server_url: str,
    api_key: Optional[str] = None,
    verbose: bool = False
) -> None:
    """Test multimodal chat with image."""
    print(f"🔍 Multimodal Query: {query}")
    print(f"🖼️  Image: {image_path}")
    print("Processing image and text..." + " " * 10, end="\r")
    
    # Encode image
    try:
        image_base64 = encode_image_to_base64(image_path)
    except Exception as e:
        print(f"❌ Error encoding image: {e}")
        return
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    
    payload = {
        "user_input": query,
        "image_base64": image_base64,
        "include_search_info": verbose
    }
    
    try:
        response = requests.post(
            f"{server_url}/multimodal-chat",
            json=payload,
            headers=headers,
            timeout=180
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ VLM Response:")
            print(result["response"])
            
            if verbose and result.get("search_results"):
                print("\n📚 Search Results:")
                for i, result_item in enumerate(result["search_results"][:3], 1):
                    print(f"  {i}. {result_item.get('subject_name', 'N/A')}")
            
            if result.get("model_info"):
                print(f"\n🤖 Model Info: {result['model_info']}")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out. VLM processing can take time.")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API server.")
        print(f"Make sure the VLM server is running with: python scripts/start_vlm_server.py")
        print(f"Server address: {server_url}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_file_upload_chat(
    query: str,
    image_path: str,
    server_url: str,
    api_key: Optional[str] = None
) -> None:
    """Test file upload chat."""
    print(f"🔍 Upload Query: {query}")
    print(f"📁 Uploading: {image_path}")
    print("Uploading and processing..." + " " * 10, end="\r")
    
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    try:
        with open(image_path, "rb") as image_file:
            files = {"image": image_file}
            data = {
                "user_input": query,
                "include_search_info": "false"
            }
            
            response = requests.post(
                f"{server_url}/upload-chat",
                files=files,
                data=data,
                headers=headers,
                timeout=180
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Upload Response:")
            print(result["response"])
            
            if result.get("model_info"):
                print(f"\n🤖 Model Info: {result['model_info']}")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

def list_vlm_models(server_url: str, api_key: Optional[str] = None) -> None:
    """List available VLM models."""
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    try:
        response = requests.get(f"{server_url}/vlm/models", headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print("🤖 Available VLM Models:")
            for model in result["available_models"]:
                current = " (current)" if model == result.get("current_model") else ""
                print(f"  - {model}{current}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test VLM-enhanced SFC chat server")
    parser.add_argument("query", nargs="?", default="SFCのプログラミング関連授業について教えてください", help="Chat query")
    parser.add_argument("-i", "--image", help="Image file path for multimodal chat")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed information")
    parser.add_argument("-k", "--api-key", help="API key for authentication")
    parser.add_argument("--host", default=API_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=API_PORT, help="Server port")
    parser.add_argument("--upload", action="store_true", help="Use file upload endpoint")
    parser.add_argument("--list-models", action="store_true", help="List available VLM models")
    
    args = parser.parse_args()
    
    # Determine server URL
    server_url = f"http://{args.host}:{args.port}"
    
    # Determine API key
    api_key = args.api_key
    if not api_key and API_REQUIRE_AUTH:
        api_key = API_PASSWORD
    
    print("=" * 60)
    print("  SFC-LLM VLMテストクライアント")
    print("=" * 60)
    print(f"Server: {server_url}")
    print(f"Auth: {'Enabled' if api_key else 'Disabled'}")
    print()
    
    # List models if requested
    if args.list_models:
        list_vlm_models(server_url, api_key)
        return
    
    # Test based on arguments
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image file not found: {args.image}")
            return
        
        if args.upload:
            test_file_upload_chat(args.query, args.image, server_url, api_key)
        else:
            test_multimodal_chat(args.query, args.image, server_url, api_key, args.verbose)
    else:
        test_text_chat(args.query, server_url, api_key, args.verbose)

if __name__ == "__main__":
    main()