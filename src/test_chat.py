#!/usr/bin/env python
"""
Test client for the SFC-LLM chat API.
"""
import requests
import argparse
import time
import os
import sys

# Add the project root to the Python path when run directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.config import API_HOST, API_PORT

def chat_with_llm(query: str, verbose: bool = True):
    """
    Send a chat request to the API and print the response.
    
    Args:
        query: The user's query
        verbose: Whether to print detailed information
    """
    url = f"http://{API_HOST}:{API_PORT}/chat"
    
    print(f"\n🔍 Query: {query}")
    print("Thinking...", end="", flush=True)
    
    try:
        start_time = time.time()
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"user_input": query},
            timeout=None  # No timeout - wait as long as needed for LLM inference
        )
        elapsed = time.time() - start_time
        
        print("\r" + " " * 10 + "\r", end="")  # Clear "Thinking..." text
        
        if response.status_code == 200:
            data = response.json()
            
            # Print the response
            print(f"\n🤖 Response:")
            print(f"{data['reply']}")
            
            # Always print additional info
            print(f"\n📊 Details:")
            print(f"  • RAG Used: {'Yes' if data['rag_used'] else 'No'}")
            print(f"  • API Processing Time: {data['processing_time']:.2f}s")
            print(f"  • Total Time (including network): {elapsed:.2f}s")
            
            # Print RAG context if available
            if data['rag_used'] and 'rag_context' in data:
                print(f"\n🔍 RAG Context:")
                print(data['rag_context'])
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
    
    except requests.exceptions.ConnectionError:
        print("\r" + " " * 10 + "\r", end="")  # Clear "Thinking..." text
        print("\n❌ Error: Could not connect to the API server.")
        print(f"Make sure the server is running with: python -m src.chat_server")
        print(f"Server address: {url}")
    
    except requests.exceptions.Timeout:
        print("\r" + " " * 10 + "\r", end="")  # Clear "Thinking..." text
        print("\n❌ Error: Request timed out.")
        print("The server is taking too long to respond. This could be due to:")
        print("1. The server is processing a large request")
        print("2. The server is experiencing high load")
        print("3. The LLM model is still loading")
        print("\nTry increasing the timeout or check server logs for issues.")
    
    except Exception as e:
        print("\r" + " " * 10 + "\r", end="")  # Clear "Thinking..." text
        print(f"\n❌ Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Test client for SFC-LLM chat API")
    parser.add_argument("query", nargs="?", default="数学の授業について教えて", 
                        help="Query to send to the API")
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="Print detailed information")
    args = parser.parse_args()
    
    chat_with_llm(args.query, args.verbose)

if __name__ == "__main__":
    main()
