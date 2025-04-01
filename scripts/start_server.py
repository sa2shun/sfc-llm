#!/usr/bin/env python
"""
Helper script to start the SFC-LLM server and provide usage instructions.
"""
import os
import sys
import subprocess
import time
import signal
import logging

# Add the project root to the Python path when run directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.config import API_HOST, API_PORT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print a welcome banner."""
    print("\n" + "=" * 60)
    print("  SFC-LLM サーバー起動スクリプト")
    print("=" * 60)
    print("\nKeio SFCの授業シラバスを検索し、LLMで自然言語回答するチャットアプリケーション\n")

def print_instructions():
    """Print usage instructions."""
    print("\n" + "-" * 60)
    print("  使用方法")
    print("-" * 60)
    print("1. サーバーが起動したら、別のターミナルで以下のコマンドを実行してテストできます：")
    print(f"   python src/test_chat.py \"プログラミングの授業を教えてください\"")
    print("\n2. 詳細情報を表示するには -v オプションを追加：")
    print(f"   python src/test_chat.py -v \"データサイエンスの授業はありますか\"")
    print("\n3. 検索性能をテストするには：")
    print(f"   python scripts/test_search_performance.py")
    print("\n4. このサーバーを停止するには Ctrl+C を押してください")
    print("-" * 60 + "\n")

def start_server():
    """Start the SFC-LLM server."""
    print_banner()
    
    # Check if the server is already running
    try:
        import requests
        response = requests.get(f"http://{API_HOST}:{API_PORT}/health", timeout=2)
        if response.status_code == 200:
            logger.info(f"サーバーは既に起動しています (http://{API_HOST}:{API_PORT})")
            print_instructions()
            return
    except:
        pass
    
    # Start the server
    logger.info(f"サーバーを起動しています (http://{API_HOST}:{API_PORT})...")
    
    try:
        # Use subprocess to start the server
        server_cmd = [sys.executable, "-m", "src.chat_server"]
        
        # Print instructions
        print_instructions()
        
        # Start the server process
        logger.info("サーバープロセスを開始します...")
        process = subprocess.Popen(server_cmd)
        
        # Handle Ctrl+C to gracefully shut down the server
        def signal_handler(sig, frame):
            logger.info("サーバーを停止しています...")
            process.terminate()
            process.wait()
            logger.info("サーバーが停止しました")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Wait for the server process to complete
        process.wait()
    
    except Exception as e:
        logger.error(f"サーバー起動中にエラーが発生しました: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    start_server()
