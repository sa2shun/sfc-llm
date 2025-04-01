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
import argparse
#!/usr/bin/env python

# Add the project root to the Python path when run directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.config import API_HOST, API_PORT, API_REQUIRE_AUTH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner(is_local):
    """Print a welcome banner."""
    print("\n" + "=" * 60)
    print("  SFC-LLM サーバー起動スクリプト")
    print("=" * 60)
    print("\nKeio SFCの授業シラバスを検索し、LLMで自然言語回答するチャットアプリケーション")
    
    mode = "ローカルモード" if is_local else "グローバルモード (外部アクセス可能)"
    auth = "認証必須" if API_REQUIRE_AUTH else "認証なし"
    print(f"\n実行モード: {mode} - {auth}\n")

def print_instructions(host, port, is_local):
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
    
    if not is_local and API_REQUIRE_AUTH:
        print("\n4. 外部からAPIにアクセスする場合は、X-API-Keyヘッダーに認証キーを設定してください")
        print(f"   例: curl -H \"X-API-Key: [パスワード]\" -H \"Content-Type: application/json\" -d '{{\"user_input\":\"質問\"}}' http://{host}:{port}/chat")
    
    print(f"\n{4 if is_local or not API_REQUIRE_AUTH else 5}. このサーバーを停止するには Ctrl+C を押してください")
    print("-" * 60 + "\n")

def start_server(is_local=True):
    """Start the SFC-LLM server."""
    # Set host based on mode
    host = "127.0.0.1" if is_local else "0.0.0.0"
    port = API_PORT
    
    # Set environment variables
    os.environ["SFC_LLM_API_HOST"] = host
    
    print_banner(is_local)
    
    # Check if the server is already running
    try:
        import requests
        response = requests.get(f"http://{host}:{port}/health", timeout=2)
        if response.status_code == 200:
            logger.info(f"サーバーは既に起動しています (http://{host}:{port})")
            print_instructions(host, port, is_local)
            return
    except:
        pass
    
    # Start the server
    logger.info(f"サーバーを起動しています (http://{host}:{port})...")
    
    try:
        # Use subprocess to start the server
        server_cmd = [sys.executable, "-m", "src.chat_server"]
        
        # Print instructions
        print_instructions(host, port, is_local)
        
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
    parser = argparse.ArgumentParser(description="SFC-LLM サーバー起動スクリプト")
    parser.add_argument(
        "--local", 
        action="store_true", 
        help="ローカルモードで起動 (127.0.0.1, 外部からアクセス不可)"
    )
    parser.add_argument(
        "--global", 
        dest="global_mode",
        action="store_true", 
        help="グローバルモードで起動 (0.0.0.0, 外部からアクセス可能)"
    )
    parser.add_argument(
        "--no-auth", 
        action="store_true", 
        help="認証を無効化"
    )
    
    args = parser.parse_args()
    
    # Set authentication mode
    if args.no_auth:
        os.environ["SFC_LLM_API_REQUIRE_AUTH"] = "false"
    
    # Determine server mode (default to local if neither specified)
    is_local = not args.global_mode if args.global_mode else True
    
    start_server(is_local)
