#!/usr/bin/env python3
"""
VLM-enhanced SFC chat server startup script.
"""
import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import API_HOST, API_PORT

def main():
    parser = argparse.ArgumentParser(description="Start VLM-enhanced SFC chat server")
    parser.add_argument(
        "--global", 
        dest="global_mode", 
        action="store_true",
        help="Enable global access (0.0.0.0)"
    )
    parser.add_argument(
        "--no-auth", 
        dest="no_auth", 
        action="store_true",
        help="Disable API authentication"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=API_PORT,
        help=f"Server port (default: {API_PORT})"
    )
    parser.add_argument(
        "--vlm-model",
        default="llava-mistral-7b",
        help="VLM model to use (default: llava-mistral-7b)"
    )
    
    args = parser.parse_args()
    
    # Set environment variables
    if args.global_mode:
        os.environ["SFC_LLM_API_HOST"] = "0.0.0.0"
    else:
        os.environ["SFC_LLM_API_HOST"] = "127.0.0.1"
    
    if args.no_auth:
        os.environ["SFC_LLM_API_REQUIRE_AUTH"] = "false"
    
    os.environ["SFC_LLM_API_PORT"] = str(args.port)
    os.environ["SFC_LLM_VLM_MODEL"] = args.vlm_model
    
    # Display startup info
    print("=" * 60)
    print("  SFC-LLM VLMサーバー起動スクリプト")
    print("=" * 60)
    print()
    print("Vision Language Model対応のSFCシラバス検索チャットサーバー")
    print()
    
    mode = "グローバルモード (0.0.0.0)" if args.global_mode else "ローカルモード (127.0.0.1)"
    auth = "認証なし" if args.no_auth else "認証あり"
    
    print(f"実行モード: {mode} - {auth}")
    print(f"ポート: {args.port}")
    print(f"VLMモデル: {args.vlm_model}")
    print()
    print()
    print("-" * 60)
    print("  VLM機能")
    print("-" * 60)
    print("1. 画像 + テキストの同時処理")
    print("2. シラバス画像の自動解析")
    print("3. マルチモーダル検索機能")
    print()
    print("-" * 60)
    print("  使用方法")
    print("-" * 60)
    print("1. テキストのみ:")
    print('   python src/test_chat.py "プログラミングの授業を教えてください"')
    print()
    print("2. 画像 + テキスト:")
    print("   curl -X POST http://localhost:9001/upload-chat \\")
    print("        -F 'user_input=この画像のシラバスについて教えて' \\")
    print("        -F 'image=@syllabus.jpg'")
    print()
    print("3. Base64画像:")
    print("   POST /multimodal-chat")
    print("   {\"user_input\": \"質問\", \"image_base64\": \"data:image/jpeg;base64,...\"}")
    print()
    print("4. 利用可能なVLMモデル確認:")
    print("   curl http://localhost:9001/vlm/models")
    print()
    print("5. このサーバーを停止するには Ctrl+C を押してください")
    print("-" * 60)
    
    # Check if server is already running
    try:
        import requests
        host = os.environ["SFC_LLM_API_HOST"]
        port = args.port
        response = requests.get(f"http://{host}:{port}/health", timeout=2)
        if response.status_code == 200:
            print(f"サーバーは既に起動しています (http://{host}:{port})")
            return
    except:
        pass
    
    # Start server
    try:
        print(f"サーバーを起動しています (http://{os.environ['SFC_LLM_API_HOST']}:{args.port})...")
        print("サーバープロセスを開始します...")
        
        # Run the VLM server
        subprocess.run([
            sys.executable, "-m", "src.vlm_chat_server"
        ], cwd=project_root)
        
    except KeyboardInterrupt:
        print("\nサーバーを停止しています...")
    except Exception as e:
        print(f"サーバー起動エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()