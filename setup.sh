#!/bin/bash

# エラー時にスクリプトを停止
set -e

echo "=== 環境構築を開始します ==="

# Poetry のインストール確認とインストール
if ! command -v poetry &> /dev/null; then
    echo "Poetry が見つかりません。インストール中..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "Poetry は既にインストールされています。"
fi

# プロジェクトルートディレクトリを設定
PROJECT_DIR=$(pwd)
echo "プロジェクトディレクトリ: $PROJECT_DIR"

# Poetry 仮想環境を作成
echo "Poetry で仮想環境を作成中..."
poetry install

# 仮想環境の場所を確認
VENV_PATH=$(poetry env info --path)
echo "仮想環境の場所: $VENV_PATH"

# 仮想環境内に `requirements.txt` をインストール
if [ -f "requirements.txt" ]; then
    echo "requirements.txt から依存関係をインストール中..."
    poetry run pip install -r requirements.txt
else
    echo "requirements.txt が見つかりません。スキップします。"
fi

echo "=== 環境構築が完了しました ==="
echo "仮想環境をアクティベートするには以下を実行してください:"
echo "poetry shell"

