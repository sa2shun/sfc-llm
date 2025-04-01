from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import torch
import pathlib
import os
from .milvus_search import search_syllabus


HF_MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
LOCAL_DIR = f"/raid/{os.environ['USER']}/meta-llama_Llama-3.1-70B-Instruct"
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# ✅ 重みファイルの存在を確認する（例: safetensors）
weight_files = [
    "pytorch_model.bin",
    "model.safetensors",
    "tf_model.h5",
    "model.ckpt.index",
    "flax_model.msgpack"
]
if not any((pathlib.Path(LOCAL_DIR) / f).exists() for f in weight_files):
    print(f"モデルが見つかりません。{HF_MODEL_ID} をダウンロードします...")
    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir=LOCAL_DIR,
        token=HF_TOKEN,
        local_dir_use_symlinks=False,
        ignore_patterns=[]  # 重要：一部しかダウンロードしないことを防ぐ
    )
else:
    print(f"既存のモデルを使用します: {LOCAL_DIR}")


tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_DIR,
    token=HF_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token

# モデル読み込み
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_DIR,
    device_map="auto",
    torch_dtype="auto",
    token=HF_TOKEN
)
model.eval()

# FastAPI アプリ定義
app = FastAPI()

class ChatRequest(BaseModel):
    user_input: str

def generate_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

@app.post("/chat")
def chat(req: ChatRequest):
    user_input = req.user_input

    # ステップ1: RAGが必要かを判断
    decision_prompt = f"""
次のユーザーの質問に答えるには、データベース検索が必要かを判断してください。
必要であれば "NEED_RAG"、不要なら "NO_RAG" とだけ答えてください。

質問: {user_input}
"""
    decision = generate_response(decision_prompt)

    if "NEED_RAG" in decision:
        hits = search_syllabus(user_input)
        print("hits sample:", hits)  # ← 確認用

        context = "\n".join([
            f"{h.get('subject_name', '[No Subject]')}: {h.get('summary', '[No Summary]')}"
            for h in hits
        ])
        final_prompt = f"""
ユーザー: {user_input}
以下の授業データを使って自然に回答してください。

{context}
"""
    else:
        final_prompt = user_input

    # ステップ2: 回答生成
    reply = generate_response(f"あなたは大学の授業を案内するAIです。\nユーザー: {final_prompt}")

    return {"reply": reply}
