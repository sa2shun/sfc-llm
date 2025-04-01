# test_chat.py
import requests

response = requests.post(
    "http://localhost:8001/chat",  # FastAPI 側のエンドポイント
    headers={"Content-Type": "application/json"},
    json={
        "user_input": "数学の授業について教えて"
    }
)

print(response.json())
