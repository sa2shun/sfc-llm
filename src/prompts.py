"""
Prompt templates for the SFC-LLM application.
"""

def get_rag_decision_prompt(query: str) -> str:
    """
    Generate a prompt to decide if RAG is needed for a query.
    
    Args:
        query: The user's query
        
    Returns:
        A prompt for the LLM to decide if RAG is needed
    """
    return f"""
あなたは大学の授業に関する質問に答えるAIアシスタントです。
以下のユーザーの質問に答えるために、SFC（慶應義塾大学湘南藤沢キャンパス）の授業データベースを検索する必要があるかどうかを判断してください。

質問が以下のような場合は「NEED_RAG」と答えてください：
- 特定の授業や科目に関する質問
- SFCのカリキュラムや履修に関する質問
- 特定の分野や教授の授業に関する質問
- SFCの授業の特徴や内容に関する質問

質問が以下のような場合は「NO_RAG」と答えてください：
- 一般的な知識や情報に関する質問
- SFCの授業と関係のない質問
- あいさつや雑談

必ず「NEED_RAG」または「NO_RAG」のどちらかだけで答えてください。

ユーザーの質問: {query}
"""

def get_rag_response_prompt(query: str, context: str) -> str:
    """
    Generate a prompt for RAG-based response.
    
    Args:
        query: The user's query
        context: The retrieved context from the database
        
    Returns:
        A prompt for the LLM to generate a response using RAG
    """
    return f"""
あなたは慶應義塾大学SFC（湘南藤沢キャンパス）の授業に関する質問に答えるAIアシスタントです。
以下の授業データを参考にして、ユーザーの質問に丁寧に答えてください。

回答の際の注意点：
1. 提供された授業データの情報のみを使用してください
2. データにない情報については「その情報は持ち合わせていません」と正直に伝えてください
3. 授業データの出典（科目名など）を回答の中で自然に言及してください
4. 学生が授業選択の参考にできるよう、具体的で役立つ情報を提供してください
5. 丁寧かつ親しみやすい口調で回答してください

【授業データ】
{context}

【ユーザーの質問】
{query}

【回答】
"""

def get_general_response_prompt(query: str) -> str:
    """
    Generate a prompt for general (non-RAG) response.
    
    Args:
        query: The user's query
        
    Returns:
        A prompt for the LLM to generate a general response
    """
    return f"""
あなたは慶應義塾大学SFC（湘南藤沢キャンパス）の授業に関する質問に答えるAIアシスタントです。
以下のユーザーの質問に丁寧に答えてください。

回答の際の注意点：
1. SFCの授業に関する具体的な情報がない場合は、「その情報は持ち合わせていません」と正直に伝えてください
2. わからない情報については「その情報は持ち合わせていません」と正直に伝えてください
3. 丁寧かつ親しみやすい口調で回答してください
4. 必要に応じて、より具体的な質問をするようユーザーに促してください

【ユーザーの質問】
{query}

【回答】
"""
