"""
Prompt templates for the SFC-LLM application.
"""
from typing import List, Dict, Any
import re

def should_use_rag(user_input: str) -> bool:
    """
    Determine if RAG should be used for a given user query.
    
    Args:
        user_input: The user's query
        
    Returns:
        True if RAG should be used, False otherwise
    """
    # Keywords that indicate SFC course-related queries
    course_keywords = [
        'プログラミング', '授業', '科目', '履修', 'シラバス', '講義', 'ゼミ', '研究会',
        'データサイエンス', '英語', '言語', '情報', '政策', '環境', 'メディア',
        '教授', '先生', '単位', '課題', '試験', 'レポート', '成績', '評価',
        'course', 'class', 'subject', 'syllabus', 'professor', 'credit', 'grade'
    ]
    
    # SFC-specific keywords
    sfc_keywords = ['sfc', 'ソニー', '湘南藤沢', '藤沢', '慶應', 'keio']
    
    user_lower = user_input.lower()
    
    # Check for course-related keywords
    has_course_keywords = any(keyword in user_lower for keyword in course_keywords)
    
    # Check for SFC-specific keywords
    has_sfc_keywords = any(keyword in user_lower for keyword in sfc_keywords)
    
    # Check for general patterns that suggest course queries
    course_patterns = [
        r'.*について教えて',
        r'.*はありますか',
        r'.*を探して',
        r'.*がしたい',
        r'.*を学びたい',
        r'おすすめ.*',
        r'どんな.*',
        r'.*の特徴'
    ]
    
    has_course_patterns = any(re.search(pattern, user_input) for pattern in course_patterns)
    
    # Use RAG if:
    # 1. Contains course keywords, OR
    # 2. Contains SFC keywords and course patterns, OR  
    # 3. Question seems academic in nature
    return has_course_keywords or (has_sfc_keywords and has_course_patterns)

def create_rag_prompt(user_input: str, search_results: List[Dict[str, Any]]) -> str:
    """
    Create a RAG prompt from user input and search results.
    
    Args:
        user_input: The user's query
        search_results: List of search results from Milvus
        
    Returns:
        A formatted prompt for the LLM
    """
    # Format search results into context
    context_parts = []
    
    for i, result in enumerate(search_results[:5], 1):  # Limit to top 5 results
        context_part = f"【授業{i}】\n"
        
        # Add available fields from search results
        if 'subject_name' in result:
            context_part += f"科目名: {result['subject_name']}\n"
        if 'category' in result:
            context_part += f"分野: {result['category']}\n"
        if 'summary' in result:
            context_part += f"概要: {result['summary']}\n"
        if 'goals' in result:
            context_part += f"目標: {result['goals']}\n"
        if 'schedule' in result:
            context_part += f"計画: {result['schedule']}\n"
        
        # Add distance/similarity score if available
        if 'distance' in result:
            context_part += f"関連度: {result['distance']:.3f}\n"
        
        context_parts.append(context_part)
    
    context = "\n".join(context_parts)
    
    return get_rag_response_prompt(user_input, context)

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
