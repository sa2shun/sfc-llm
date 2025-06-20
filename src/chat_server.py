"""
FastAPI server for the SFC-LLM chat application.
"""
from fastapi import FastAPI, HTTPException, Request, Depends, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import logging
import time
import traceback
import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# Add the project root to the Python path when run directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from utils.llm import generate_response
from src.milvus_search import search_syllabus
from src.prompts import (
    get_rag_decision_prompt,
    get_rag_response_prompt,
    get_general_response_prompt
)
from src.config import API_HOST, API_PORT, API_PASSWORD, API_REQUIRE_AUTH

# Import config for logging settings
from src.config import LOG_LEVEL, LOG_FORMAT

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

# FastAPI app definition
app = FastAPI(
    title="SFC-LLM API",
    description="API for SFC syllabus search and LLM-powered responses",
    version="1.0.0"
)

# API key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """Validate API key if authentication is required."""
    if not API_REQUIRE_AUTH:
        return True
    
    if api_key_header and api_key_header == API_PASSWORD:
        return True
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "ApiKey"},
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    reply: str
    rag_used: bool
    processing_time: float
    rag_context: Optional[str] = None

def should_use_rag_heuristic(user_input: str) -> bool:
    """
    Quick heuristic to determine if RAG should be used.
    This helps avoid LLM calls for obvious cases.
    """
    query_lower = user_input.lower()
    
    # Japanese keywords that likely need SFC syllabus data
    sfc_keywords = [
        "授業", "科目", "講義", "セミナー", "演習", "ゼミ", "単位", "履修",
        "プログラミング", "データサイエンス", "AI", "機械学習", "統計",
        "経済", "政策", "メディア", "デザイン", "環境", "バイオ",
        "英語", "中国語", "韓国語", "language", "english",
        "学部", "大学院", "faculty", "graduate",
        "春学期", "秋学期", "semester", "開講",
        "sfc", "湘南藤沢", "慶應"
    ]
    
    # If query contains SFC-related keywords, likely needs RAG
    if any(keyword in query_lower for keyword in sfc_keywords):
        return True
    
    # General greetings or questions that don't need syllabus data
    general_keywords = [
        "こんにちは", "hello", "はじめまして", "ありがとう", "thank you",
        "今日", "明日", "天気", "weather", "時間", "time",
        "どうして", "なぜ", "why", "how are you", "元気"
    ]
    
    # If query is general greeting/question, likely doesn't need RAG
    if any(keyword in query_lower for keyword in general_keywords):
        return False
    
    # For ambiguous cases, default to using RAG
    return True

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for the API."""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."}
    )

def _process_chat_sync(user_input: str) -> dict:
    """Synchronous chat processing function for thread pool."""
    # Step 1: Fast heuristic check first
    rag_needed_heuristic = should_use_rag_heuristic(user_input)
    
    if not rag_needed_heuristic:
        # Skip RAG decision LLM call for obvious general queries
        logger.info("Heuristic determined no RAG needed")
        final_prompt = get_general_response_prompt(user_input)
        reply = generate_response(final_prompt)
        return {
            "reply": reply,
            "rag_used": False,
            "rag_context": None
        }
    
    # Step 2: Use LLM for ambiguous cases
    decision_prompt = get_rag_decision_prompt(user_input)
    decision = generate_response(decision_prompt)
    rag_used = "NEED_RAG" in decision
    
    logger.info(f"RAG decision: {decision.strip()} (RAG used: {rag_used})")
    
    # Step 3: Generate response based on RAG decision
    if rag_used:
        # Get relevant documents from Milvus
        hits = search_syllabus(user_input)
        logger.info(f"Found {len(hits)} relevant documents")
        
        # Format context from search results
        context_items = []
        for h in hits:
            # Get subject name from either top level or entity
            subject = h.get('subject_name', '[No Subject]')
            matched_field = h.get('_matched_field', 'unknown')
            score = h.get('_weight', 0)
            
            # Add metadata about the course
            metadata = []
            if h.get('faculty') is not None:
                metadata.append("学部" if h.get('faculty') else "大学院")
            if h.get('category'):
                metadata.append(f"分野: {h.get('category')}")
            if h.get('credits'):
                metadata.append(f"{h.get('credits')}単位")
            if h.get('language'):
                metadata.append(f"言語: {h.get('language')}")
            
            # Get URL from either top level or entity
            url = h.get('url', 'N/A')
            
            context_items.append(
                f"【{subject}】({', '.join(metadata)})\n"
                f"URL: {url}\n"
                f"マッチ: {matched_field} (スコア: {score:.2f})"
            )
        
        context = "\n\n".join(context_items)
        
        # Generate response using RAG
        final_prompt = get_rag_response_prompt(user_input, context)
        reply = generate_response(final_prompt)
        
        return {
            "reply": reply,
            "rag_used": True,
            "rag_context": context
        }
    else:
        # Generate general response without RAG
        final_prompt = get_general_response_prompt(user_input)
        reply = generate_response(final_prompt)
        
        return {
            "reply": reply,
            "rag_used": False,
            "rag_context": None
        }

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, authenticated: bool = Depends(get_api_key)):
    """
    Process a chat request and return a response.
    
    Args:
        req: The chat request containing the user's input
        
    Returns:
        A response containing the LLM's reply
    """
    start_time = time.time()
    user_input = req.user_input
    
    if not user_input or not user_input.strip():
        raise HTTPException(status_code=400, detail="User input cannot be empty")
    
    logger.info(f"Received chat request: {user_input[:50]}...")
    
    try:
        # Process chat in thread pool for CPU-intensive operations
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, _process_chat_sync, user_input)
        
        processing_time = time.time() - start_time
        logger.info(f"Request processed in {processing_time:.2f} seconds")
        
        # Prepare response
        result["processing_time"] = processing_time
        return result
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing request")

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "auth_required": API_REQUIRE_AUTH}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.chat_server:app", host=API_HOST, port=API_PORT, reload=True)
