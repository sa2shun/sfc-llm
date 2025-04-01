"""
FastAPI server for the SFC-LLM chat application.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import time
import traceback
import os
import sys

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
from src.config import API_HOST, API_PORT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app definition
app = FastAPI(
    title="SFC-LLM API",
    description="API for SFC syllabus search and LLM-powered responses",
    version="1.0.0"
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
    rag_context: str = None

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for the API."""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."}
    )

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
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
    
    # Step 1: Determine if RAG is needed
    decision_prompt = get_rag_decision_prompt(user_input)
    decision = generate_response(decision_prompt)
    rag_used = "NEED_RAG" in decision
    
    logger.info(f"RAG decision: {decision.strip()} (RAG used: {rag_used})")
    
    # Step 2: Generate response based on RAG decision
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
    else:
        # Generate general response without RAG
        final_prompt = get_general_response_prompt(user_input)
        reply = generate_response(final_prompt)
    
    processing_time = time.time() - start_time
    logger.info(f"Request processed in {processing_time:.2f} seconds")
    
    # Prepare response
    response = {
        "reply": reply,
        "rag_used": rag_used,
        "processing_time": processing_time
    }
    
    # Include RAG context if used
    if rag_used:
        response["rag_context"] = context
    
    return response

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.chat_server:app", host=API_HOST, port=API_PORT, reload=True)
