"""
VLM-enhanced chat server for SFC syllabus with vision capabilities.
"""
import os
import sys
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union, List, Dict, Any
from contextlib import asynccontextmanager

# Add the project root to the Python path
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from PIL import Image
import io

from src.config import API_HOST, API_PORT, API_PASSWORD, API_REQUIRE_AUTH, LOG_LEVEL
from src.milvus_search import search_syllabus
from src.prompts import create_rag_prompt, should_use_rag
from utils.llm import generate_response
from utils.vlm_engine import (
    get_vlm_engine, 
    process_multimodal_query,
    VLM_AVAILABLE,
    list_available_vlm_models,
    switch_vlm_model
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

# Security
security = HTTPBearer(auto_error=False)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key if authentication is required."""
    if not API_REQUIRE_AUTH:
        return True
    
    if not credentials or credentials.credentials != API_PASSWORD:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting VLM-enhanced SFC chat server...")
    
    # Initialize VLM engine
    if VLM_AVAILABLE:
        vlm = get_vlm_engine()
        if vlm:
            logger.info("VLM engine initialized successfully")
        else:
            logger.warning("VLM engine failed to initialize")
    else:
        logger.warning("VLM dependencies not available")
    
    yield
    
    logger.info("Shutting down VLM chat server...")

# FastAPI app
app = FastAPI(
    title="SFC-LLM VLM Chat Server",
    description="Vision Language Model enhanced chat server for SFC syllabus search",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ChatRequest(BaseModel):
    user_input: str = Field(..., description="User query text")
    include_search_info: bool = Field(default=False, description="Include search details")

class MultimodalChatRequest(BaseModel):
    user_input: str = Field(..., description="User query text")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded image")
    include_search_info: bool = Field(default=False, description="Include search details")

class VLMModelRequest(BaseModel):
    model_key: str = Field(..., description="VLM model key to switch to")

# Response models
class ChatResponse(BaseModel):
    response: str
    search_results: Optional[List[Dict]] = None
    model_info: Optional[Dict] = None

async def process_chat_request(user_input: str, image_data: Optional[bytes] = None) -> Dict[str, Any]:
    """Process chat request with optional image."""
    loop = asyncio.get_event_loop()
    
    try:
        if image_data:
            # Multimodal processing with VLM
            logger.info("Processing multimodal query with VLM")
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Enhanced prompt for syllabus analysis
            vlm_prompt = f"""
            ユーザーの質問: {user_input}
            
            提供された画像を分析し、SFCのシラバス情報として以下を考慮して回答してください：
            1. 画像に含まれる授業情報
            2. ユーザーの質問に対する具体的な回答
            3. 関連する他の授業の推薦
            
            日本語で詳細に回答してください。
            """
            
            response = await loop.run_in_executor(
                executor,
                process_multimodal_query,
                vlm_prompt,
                image
            )
            
            return {
                "response": response,
                "model_info": {"type": "VLM", "multimodal": True}
            }
        
        else:
            # Text-only processing with RAG
            logger.info("Processing text-only query")
            
            # Check if RAG is needed
            use_rag = await loop.run_in_executor(executor, should_use_rag, user_input)
            
            if use_rag:
                # Perform RAG search
                search_results = await loop.run_in_executor(
                    executor, search_syllabus, user_input
                )
                
                # Create RAG prompt
                rag_prompt = create_rag_prompt(user_input, search_results)
                
                # Generate response
                response = await loop.run_in_executor(
                    executor, generate_response, rag_prompt
                )
                
                return {
                    "response": response,
                    "search_results": search_results,
                    "model_info": {"type": "LLM", "rag_used": True}
                }
            else:
                # Direct LLM response
                response = await loop.run_in_executor(
                    executor, generate_response, user_input
                )
                
                return {
                    "response": response,
                    "model_info": {"type": "LLM", "rag_used": False}
                }
                
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with server info."""
    vlm_status = "Available" if VLM_AVAILABLE and get_vlm_engine() else "Not Available"
    
    return {
        "message": "SFC-LLM VLM Chat Server",
        "version": "2.0.0",
        "vlm_status": vlm_status,
        "available_models": list_available_vlm_models() if VLM_AVAILABLE else [],
        "endpoints": {
            "chat": "/chat",
            "multimodal_chat": "/multimodal-chat", 
            "upload_chat": "/upload-chat",
            "vlm_models": "/vlm/models",
            "switch_model": "/vlm/switch-model"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, _: bool = Depends(verify_api_key)):
    """Standard text chat endpoint."""
    result = await process_chat_request(request.user_input)
    
    return ChatResponse(
        response=result["response"],
        search_results=result.get("search_results") if request.include_search_info else None,
        model_info=result.get("model_info")
    )

@app.post("/multimodal-chat", response_model=ChatResponse)
async def multimodal_chat(request: MultimodalChatRequest, _: bool = Depends(verify_api_key)):
    """Multimodal chat with base64 image input."""
    if not VLM_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="VLM functionality not available. Please install required dependencies."
        )
    
    image_data = None
    if request.image_base64:
        try:
            import base64
            # Remove data URL prefix if present
            if request.image_base64.startswith("data:image"):
                request.image_base64 = request.image_base64.split(",")[1]
            
            image_data = base64.b64decode(request.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
    
    result = await process_chat_request(request.user_input, image_data)
    
    return ChatResponse(
        response=result["response"],
        search_results=result.get("search_results") if request.include_search_info else None,
        model_info=result.get("model_info")
    )

@app.post("/upload-chat")
async def upload_chat(
    user_input: str = Form(...),
    image: Optional[UploadFile] = File(default=None),
    include_search_info: bool = Form(default=False),
    _: bool = Depends(verify_api_key)
):
    """Chat with file upload."""
    if not VLM_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="VLM functionality not available"
        )
    
    image_data = None
    if image:
        # Validate image file
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await image.read()
    
    result = await process_chat_request(user_input, image_data)
    
    return JSONResponse({
        "response": result["response"],
        "search_results": result.get("search_results") if include_search_info else None,
        "model_info": result.get("model_info")
    })

@app.get("/vlm/models")
async def get_vlm_models(_: bool = Depends(verify_api_key)):
    """Get available VLM models."""
    if not VLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="VLM not available")
    
    return {
        "available_models": list_available_vlm_models(),
        "current_model": get_vlm_engine().model_name if get_vlm_engine() else None
    }

@app.post("/vlm/switch-model")
async def switch_vlm_model_endpoint(
    request: VLMModelRequest, 
    _: bool = Depends(verify_api_key)
):
    """Switch VLM model."""
    if not VLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="VLM not available")
    
    success = switch_vlm_model(request.model_key)
    
    if success:
        return {"message": f"Successfully switched to {request.model_key}"}
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to switch to {request.model_key}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    vlm_engine = get_vlm_engine()
    
    return {
        "status": "healthy",
        "vlm_available": VLM_AVAILABLE,
        "vlm_loaded": vlm_engine is not None and vlm_engine.model is not None,
        "current_model": vlm_engine.model_name if vlm_engine else None
    }

if __name__ == "__main__":
    logger.info(f"Starting VLM-enhanced server on {API_HOST}:{API_PORT}")
    logger.info(f"Authentication: {'Enabled' if API_REQUIRE_AUTH else 'Disabled'}")
    logger.info(f"VLM Support: {'Available' if VLM_AVAILABLE else 'Not Available'}")
    
    uvicorn.run(
        "src.vlm_chat_server:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level=LOG_LEVEL.lower()
    )