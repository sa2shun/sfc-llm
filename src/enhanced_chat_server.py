"""
Enhanced FastAPI server with advanced features for SFC-LLM.
"""
from fastapi import FastAPI, HTTPException, Request, Depends, Security, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import logging
import time
import asyncio
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any
from datetime import datetime

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
from src.config import API_HOST, API_PORT, API_PASSWORD, API_REQUIRE_AUTH, LOG_LEVEL, LOG_FORMAT
from utils.multimodal import get_multimodal_processor, process_uploaded_file
from utils.course_planner import (
    get_course_planner, 
    create_student_profile_from_query,
    StudentProfile,
    Semester,
    Grade
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=6)

# Enhanced FastAPI app
app = FastAPI(
    title="SFC-LLM Enhanced API",
    description="Advanced API for SFC syllabus search with multimodal and planning features",
    version="2.0.0"
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatRequest(BaseModel):
    user_input: str
    context: Optional[str] = None
    use_multimodal: bool = False
    student_profile: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    reply: str
    rag_used: bool
    processing_time: float
    rag_context: Optional[str] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    updated_profile: Optional[Dict[str, Any]] = None

class PlanningRequest(BaseModel):
    student_profile: Dict[str, Any]
    semester: str  # "春学期" or "秋学期"
    year: int
    target_credits: int = 20

class PlanningResponse(BaseModel):
    semester_plan: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    analysis: Dict[str, Any]

class FileUploadResponse(BaseModel):
    filename: str
    file_type: str
    processed_content: Dict[str, Any]
    success: bool
    message: str

def should_use_rag_heuristic(user_input: str) -> bool:
    """Quick heuristic to determine if RAG should be used."""
    query_lower = user_input.lower()
    
    # Japanese keywords that likely need SFC syllabus data
    sfc_keywords = [
        "授業", "科目", "講義", "セミナー", "演習", "ゼミ", "単位", "履修",
        "プログラミング", "データサイエンス", "AI", "機械学習", "統計",
        "経済", "政策", "メディア", "デザイン", "環境", "バイオ",
        "英語", "中国語", "韓国語", "language", "english",
        "学部", "大学院", "faculty", "graduate",
        "春学期", "秋学期", "semester", "開講",
        "sfc", "湘南藤沢", "慶應", "履修計画", "卒業"
    ]
    
    if any(keyword in query_lower for keyword in sfc_keywords):
        return True
    
    # General greetings that don't need syllabus data
    general_keywords = [
        "こんにちは", "hello", "はじめまして", "ありがとう", "thank you",
        "今日", "明日", "天気", "weather", "時間", "time"
    ]
    
    if any(keyword in query_lower for keyword in general_keywords):
        return False
    
    return True

def _process_enhanced_chat(user_input: str, 
                          context: Optional[str] = None,
                          student_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Enhanced chat processing with planning integration."""
    
    # Create or update student profile
    if student_profile:
        profile = StudentProfile(**student_profile)
    else:
        profile = create_student_profile_from_query(user_input)
    
    # Check if this is a planning-related query
    planning_keywords = ["履修計画", "おすすめ", "recommendation", "プラン", "plan", "科目選択"]
    is_planning_query = any(keyword in user_input.lower() for keyword in planning_keywords)
    
    result = {
        "reply": "",
        "rag_used": False,
        "rag_context": None,
        "recommendations": [],
        "updated_profile": profile.__dict__ if hasattr(profile, '__dict__') else {}
    }
    
    if is_planning_query:
        # Planning-specific response
        try:
            planner = get_course_planner()
            
            # Get current semester
            current_month = datetime.now().month
            current_year = datetime.now().year
            semester = Semester.SPRING if current_month <= 6 else Semester.FALL
            
            # Generate recommendations
            recommendations = planner.recommend_courses(profile, semester, current_year)
            result["recommendations"] = [
                {
                    "course_name": rec["course"].subject_name,
                    "credits": rec["course"].credits,
                    "category": rec["course"].category,
                    "score": rec["score"],
                    "reasons": rec["reasons"]
                }
                for rec in recommendations[:5]  # Top 5
            ]
            
            # Create personalized response
            if recommendations:
                course_list = "\n".join([
                    f"・{rec['course_name']} ({rec['credits']}単位) - {', '.join(rec['reasons'])}"
                    for rec in recommendations[:3]
                ])
                
                result["reply"] = f"""あなたのプロファイルに基づいて、以下の科目をお勧めします：

{course_list}

これらの科目は、あなたの興味分野「{', '.join(profile.interests)}」や進路目標「{', '.join(profile.career_goals)}」に適合しています。

詳細な履修計画や他の科目についても相談できますので、お気軽にお聞きください。"""
            else:
                result["reply"] = "申し訳ございませんが、現在のプロファイルに基づく適切な科目推薦が見つかりませんでした。より詳細な情報をお聞かせください。"
            
            result["rag_used"] = True
            
        except Exception as e:
            logger.error(f"Planning processing failed: {e}")
            result["reply"] = "履修計画の処理中にエラーが発生しました。"
    
    else:
        # Standard RAG processing
        rag_needed_heuristic = should_use_rag_heuristic(user_input)
        
        if not rag_needed_heuristic:
            final_prompt = get_general_response_prompt(user_input)
            result["reply"] = generate_response(final_prompt)
            result["rag_used"] = False
        else:
            # Use RAG decision LLM for ambiguous cases
            decision_prompt = get_rag_decision_prompt(user_input)
            decision = generate_response(decision_prompt)
            rag_used = "NEED_RAG" in decision
            
            if rag_used:
                # Get relevant documents
                hits = search_syllabus(user_input)
                
                if hits:
                    # Format context
                    context_items = []
                    for h in hits:
                        subject = h.get('subject_name', '[No Subject]')
                        category = h.get('category', '')
                        credits = h.get('credits', '')
                        language = h.get('language', '')
                        
                        metadata = []
                        if category:
                            metadata.append(f"分野: {category}")
                        if credits:
                            metadata.append(f"{credits}単位")
                        if language:
                            metadata.append(f"言語: {language}")
                        
                        context_items.append(f"【{subject}】({', '.join(metadata)})")
                    
                    rag_context = "\n".join(context_items)
                    
                    # Add additional context if provided
                    if context:
                        rag_context = f"{context}\n\n関連科目:\n{rag_context}"
                    
                    final_prompt = get_rag_response_prompt(user_input, rag_context)
                    result["reply"] = generate_response(final_prompt)
                    result["rag_used"] = True
                    result["rag_context"] = rag_context
                else:
                    final_prompt = get_general_response_prompt(user_input)
                    result["reply"] = generate_response(final_prompt)
                    result["rag_used"] = False
            else:
                final_prompt = get_general_response_prompt(user_input)
                result["reply"] = generate_response(final_prompt)
                result["rag_used"] = False
    
    return result

@app.post("/chat", response_model=ChatResponse)
async def enhanced_chat(req: ChatRequest, authenticated: bool = Depends(get_api_key)):
    """Enhanced chat endpoint with planning integration."""
    start_time = time.time()
    
    if not req.user_input or not req.user_input.strip():
        raise HTTPException(status_code=400, detail="User input cannot be empty")
    
    logger.info(f"Enhanced chat request: {req.user_input[:50]}...")
    
    try:
        # Process chat in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            _process_enhanced_chat, 
            req.user_input,
            req.context,
            req.student_profile
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Enhanced request processed in {processing_time:.2f} seconds")
        
        result["processing_time"] = processing_time
        return result
        
    except Exception as e:
        logger.error(f"Enhanced chat processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing request")

@app.post("/plan", response_model=PlanningResponse)
async def create_course_plan(req: PlanningRequest, authenticated: bool = Depends(get_api_key)):
    """Create a semester course plan."""
    try:
        profile = StudentProfile(**req.student_profile)
        planner = get_course_planner()
        
        # Convert semester string to enum
        semester = Semester.SPRING if req.semester == "春学期" else Semester.FALL
        
        # Create semester plan
        plan = planner.create_semester_plan(profile, semester, req.year, req.target_credits)
        
        # Get additional recommendations
        recommendations = planner.recommend_courses(profile, semester, req.year, 10)
        
        # Analyze student progress
        analysis = planner.analyze_student_progress(profile)
        
        return {
            "semester_plan": plan,
            "recommendations": [
                {
                    "course_name": rec["course"].subject_name,
                    "credits": rec["course"].credits,
                    "category": rec["course"].category,
                    "score": rec["score"],
                    "reasons": rec["reasons"]
                }
                for rec in recommendations
            ],
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Course planning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Planning error: {str(e)}")

@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...), authenticated: bool = Depends(get_api_key)):
    """Upload and process files (PDF syllabi, images)."""
    try:
        # Read file content
        file_content = await file.read()
        
        # Process file
        result = process_uploaded_file(file_content, file.filename, file.content_type)
        
        return {
            "filename": file.filename,
            "file_type": result["type"],
            "processed_content": result["content"],
            "success": result["processed"],
            "message": "File processed successfully" if result["processed"] else "File processing failed"
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.get("/health")
def health_check():
    """Enhanced health check."""
    return {
        "status": "ok",
        "version": "2.0.0",
        "features": {
            "vllm_available": os.environ.get("SFC_LLM_USE_VLLM", "false") == "true",
            "multimodal_enabled": True,
            "planning_enabled": True,
            "auth_required": API_REQUIRE_AUTH
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/student-profile/demo")
def get_demo_profile():
    """Get a demo student profile for testing."""
    return {
        "student_id": "demo_student",
        "grade": 2,
        "major": "総合政策",
        "completed_courses": ["基礎統計学", "プログラミング入門", "英語コミュニケーション"],
        "current_courses": ["データサイエンス概論"],
        "target_graduation_year": datetime.now().year + 2,
        "interests": ["データサイエンス", "AI", "政策分析"],
        "career_goals": ["データサイエンティスト", "政策アナリスト"],
        "language_preference": "japanese",
        "max_credits_per_semester": 22,
        "preferred_difficulty": 3
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.enhanced_chat_server:app", host=API_HOST, port=API_PORT, reload=True)