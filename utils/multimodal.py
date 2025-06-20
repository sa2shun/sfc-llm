"""
Multimodal functionality for SFC-LLM: PDF processing and image analysis.
"""
import logging
import os
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False

logger = logging.getLogger(__name__)

class MultimodalProcessor:
    """Handles PDF, image, and other multimodal content processing."""
    
    def __init__(self):
        self.blip_processor = None
        self.blip_model = None
        self._init_image_models()
    
    def _init_image_models(self):
        """Initialize image processing models."""
        if BLIP_AVAILABLE:
            try:
                logger.info("Loading BLIP model for image captioning...")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                if torch.cuda.is_available():
                    self.blip_model = self.blip_model.to("cuda")
                logger.info("BLIP model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load BLIP model: {e}")
                self.blip_processor = None
                self.blip_model = None
    
    def extract_pdf_content(self, pdf_path: Union[str, Path, bytes]) -> Dict[str, Any]:
        """
        Extract content from PDF files (syllabus documents).
        
        Args:
            pdf_path: Path to PDF file or PDF bytes
            
        Returns:
            Dictionary containing extracted content
        """
        if not PDF_AVAILABLE:
            raise ImportError("pdfplumber not installed. Install with: pip install pdfplumber")
        
        result = {
            "text": "",
            "tables": [],
            "metadata": {},
            "pages": []
        }
        
        try:
            # Handle different input types
            if isinstance(pdf_path, bytes):
                pdf_file = BytesIO(pdf_path)
            else:
                pdf_file = pdf_path
            
            with pdfplumber.open(pdf_file) as pdf:
                result["metadata"] = {
                    "page_count": len(pdf.pages),
                    "title": pdf.metadata.get("Title", ""),
                    "author": pdf.metadata.get("Author", ""),
                    "subject": pdf.metadata.get("Subject", "")
                }
                
                full_text = []
                
                for i, page in enumerate(pdf.pages):
                    page_content = {
                        "page_number": i + 1,
                        "text": "",
                        "tables": []
                    }
                    
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        page_content["text"] = page_text.strip()
                        full_text.append(page_text)
                    
                    # Extract tables
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            if table:  # Skip empty tables
                                page_content["tables"].append(table)
                                result["tables"].append({
                                    "page": i + 1,
                                    "data": table
                                })
                    
                    result["pages"].append(page_content)
                
                result["text"] = "\n\n".join(full_text)
                
                logger.info(f"Extracted {len(result['pages'])} pages from PDF")
                return result
                
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise RuntimeError(f"Failed to process PDF: {str(e)}")
    
    def analyze_image(self, image_path: Union[str, Path, bytes, Image.Image]) -> Dict[str, Any]:
        """
        Analyze images (course materials, diagrams, etc.).
        
        Args:
            image_path: Path to image, image bytes, or PIL Image
            
        Returns:
            Dictionary containing analysis results
        """
        result = {
            "caption": "",
            "ocr_text": "",
            "analysis": "",
            "metadata": {}
        }
        
        try:
            # Load image
            if isinstance(image_path, Image.Image):
                image = image_path
            elif isinstance(image_path, bytes):
                image = Image.open(BytesIO(image_path))
            else:
                image = Image.open(image_path)
            
            result["metadata"] = {
                "size": image.size,
                "mode": image.mode,
                "format": getattr(image, 'format', 'Unknown')
            }
            
            # Generate caption using BLIP
            if BLIP_AVAILABLE and self.blip_model:
                try:
                    inputs = self.blip_processor(image, return_tensors="pt")
                    if torch.cuda.is_available():
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        out = self.blip_model.generate(**inputs, max_length=50)
                    
                    caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
                    result["caption"] = caption
                    logger.debug(f"Generated image caption: {caption}")
                    
                except Exception as e:
                    logger.warning(f"Image captioning failed: {e}")
            
            # Perform OCR
            if OCR_AVAILABLE:
                try:
                    ocr_text = pytesseract.image_to_string(image, lang='eng+jpn')
                    result["ocr_text"] = ocr_text.strip()
                    logger.debug(f"Extracted OCR text: {ocr_text[:100]}...")
                    
                except Exception as e:
                    logger.warning(f"OCR failed: {e}")
            
            # Combine analysis
            analysis_parts = []
            if result["caption"]:
                analysis_parts.append(f"画像の説明: {result['caption']}")
            if result["ocr_text"]:
                analysis_parts.append(f"抽出されたテキスト: {result['ocr_text'][:200]}...")
            
            result["analysis"] = "\n".join(analysis_parts)
            
            return result
            
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            raise RuntimeError(f"Failed to analyze image: {str(e)}")
    
    def process_syllabus_pdf(self, pdf_path: Union[str, Path, bytes]) -> Dict[str, Any]:
        """
        Specialized processing for SFC syllabus PDFs.
        
        Args:
            pdf_path: Path to PDF file or PDF bytes
            
        Returns:
            Structured syllabus information
        """
        pdf_content = self.extract_pdf_content(pdf_path)
        
        # Extract structured information from syllabus
        syllabus_info = {
            "course_title": "",
            "instructor": "",
            "credits": "",
            "semester": "",
            "objectives": "",
            "schedule": "",
            "evaluation": "",
            "textbooks": "",
            "raw_text": pdf_content["text"],
            "tables": pdf_content["tables"]
        }
        
        # Parse syllabus text using patterns
        text = pdf_content["text"]
        
        # Common syllabus patterns (Japanese)
        patterns = {
            "course_title": [r"科目名[：:]\s*(.+)", r"授業名[：:]\s*(.+)", r"Course Title[：:]\s*(.+)"],
            "instructor": [r"担当教員[：:]\s*(.+)", r"教員[：:]\s*(.+)", r"Instructor[：:]\s*(.+)"],
            "credits": [r"単位数[：:]\s*(.+)", r"Credits[：:]\s*(.+)"],
            "semester": [r"学期[：:]\s*(.+)", r"開講期[：:]\s*(.+)", r"Semester[：:]\s*(.+)"],
            "objectives": [r"授業の概要[：:]\s*(.+?)(?=\n\n|\n[A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+[：:])", r"Course Objectives[：:]\s*(.+?)(?=\n\n|\n[A-Z])"],
            "evaluation": [r"成績評価[：:]\s*(.+?)(?=\n\n|\n[A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+[：:])", r"Evaluation[：:]\s*(.+?)(?=\n\n|\n[A-Z])"]
        }
        
        import re
        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    syllabus_info[field] = match.group(1).strip()
                    break
        
        logger.info(f"Processed syllabus PDF: {syllabus_info['course_title']}")
        return syllabus_info
    
    def create_multimodal_context(self, 
                                 text_content: str = "",
                                 pdf_content: Optional[Dict] = None,
                                 image_content: Optional[Dict] = None) -> str:
        """
        Create unified context from multiple modalities.
        
        Args:
            text_content: Regular text content
            pdf_content: Processed PDF content
            image_content: Analyzed image content
            
        Returns:
            Combined multimodal context string
        """
        context_parts = []
        
        if text_content:
            context_parts.append(f"テキスト情報:\n{text_content}")
        
        if pdf_content:
            context_parts.append("PDFから抽出された情報:")
            if pdf_content.get("course_title"):
                context_parts.append(f"科目名: {pdf_content['course_title']}")
            if pdf_content.get("instructor"):
                context_parts.append(f"担当教員: {pdf_content['instructor']}")
            if pdf_content.get("objectives"):
                context_parts.append(f"授業概要: {pdf_content['objectives']}")
            if pdf_content.get("raw_text"):
                context_parts.append(f"詳細内容: {pdf_content['raw_text'][:500]}...")
        
        if image_content:
            context_parts.append("画像から抽出された情報:")
            if image_content.get("caption"):
                context_parts.append(f"画像の説明: {image_content['caption']}")
            if image_content.get("ocr_text"):
                context_parts.append(f"画像内テキスト: {image_content['ocr_text'][:200]}...")
        
        return "\n\n".join(context_parts)

# Global processor instance
_multimodal_processor: Optional[MultimodalProcessor] = None

def get_multimodal_processor() -> MultimodalProcessor:
    """Get cached multimodal processor instance."""
    global _multimodal_processor
    
    if _multimodal_processor is None:
        _multimodal_processor = MultimodalProcessor()
    
    return _multimodal_processor

def process_uploaded_file(file_content: bytes, 
                         filename: str, 
                         content_type: str) -> Dict[str, Any]:
    """
    Process uploaded files (PDF, images) for SFC syllabus system.
    
    Args:
        file_content: Raw file bytes
        filename: Original filename
        content_type: MIME content type
        
    Returns:
        Processed file information
    """
    processor = get_multimodal_processor()
    
    try:
        if content_type.startswith('application/pdf') or filename.lower().endswith('.pdf'):
            # Process PDF
            return {
                "type": "pdf",
                "filename": filename,
                "content": processor.process_syllabus_pdf(file_content),
                "processed": True
            }
        
        elif content_type.startswith('image/'):
            # Process image
            return {
                "type": "image", 
                "filename": filename,
                "content": processor.analyze_image(file_content),
                "processed": True
            }
        
        else:
            logger.warning(f"Unsupported file type: {content_type}")
            return {
                "type": "unknown",
                "filename": filename,
                "content": {"error": "Unsupported file type"},
                "processed": False
            }
            
    except Exception as e:
        logger.error(f"File processing failed for {filename}: {str(e)}")
        return {
            "type": "error",
            "filename": filename,
            "content": {"error": str(e)},
            "processed": False
        }