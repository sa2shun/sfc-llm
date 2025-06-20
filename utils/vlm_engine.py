"""
Vision Language Model (VLM) engine for multimodal syllabus processing.
Supports image + text input for enhanced SFC course search.
"""
import os
import logging
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

try:
    import torch
    from transformers import (
        LlavaNextProcessor, 
        LlavaNextForConditionalGeneration,
        Qwen2VLForConditionalGeneration, 
        AutoProcessor
    )
    from PIL import Image
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False

logger = logging.getLogger(__name__)

class VLMEngine:
    """Vision Language Model engine for multimodal processing."""
    
    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf"):
        """
        Initialize VLM engine.
        
        Args:
            model_name: HuggingFace model identifier for VLM
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not VLM_AVAILABLE:
            logger.error("VLM dependencies not available. Install: transformers, torch, pillow")
            return
            
        self._load_model()
    
    def _load_model(self):
        """Load VLM model and processor."""
        try:
            logger.info(f"Loading VLM model: {self.model_name}")
            
            if "llava" in self.model_name.lower():
                self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    load_in_8bit=True if self.device == "cuda" else False
                )
            elif "qwen2-vl" in self.model_name.lower():
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            
            logger.info(f"VLM model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load VLM model: {e}")
            self.model = None
            self.processor = None
    
    def process_image_and_text(
        self, 
        image: Union[str, Image.Image, bytes], 
        text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Process image and text together using VLM.
        
        Args:
            image: Image path, PIL Image, or bytes
            text: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        if not self.model or not self.processor:
            return "VLM model not available"
        
        try:
            # Process image input
            if isinstance(image, str):
                if image.startswith("data:image"):
                    # Base64 encoded image
                    image_data = base64.b64decode(image.split(",")[1])
                    image = Image.open(BytesIO(image_data))
                else:
                    # File path
                    image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            
            # Ensure RGB format
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Create conversation format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text}
                    ]
                }
            ]
            
            # Process inputs
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"VLM processing failed: {e}")
            return f"Error processing image and text: {e}"
    
    def analyze_syllabus_image(self, image: Union[str, Image.Image, bytes]) -> Dict[str, Any]:
        """
        Analyze syllabus image and extract course information.
        
        Args:
            image: Syllabus image
            
        Returns:
            Extracted course information
        """
        prompt = """
        このシラバス画像を分析して、以下の情報を抽出してください：

        1. 科目名
        2. 担当教員
        3. 授業概要
        4. 主題と目標
        5. 開講年度・学期
        6. 単位数
        7. 実施形態（対面/オンライン/ハイブリッド）
        8. 授業で使う言語

        JSON形式で回答してください。
        """
        
        response = self.process_image_and_text(image, prompt)
        
        try:
            import json
            # Try to parse as JSON
            if "{" in response and "}" in response:
                json_part = response[response.find("{"):response.rfind("}")+1]
                return json.loads(json_part)
        except:
            pass
        
        # Fallback: return raw text
        return {"raw_analysis": response}
    
    def compare_syllabi(self, image1: Union[str, Image.Image], image2: Union[str, Image.Image]) -> str:
        """
        Compare two syllabus images and highlight differences.
        
        Args:
            image1: First syllabus image
            image2: Second syllabus image
            
        Returns:
            Comparison analysis
        """
        # Analyze both images
        analysis1 = self.analyze_syllabus_image(image1)
        analysis2 = self.analyze_syllabus_image(image2)
        
        prompt = f"""
        以下の2つのシラバス分析結果を比較してください：

        シラバス1: {analysis1}
        シラバス2: {analysis2}

        違いや類似点を詳しく説明してください。
        """
        
        # Use text-only processing for comparison
        return self.process_image_and_text(image1, prompt)

# Global VLM engine instance
_vlm_engine = None

def get_vlm_engine(model_name: str = None) -> Optional[VLMEngine]:
    """Get global VLM engine instance."""
    global _vlm_engine
    
    if _vlm_engine is None:
        default_model = model_name or os.environ.get(
            "SFC_LLM_VLM_MODEL", 
            "llava-hf/llava-v1.6-mistral-7b-hf"
        )
        _vlm_engine = VLMEngine(default_model)
    
    return _vlm_engine if _vlm_engine.model else None

def process_multimodal_query(
    text: str, 
    image: Optional[Union[str, Image.Image, bytes]] = None,
    **kwargs
) -> str:
    """
    Process multimodal query with optional image.
    
    Args:
        text: Text query
        image: Optional image input
        **kwargs: Additional parameters
        
    Returns:
        Response text
    """
    vlm = get_vlm_engine()
    
    if image and vlm:
        return vlm.process_image_and_text(image, text, **kwargs)
    else:
        # Fallback to text-only processing
        from utils.llm import generate_response
        return generate_response(text)

# Configuration
VLM_MODELS = {
    "llava-mistral-7b": "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-vicuna-7b": "llava-hf/llava-v1.6-vicuna-7b-hf", 
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct"
}

def list_available_vlm_models() -> List[str]:
    """List available VLM models."""
    return list(VLM_MODELS.keys())

def switch_vlm_model(model_key: str) -> bool:
    """
    Switch to a different VLM model.
    
    Args:
        model_key: Model key from VLM_MODELS
        
    Returns:
        Success status
    """
    global _vlm_engine
    
    if model_key not in VLM_MODELS:
        logger.error(f"Unknown VLM model: {model_key}")
        return False
    
    try:
        _vlm_engine = VLMEngine(VLM_MODELS[model_key])
        return _vlm_engine.model is not None
    except Exception as e:
        logger.error(f"Failed to switch VLM model: {e}")
        return False