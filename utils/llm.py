"""
LLM utility functions for text generation.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import pathlib
import logging
from functools import lru_cache
from typing import Optional, Union
from src.config import (
    HF_MODEL_ID, 
    LOCAL_MODEL_DIR, 
    HF_TOKEN, 
    MAX_NEW_TOKENS, 
    TEMPERATURE, 
    TOP_P
)

# Check if vLLM should be used
USE_VLLM = os.environ.get("SFC_LLM_USE_VLLM", "false").lower() == "true"

if USE_VLLM:
    try:
        from utils.vllm_engine import generate_response_vllm, VLLM_AVAILABLE
        if VLLM_AVAILABLE:
            logger.info("vLLM engine will be used for inference")
        else:
            logger.warning("vLLM requested but not available, falling back to transformers")
            USE_VLLM = False
    except ImportError:
        logger.warning("vLLM import failed, falling back to transformers")
        USE_VLLM = False

logger = logging.getLogger(__name__)

def ensure_model_downloaded() -> None:
    """Ensure the model is downloaded to the local directory."""
    try:
        # Check if model files exist
        weight_files = [
            "pytorch_model.bin",
            "model.safetensors", 
            "tf_model.h5",
            "model.ckpt.index",
            "flax_model.msgpack"
        ]
        
        model_path = pathlib.Path(LOCAL_MODEL_DIR)
        if not model_path.exists():
            model_path.mkdir(parents=True, exist_ok=True)
        
        if not any((model_path / f).exists() for f in weight_files):
            logger.info(f"Model not found. Downloading {HF_MODEL_ID}...")
            
            if not HF_TOKEN:
                raise ValueError("HUGGINGFACE_TOKEN environment variable is required")
            
            snapshot_download(
                repo_id=HF_MODEL_ID,
                local_dir=LOCAL_MODEL_DIR,
                token=HF_TOKEN,
                local_dir_use_symlinks=False,
                ignore_patterns=[]
            )
            logger.info(f"Model downloaded to {LOCAL_MODEL_DIR}")
        else:
            logger.info(f"Using existing model at {LOCAL_MODEL_DIR}")
            
    except Exception as e:
        logger.error(f"Failed to ensure model is downloaded: {str(e)}")
        raise

@lru_cache(maxsize=1)
def get_tokenizer() -> AutoTokenizer:
    """Get the tokenizer (cached)."""
    try:
        ensure_model_downloaded()
        logger.info(f"Loading tokenizer from {LOCAL_MODEL_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_DIR,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {str(e)}")
        raise

@lru_cache(maxsize=1)  
def get_model() -> AutoModelForCausalLM:
    """Get the model (cached)."""
    try:
        ensure_model_downloaded()
        logger.info(f"Loading model from {LOCAL_MODEL_DIR}")
        
        # Check if CUDA is available
        device_available = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device_available}")
        
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_DIR,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            load_in_8bit=torch.cuda.is_available(),  # Only use 8-bit on GPU
            use_cache=True,
            trust_remote_code=True,
            token=HF_TOKEN,
            low_cpu_mem_usage=True  # Reduce CPU memory usage during loading
        )
        model.eval()
        
        # PyTorch 2.0 compile optimization if available
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            logger.info("Applying PyTorch 2.0 compile optimization")
            model = torch.compile(model, mode="reduce-overhead")
        
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def generate_response(prompt: str, 
                     max_new_tokens: int = MAX_NEW_TOKENS, 
                     temperature: float = TEMPERATURE, 
                     top_p: float = TOP_P) -> str:
    """
    Generate a response from the LLM.
    
    Args:
        prompt: The input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        
    Returns:
        The generated response text
        
    Raises:
        ValueError: If prompt is empty or parameters are invalid
        RuntimeError: If model generation fails
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0.0 and 2.0")
    
    if not 0.0 <= top_p <= 1.0:
        raise ValueError("top_p must be between 0.0 and 1.0")
    
    # Use vLLM if available and enabled
    if USE_VLLM:
        try:
            return generate_response_vllm(
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
        except Exception as e:
            logger.warning(f"vLLM generation failed: {e}, falling back to transformers")
    
    # Fallback to transformers
    try:
        tokenizer = get_tokenizer()
        model = get_model()
        
        # Tokenize input with proper error handling
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=2048  # Prevent extremely long inputs
        )
        
        # Move to device
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        logger.debug(f"Generating response for prompt: {prompt[:100]}...")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.replace(prompt, "").strip()
        
        if not response:
            logger.warning("Empty response generated")
            return "申し訳ございませんが、適切な回答を生成できませんでした。"
        
        logger.debug(f"Generated response: {response[:100]}...")
        return response
        
    except Exception as e:
        logger.error(f"Failed to generate response: {str(e)}")
        raise RuntimeError(f"Response generation failed: {str(e)}")
