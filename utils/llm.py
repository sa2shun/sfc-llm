"""
LLM utility functions for text generation.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import pathlib
import logging
from functools import lru_cache
from typing import Optional
from src.config import (
    HF_MODEL_ID, 
    LOCAL_MODEL_DIR, 
    HF_TOKEN, 
    MAX_NEW_TOKENS, 
    TEMPERATURE, 
    TOP_P
)

logger = logging.getLogger(__name__)

def ensure_model_downloaded() -> None:
    """Ensure the model is downloaded to the local directory."""
    # Check if model files exist
    weight_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "tf_model.h5",
        "model.ckpt.index",
        "flax_model.msgpack"
    ]
    
    if not any((pathlib.Path(LOCAL_MODEL_DIR) / f).exists() for f in weight_files):
        logger.info(f"Model not found. Downloading {HF_MODEL_ID}...")
        snapshot_download(
            repo_id=HF_MODEL_ID,
            local_dir=LOCAL_MODEL_DIR,
            token=HF_TOKEN,
            local_dir_use_symlinks=False,
            ignore_patterns=[]  # Important: Don't skip any files
        )
        logger.info(f"Model downloaded to {LOCAL_MODEL_DIR}")
    else:
        logger.info(f"Using existing model at {LOCAL_MODEL_DIR}")

@lru_cache(maxsize=1)
def get_tokenizer() -> AutoTokenizer:
    """Get the tokenizer (cached)."""
    ensure_model_downloaded()
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_DIR,
        token=HF_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

@lru_cache(maxsize=1)
def get_model() -> AutoModelForCausalLM:
    """Get the model (cached)."""
    ensure_model_downloaded()
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        device_map="auto",
        torch_dtype="auto",
        token=HF_TOKEN
    )
    model.eval()
    return model

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
    """
    tokenizer = get_tokenizer()
    model = get_model()
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    response = full_response.replace(prompt, "").strip()
    
    return response
