"""
vLLM-based high-performance inference engine for SFC-LLM.
"""
import logging
import os
from typing import List, Optional, Dict, Any
from functools import lru_cache

try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

from src.config import (
    LOCAL_MODEL_DIR,
    HF_TOKEN,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P
)

logger = logging.getLogger(__name__)

class VLLMEngine:
    """
    High-performance vLLM inference engine.
    
    Benefits:
    - 10-20x faster inference than standard transformers
    - Efficient GPU memory management with PagedAttention
    - Optimized batching and continuous batching
    - Better throughput for concurrent requests
    """
    
    def __init__(self, 
                 model_path: str = LOCAL_MODEL_DIR,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85,
                 max_model_len: Optional[int] = None):
        """
        Initialize vLLM engine.
        
        Args:
            model_path: Path to the model
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            max_model_len: Maximum sequence length
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Install with: pip install vllm")
        
        self.model_path = model_path
        self.llm = None
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        
        logger.info(f"Initializing vLLM engine with model: {model_path}")
        self._load_model()
    
    def _load_model(self):
        """Load the vLLM model."""
        try:
            # vLLM configuration
            llm_config = {
                "model": self.model_path,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "trust_remote_code": True,
                "dtype": "half",  # Use fp16 for better performance
                "download_dir": None,
                "enforce_eager": False,  # Use CUDA graphs for better performance
            }
            
            if self.max_model_len:
                llm_config["max_model_len"] = self.max_model_len
            
            # Add token if available
            if HF_TOKEN:
                os.environ["HF_TOKEN"] = HF_TOKEN
            
            self.llm = LLM(**llm_config)
            logger.info("vLLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load vLLM model: {str(e)}")
            raise
    
    def generate(self, 
                prompt: str,
                max_tokens: int = MAX_NEW_TOKENS,
                temperature: float = TEMPERATURE,
                top_p: float = TOP_P,
                **kwargs) -> str:
        """
        Generate text using vLLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional sampling parameters
            
        Returns:
            Generated text
        """
        if not self.llm:
            raise RuntimeError("vLLM model not loaded")
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=kwargs.get("stop_sequences"),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.1),
            top_k=kwargs.get("top_k", 50),
        )
        
        try:
            # Generate response
            outputs = self.llm.generate([prompt], sampling_params)
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0].outputs[0].text.strip()
                logger.debug(f"Generated response: {generated_text[:100]}...")
                return generated_text
            else:
                logger.warning("No output generated")
                return "申し訳ございませんが、適切な回答を生成できませんでした。"
                
        except Exception as e:
            logger.error(f"vLLM generation failed: {str(e)}")
            raise RuntimeError(f"Text generation failed: {str(e)}")
    
    def generate_batch(self, 
                      prompts: List[str],
                      max_tokens: int = MAX_NEW_TOKENS,
                      temperature: float = TEMPERATURE,
                      top_p: float = TOP_P,
                      **kwargs) -> List[str]:
        """
        Generate text for multiple prompts efficiently.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature  
            top_p: Nucleus sampling parameter
            **kwargs: Additional sampling parameters
            
        Returns:
            List of generated texts
        """
        if not self.llm:
            raise RuntimeError("vLLM model not loaded")
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=kwargs.get("stop_sequences"),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.1),
            top_k=kwargs.get("top_k", 50),
        )
        
        try:
            outputs = self.llm.generate(prompts, sampling_params)
            results = []
            
            for output in outputs:
                if output.outputs:
                    generated_text = output.outputs[0].text.strip()
                    results.append(generated_text)
                else:
                    results.append("申し訳ございませんが、適切な回答を生成できませんでした。")
            
            return results
            
        except Exception as e:
            logger.error(f"vLLM batch generation failed: {str(e)}")
            raise RuntimeError(f"Batch text generation failed: {str(e)}")
    
    def __del__(self):
        """Cleanup vLLM resources."""
        if hasattr(self, 'llm') and self.llm:
            try:
                destroy_model_parallel()
                logger.info("vLLM resources cleaned up")
            except Exception as e:
                logger.warning(f"Error during vLLM cleanup: {e}")

# Global vLLM engine instance
_vllm_engine: Optional[VLLMEngine] = None

@lru_cache(maxsize=1)
def get_vllm_engine() -> VLLMEngine:
    """Get cached vLLM engine instance."""
    global _vllm_engine
    
    if _vllm_engine is None:
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not available. Install with: pip install vllm\n"
                "Falling back to standard transformers inference."
            )
        
        # Auto-detect GPU configuration
        import torch
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # Configure based on available hardware
        if num_gpus >= 2:
            tensor_parallel_size = min(num_gpus, 4)  # Use up to 4 GPUs
            gpu_memory_utilization = 0.85
        elif num_gpus == 1:
            tensor_parallel_size = 1
            gpu_memory_utilization = 0.90
        else:
            raise RuntimeError("vLLM requires at least one GPU")
        
        logger.info(f"Initializing vLLM with {tensor_parallel_size} GPUs")
        
        _vllm_engine = VLLMEngine(
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization
        )
    
    return _vllm_engine

def generate_response_vllm(prompt: str, **kwargs) -> str:
    """
    Generate response using vLLM engine.
    
    Args:
        prompt: Input prompt
        **kwargs: Generation parameters
        
    Returns:
        Generated text
    """
    engine = get_vllm_engine()
    return engine.generate(prompt, **kwargs)

def generate_batch_vllm(prompts: List[str], **kwargs) -> List[str]:
    """
    Generate responses for multiple prompts using vLLM.
    
    Args:
        prompts: List of input prompts
        **kwargs: Generation parameters
        
    Returns:
        List of generated texts
    """
    engine = get_vllm_engine()
    return engine.generate_batch(prompts, **kwargs)