"""
Transformer-based LLM module for semindex
Supports lightweight transformer models for local inference
"""
import os
import warnings
from typing import Iterable, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from .config import get_config


class TransformerLLM:
    """
    Wrapper around HuggingFace transformers for local CPU inference.
    
    Example:
        llm = TransformerLLM(model_name="microsoft/phi-2")  # or other lightweight model
        text = llm.generate("You are a helpful assistant.", "Explain FAISS in two sentences.")
        
    The model_name parameter will default to the value from configuration (MODELS.CODE_LLM_MODEL),
    then to SEMINDEX_TRANSFORMER_MODEL environment variable, and finally to "microsoft/Phi-3-mini-4k-instruct".
    """
    
    def __init__(
        self,
        model_name: str = None,
        n_ctx: int = 2048,
        temperature: float = 0.2,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        device: str = None,
        torch_dtype: torch.dtype = None,
    ) -> None:
        # Use configuration to get default model, fallback to environment variable or default
        if model_name is None:
            config = get_config()
            model_from_config = config.get("MODELS.CODE_LLM_MODEL")
            if model_from_config and model_from_config.strip():
                model_name = model_from_config
            else:
                model_name = os.environ.get("SEMINDEX_TRANSFORMER_MODEL", "microsoft/Phi-3-mini-4k-instruct")  # Recommended model
        # Set device (default to CPU for compatibility)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set dtype (default to float32 for CPU compatibility)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self.device == "cpu" else torch.float16
        
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.n_ctx = n_ctx
        
        # Suppress warnings during model loading
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with appropriate parameters to avoid accelerate conflicts
        # Don't specify device_map if it might conflict with accelerate
        try:
            # First try loading without device_map to avoid accelerate conflicts
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
        except Exception:
            # If that fails, attempt with device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device != "cpu" else None,
                trust_remote_code=True
            )
        
        # Create text generation pipeline
        # For CPU, always use device -1; for GPU, use device 0 or appropriate GPU index
        pipeline_device = -1 if self.device == "cpu" else (0 if self.device == "cuda" else self.device)
        
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=pipeline_device,
            max_new_tokens=min(512, n_ctx // 2),  # Generate at most half the context
        )

    def build_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        context_chunks: Optional[Iterable[str]] = None,
    ) -> str:
        """Build a prompt suitable for transformer models."""
        ctx = "\n\n".join(chunk.strip() for chunk in (context_chunks or []) if chunk and chunk.strip())
        parts: List[str] = []
        
        # Add system prompt if provided
        if system_prompt:
            parts.append(f"### System:\n{system_prompt.strip()}\n")
        
        # Add context if provided
        if ctx:
            parts.append(f"### Context:\n{ctx}\n")
        
        # Add user prompt
        parts.append(f"### User:\n{user_prompt.strip()}\n")
        parts.append("### Assistant:\n")
        
        return "".join(parts)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        context_chunks: Optional[Iterable[str]] = None,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> str:
        prompt = self.build_prompt(system_prompt, user_prompt, context_chunks)
        
        # Set max_new_tokens based on the requested max_tokens, but cap it appropriately
        max_new_tokens = min(max_tokens, self.n_ctx // 2)
        
        # Generate text
        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # Stop sequences are handled differently in transformers pipelines
        )
        
        # Extract the generated text
        generated_text = outputs[0]["generated_text"]
        
        # Remove the prompt from the generated text to return only the model's response
        response = generated_text[len(prompt):]
        
        # Apply basic post-processing to clean up the response
        if stop:
            for stop_seq in stop:
                if stop_seq in response:
                    response = response.split(stop_seq)[0]
        
        return response.strip()


# Define some recommended lightweight models
LIGHTWEIGHT_MODELS = {
    "phi-2": "microsoft/phi-2",
    "phi-1_5": "microsoft/phi-1_5", 
    "stable-code-3b": "stabilityai/stable-code-3b",
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mpt-7b": "mosaicml/mpt-7b-instruct",  # Larger but still efficient
    "falcon-7b": "tiiuae/falcon-7b-instruct",  # Another efficient option
}


def get_recommended_model(model_type: str = "phi-2") -> str:
    """Get a recommended lightweight model name."""
    return LIGHTWEIGHT_MODELS.get(model_type, LIGHTWEIGHT_MODELS["phi-2"])