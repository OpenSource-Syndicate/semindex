"""
Model Manager for semindex
Provides caching and management for embedding and LLM models to avoid repeated loading
"""
import threading
from typing import Dict, Optional
from .embed import Embedder
from .local_llm import LocalLLM


class ModelManager:
    """Singleton class to manage and cache models for semindex"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._embedders: Dict[str, Embedder] = {}
            self._llms: Dict[str, LocalLLM] = {}
            self._lock = threading.Lock()
            self._initialized = True
    
    def get_embedder(self, model_name: str) -> Embedder:
        """Get a cached embedder instance or create a new one"""
        with self._lock:
            if model_name not in self._embedders:
                self._embedders[model_name] = Embedder(model_name=model_name)
            return self._embedders[model_name]
    
    def get_llm(self, model_type: str = "transformer", model_name: Optional[str] = None) -> LocalLLM:
        """Get a cached LLM instance or create a new one"""
        key = f"{model_type}_{model_name}" if model_name else model_type
        with self._lock:
            if key not in self._llms:
                self._llms[key] = LocalLLM(model_type=model_type, model_name=model_name)
            return self._llms[key]
    
    def clear_cache(self):
        """Clear all cached models"""
        with self._lock:
            self._embedders.clear()
            self._llms.clear()


# Global instance
model_manager = ModelManager()