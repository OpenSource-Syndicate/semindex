"""
Ollama LLM integration for semindex.

This module provides an interface to use Ollama for local GPU/CPU accelerated
language model inference. Ollama allows running various open-source models
locally with GPU acceleration when available.
"""

import os
import json
from typing import Iterable, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class OllamaError(RuntimeError):
    """Raised when an Ollama API call fails."""


class OllamaLLM:
    """
    Interface for communicating with local Ollama service.
    
    Example:
        # Using default settings (localhost:11434, llama3 model)
        llm = OllamaLLM()
        response = llm.generate("You are a helpful coding assistant.", "What is Python?")
        
        # Using specific model and server
        llm = OllamaLLM(model="codellama:7b", base_url="http://localhost:11434")
        response = llm.generate("System prompt", "User prompt")
    """

    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
        timeout: int = 120,
        **kwargs
    ) -> None:
        """
        Initialize the Ollama LLM client.
        
        Args:
            model: The specific Ollama model to use (e.g., "llama3", "codellama:7b", etc.)
            base_url: URL for Ollama API (default: http://localhost:11434)
            temperature: Sampling temperature (default: 0.2)
            top_p: Top-p sampling parameter (default: 0.9)
            max_tokens: Maximum tokens for response (default: 512)
            timeout: Request timeout in seconds (default: 120)
            **kwargs: Additional parameters to pass to the Ollama API
        """
        self.base_url = base_url or os.environ.get("SEMINDEX_OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.environ.get("SEMINDEX_OLLAMA_MODEL", "llama3")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.options = kwargs
        self.session = requests.Session()
        
        # Add retry strategy for robustness
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Verify Ollama is running
        if not self._check_connection():
            raise OllamaError(f"Cannot connect to Ollama at {self.base_url}. "
                             "Make sure Ollama is running and accessible.")

    def _check_connection(self) -> bool:
        """Check if Ollama service is accessible."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def build_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        context_chunks: Optional[Iterable[str]] = None,
    ) -> List[dict]:
        """
        Build a messages list for Ollama's chat API.
        
        Args:
            system_prompt: System/instruction prompt
            user_prompt: User query
            context_chunks: Optional context chunks to include
            
        Returns:
            List of message dictionaries for Ollama API
        """
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt.strip()
            })
        
        # Add context chunks if provided
        if context_chunks:
            ctx = "\n\n".join(chunk.strip() for chunk in context_chunks if chunk and chunk.strip())
            if ctx:
                messages.append({
                    "role": "system",
                    "content": f"Context:\n{ctx}"
                })
        
        # Add user message
        messages.append({
            "role": "user",
            "content": user_prompt.strip()
        })
        
        return messages

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        context_chunks: Optional[Iterable[str]] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a response from the Ollama model with provided context.
        
        Args:
            system_prompt: System/instruction prompt
            user_prompt: User query
            context_chunks: Optional context chunks to include
            max_tokens: Override default max tokens
            stop: Stop sequences
            
        Returns:
            Generated text response
        """
        messages = self.build_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context_chunks=context_chunks
        )
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                **self.options
            }
        }
        
        # Override max tokens if specified
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        else:
            payload["options"]["num_predict"] = self.max_tokens
            
        if stop:
            payload["options"]["stop"] = stop

        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise OllamaError(f"Ollama API returned status {response.status_code}: {response.text}")
            
            result = response.json()
            if "message" not in result or "content" not in result["message"]:
                raise OllamaError(f"Unexpected response format from Ollama: {result}")
                
            return result["message"]["content"].strip()
        
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"Request to Ollama failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise OllamaError(f"Invalid JSON response from Ollama: {str(e)}")
        except Exception as e:
            raise OllamaError(f"Error generating response with Ollama: {str(e)}")


def get_available_models(base_url: str = "http://localhost:11434") -> List[dict]:
    """
    Get list of available models from Ollama server.
    
    Args:
        base_url: URL for Ollama API
        
    Returns:
        List of available models
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("models", [])
        else:
            return []
    except Exception:
        return []


def pull_model(model: str, base_url: str = "http://localhost:11434") -> bool:
    """
    Pull a model from Ollama registry.
    
    Args:
        model: Name of model to pull (e.g., "llama3", "codellama:7b")
        base_url: URL for Ollama API
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.post(
            f"{base_url}/api/pull",
            json={"name": model},
            timeout=300  # Longer timeout for model downloads
        )
        return response.status_code == 200
    except Exception:
        return False