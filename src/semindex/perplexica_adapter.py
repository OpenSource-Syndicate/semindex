"""
Perplexica Search Adapter for semindex
Provides integration with Perplexica's search API for external knowledge and documentation
"""
import json
import os
import requests
from typing import Dict, List, Optional, Any

from .config import get_config
from .model import Symbol


class PerplexicaSearchAdapter:
    """
    Adapter for calling Perplexica's search API
    API Reference: https://raw.githubusercontent.com/ItzCrazyKns/Perplexica/refs/heads/master/docs/API/SEARCH.md
    """
    
    def __init__(self, config_path: Optional[str] = None):
        config = get_config(config_path)
        perplexica_config = config.get_perplexica_config()
        
        self.api_base = perplexica_config["api_base"].rstrip('/')
        self.search_endpoint = f"{self.api_base}/api/search"
        self.chat_model = perplexica_config["chat_model"]
        self.embedding_model = perplexica_config["embedding_model"]
        
    def search(
        self,
        query: str,
        focus_mode: str = "webSearch",
        chat_model: Optional[Dict[str, Any]] = None,
        embedding_model: Optional[Dict[str, Any]] = None,
        optimization_mode: Optional[str] = "balanced",
        system_instructions: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Perform a search using Perplexica's API
        
        Args:
            query: The search query
            focus_mode: Search mode (webSearch, academicSearch, writingAssistant, wolframAlphaSearch, youtubeSearch, redditSearch)
            chat_model: Chat model configuration
            embedding_model: Embedding model configuration
            optimization_mode: Optimization mode (speed, balanced)
            system_instructions: Custom instructions to guide the AI
            history: Conversation history as message pairs
            stream: Whether to enable streaming responses
        
        Returns:
            Search results from Perplexica API
        """
        # Use configured models if not provided explicitly
        if chat_model is None:
            chat_model = self.chat_model
        if embedding_model is None:
            embedding_model = self.embedding_model
        
        # Prepare the request payload
        payload = {
            "query": query,
            "focusMode": focus_mode,
            "chatModel": chat_model,
            "embeddingModel": embedding_model,
            "stream": stream
        }
        
        if optimization_mode:
            payload["optimizationMode"] = optimization_mode
        if system_instructions:
            payload["systemInstructions"] = system_instructions
        if history:
            payload["history"] = history
        
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.search_endpoint,
                json=payload,
                headers=headers,
                stream=stream
            )
            
            if response.status_code == 200:
                if stream:
                    # For streaming, return the response object to be processed
                    return {"streaming_response": response}
                else:
                    # For non-streaming, return the JSON response
                    return response.json()
            else:
                return {
                    "error": f"API request failed with status {response.status_code}",
                    "details": response.text
                }
        except requests.exceptions.ConnectionError:
            return {
                "error": f"Could not connect to Perplexica API at {self.search_endpoint}. Please ensure Perplexica is running.",
                "details": "Connection refused"
            }
        except Exception as e:
            return {
                "error": f"An error occurred while calling Perplexica API: {str(e)}",
                "details": str(e)
            }
    
    def search_web(self, query: str, optimization_mode: str = "balanced") -> Dict[str, Any]:
        """Perform a web search using Perplexica API"""
        return self.search(
            query=query,
            focus_mode="webSearch",
            optimization_mode=optimization_mode
        )
    
    def search_academic(self, query: str) -> Dict[str, Any]:
        """Perform an academic search using Perplexica API"""
        return self.search(
            query=query,
            focus_mode="academicSearch"
        )
    
    def search_documentation(self, query: str, library_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform a documentation search for a specific library or general documentation
        """
        if library_name:
            query = f"documentation for {library_name}: {query}"
        
        return self.search_web(query, optimization_mode="balanced")
    
    def search_youtube(self, query: str) -> Dict[str, Any]:
        """Perform a YouTube search using Perplexica API"""
        return self.search(
            query=query,
            focus_mode="youtubeSearch"
        )
    
    def search_reddit(self, query: str) -> Dict[str, Any]:
        """Perform a Reddit search using Perplexica API"""
        return self.search(
            query=query,
            focus_mode="redditSearch"
        )
    
    def is_available(self) -> bool:
        """
        Check if the Perplexica API is available
        """
        try:
            # Try to make a simple request to check availability
            response = requests.get(f"{self.api_base}/api/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def search_with_fallback(
        self,
        query: str,
        focus_mode: str = "webSearch",
        fallback_to_local: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform search with fallback to local search if Perplexica is not available
        """
        if self.is_available():
            # Use Perplexica API
            return self.search(query, focus_mode, **kwargs)
        elif fallback_to_local:
            # Fallback to local search (this would typically be implemented by the caller)
            # Return a standardized response format
            return {
                "error": "Perplexica API not available, using local search only",
                "fallback_used": True,
                "query": query,
                "focus_mode": focus_mode
            }
        else:
            return {
                "error": "Perplexica API not available",
                "available": False
            }


class PerplexicaSearchConfig:
    """Configuration for Perplexica search"""
    
    def __init__(
        self,
        api_base: Optional[str] = None,
        default_focus_mode: str = "webSearch",
        default_optimization_mode: str = "balanced",
        chat_model: Optional[Dict[str, Any]] = None,
        embedding_model: Optional[Dict[str, Any]] = None
    ):
        self.api_base = api_base or os.environ.get("PERPLEXICA_API_BASE", "http://localhost:3000")
        self.default_focus_mode = default_focus_mode
        self.default_optimization_mode = default_optimization_mode
        self.chat_model = chat_model or self._get_default_chat_model()
        self.embedding_model = embedding_model or self._get_default_embedding_model()
    
    def _get_default_chat_model(self) -> Dict[str, Any]:
        """Get default chat model configuration"""
        return {
            "provider": "openai",  # Default provider
            "name": os.environ.get("PERPLEXICA_CHAT_MODEL", "gpt-3.5-turbo")
        }
    
    def _get_default_embedding_model(self) -> Dict[str, Any]:
        """Get default embedding model configuration"""
        return {
            "provider": "openai",  # Default provider
            "name": os.environ.get("PERPLEXICA_EMBEDDING_MODEL", "text-embedding-ada-002")
        }