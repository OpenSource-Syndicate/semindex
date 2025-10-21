"""
Focus Modes for semindex AI CLI
Implements different search strategies leveraging Perplexica's capabilities
"""
from enum import Enum
from typing import Dict, List, Optional, Any

from .ai_implementation_assistant import AIImplementationAssistant
from .perplexica_adapter import PerplexicaSearchAdapter


class FocusMode(Enum):
    """Different focus modes for the AI assistant"""
    CODE_SEARCH = "codeSearch"
    DOC_SEARCH = "docSearch"
    WEB_SEARCH = "webSearch"
    ACADEMIC_SEARCH = "academicSearch"
    LIBRARY_SEARCH = "librarySearch"
    YOUTUBE_SEARCH = "youtubeSearch"
    REDDIT_SEARCH = "redditSearch"
    HYBRID_SEARCH = "hybridSearch"


class FocusModeManager:
    """Manages different focus modes and their search strategies"""
    
    def __init__(self, ai_assistant: AIImplementationAssistant):
        self.ai_assistant = ai_assistant
        self.perplexica_adapter = ai_assistant.perplexica_adapter
        self.focus_modes = {
            FocusMode.CODE_SEARCH: self._code_search,
            FocusMode.DOC_SEARCH: self._doc_search,
            FocusMode.WEB_SEARCH: self._web_search,
            FocusMode.ACADEMIC_SEARCH: self._academic_search,
            FocusMode.LIBRARY_SEARCH: self._library_search,
            FocusMode.YOUTUBE_SEARCH: self._youtube_search,
            FocusMode.REDDIT_SEARCH: self._reddit_search,
            FocusMode.HYBRID_SEARCH: self._hybrid_search
        }
    
    def search(self, query: str, focus_mode: FocusMode, **kwargs) -> Dict[str, Any]:
        """Execute search using the specified focus mode"""
        if focus_mode not in self.focus_modes:
            return {
                "error": f"Unknown focus mode: {focus_mode}",
                "available_modes": [mode.value for mode in FocusMode]
            }
        
        search_func = self.focus_modes[focus_mode]
        return search_func(query, **kwargs)
    
    def _code_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform code search within the indexed codebase"""
        try:
            results = self.ai_assistant.function_executor.search_code(query, top_k=kwargs.get('top_k', 5))
            return {
                "results": results,
                "source": "codebase",
                "query": query
            }
        except Exception as e:
            return {
                "error": f"Error in code search: {str(e)}",
                "source": "codebase"
            }
    
    def _doc_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform documentation search using Perplexica"""
        if not self.perplexica_adapter.is_available():
            return {
                "error": "Perplexica API is not available for documentation search",
                "source": "documentation"
            }
        
        try:
            results = self.perplexica_adapter.search_web(query, optimization_mode=kwargs.get('optimization_mode', 'balanced'))
            return {
                "results": results,
                "source": "documentation",
                "query": query
            }
        except Exception as e:
            return {
                "error": f"Error in documentation search: {str(e)}",
                "source": "documentation"
            }
    
    def _web_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform web search using Perplexica"""
        if not self.perplexica_adapter.is_available():
            return {
                "error": "Perplexica API is not available for web search",
                "source": "web"
            }
        
        try:
            results = self.perplexica_adapter.search_web(query, optimization_mode=kwargs.get('optimization_mode', 'balanced'))
            return {
                "results": results,
                "source": "web",
                "query": query
            }
        except Exception as e:
            return {
                "error": f"Error in web search: {str(e)}",
                "source": "web"
            }
    
    def _academic_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform academic search using Perplexica"""
        if not self.perplexica_adapter.is_available():
            return {
                "error": "Perplexica API is not available for academic search",
                "source": "academic"
            }
        
        try:
            results = self.perplexica_adapter.search_academic(query)
            return {
                "results": results,
                "source": "academic",
                "query": query
            }
        except Exception as e:
            return {
                "error": f"Error in academic search: {str(e)}",
                "source": "academic"
            }
    
    def _library_search(self, query: str, library_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Search documentation for a specific library"""
        if not self.perplexica_adapter.is_available():
            return {
                "error": "Perplexica API is not available for library search",
                "source": "library"
            }
        
        try:
            results = self.perplexica_adapter.search_documentation(query, library_name)
            return {
                "results": results,
                "source": "library",
                "query": query,
                "library": library_name
            }
        except Exception as e:
            return {
                "error": f"Error in library search: {str(e)}",
                "source": "library"
            }
    
    def _youtube_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform YouTube search using Perplexica"""
        if not self.perplexica_adapter.is_available():
            return {
                "error": "Perplexica API is not available for YouTube search",
                "source": "youtube"
            }
        
        try:
            results = self.perplexica_adapter.search_youtube(query)
            return {
                "results": results,
                "source": "youtube",
                "query": query
            }
        except Exception as e:
            return {
                "error": f"Error in YouTube search: {str(e)}",
                "source": "youtube"
            }
    
    def _reddit_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform Reddit search using Perplexica"""
        if not self.perplexica_adapter.is_available():
            return {
                "error": "Perplexica API is not available for Reddit search",
                "source": "reddit"
            }
        
        try:
            results = self.perplexica_adapter.search_reddit(query)
            return {
                "results": results,
                "source": "reddit",
                "query": query
            }
        except Exception as e:
            return {
                "error": f"Error in Reddit search: {str(e)}",
                "source": "reddit"
            }
    
    def _hybrid_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform a hybrid search combining codebase and web search"""
        results = {
            "code_results": [],
            "web_results": [],
            "errors": []
        }
        
        # Get code results
        try:
            code_results = self.ai_assistant.function_executor.search_code(query, top_k=kwargs.get('top_k', 3))
            results["code_results"] = code_results
        except Exception as e:
            results["errors"].append(f"Code search error: {str(e)}")
        
        # Get web results if Perplexica is available
        if self.perplexica_adapter.is_available():
            try:
                web_results = self.perplexica_adapter.search_web(query, optimization_mode=kwargs.get('optimization_mode', 'balanced'))
                results["web_results"] = web_results
            except Exception as e:
                results["errors"].append(f"Web search error: {str(e)}")
        else:
            results["errors"].append("Perplexica API not available for web search")
        
        results["source"] = "hybrid"
        results["query"] = query
        
        return results