import os
from typing import List, Tuple

from .config import get_config
from .embed import Embedder
from .local_llm import LocalLLM
from .perplexica_adapter import PerplexicaSearchAdapter
from .remote_llm import (
    OpenAICompatibleConfig,
    OpenAICompatibleError,
    OpenAICompatibleLLM,
    resolve_groq_config,
)


def _load_snippet(path: str, start_line: int, end_line: int) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        start = max(start_line - 1, 0)
        end = min(end_line, src.count("\n") + 1)
        return "\n".join(src.splitlines()[start:end])
    except Exception as e:
        return f"[ERROR reading {path}:{start_line}-{end_line}] {e}"


def retrieve_context(index_dir: str, query: str, top_k: int = 8, hybrid: bool = False, embed_model: str | None = None, config_path: str | None = None) -> List[Tuple[float, str]]:
    """
    Returns list of (score, snippet) pairs.
    """
    embedder = Embedder(model_name=embed_model) if embed_model else Embedder()
    qvec = embedder.encode([query])

    if hybrid:
        from .hybrid_search import hybrid_search
        results = hybrid_search(index_dir, qvec, query, top_k=top_k)
    else:
        from .search import search_similar
        results = search_similar(index_dir, qvec, top_k=top_k)

    snippets: List[Tuple[float, str]] = []
    for score, _sid, (path, _name, _kind, start, end, _sig) in results:
        snippet = _load_snippet(path, start, end)
        header = f"# File: {path}:{start}-{end}\n"
        snippets.append((float(score), header + snippet))
    return snippets


def retrieve_context_with_web(index_dir: str, query: str, top_k: int = 8, web_results_count: int = 3, embed_model: str | None = None, config_path: str | None = None) -> List[Tuple[float, str]]:
    """
    Returns list of (score, snippet) pairs combining local codebase results with web search results.
    """
    # Get local codebase results
    local_results = retrieve_context(index_dir, query, top_k, hybrid=False, embed_model=embed_model)
    
    # Add web search results using Perplexica with config-based settings
    perplexica = PerplexicaSearchAdapter(config_path=config_path)
    if perplexica.is_available():
        web_results = perplexica.search_web(query, optimization_mode="balanced")
        
        if "error" not in web_results and "results" in web_results:
            # Extract relevant content from web results
            web_snippets = []
            for result in web_results["results"][:web_results_count]:  # Take top web results
                if "pageContent" in result:
                    title = result.get("title", "No Title")
                    source_url = result.get("url", "No URL")
                    content = result["pageContent"][:500]  # Limit content length
                    snippet = f"# Web Source: {title}\nURL: {source_url}\n\n{content}..."
                    web_snippets.append((0.8, snippet))  # Assign a default score for web results
                elif "content" in result:
                    title = result.get("title", "No Title")
                    source_url = result.get("url", "No URL")
                    content = result["content"][:500]  # Limit content length
                    snippet = f"# Web Source: {title}\nURL: {source_url}\n\n{content}..."
                    web_snippets.append((0.8, snippet))  # Assign a default score for web results
            
            # Combine local and web results
            combined_results = local_results + web_snippets
            
            # Sort by score in descending order (local results will retain their original scores,
            # web results will have a default score that can still be meaningful)
            combined_results.sort(key=lambda x: x[0], reverse=True)
            
            # Return top_k results
            return combined_results[:top_k]
    
    # If web search is not available, return just local results
    return local_results


def retrieve_context_by_mode(index_dir: str, query: str, focus_mode: str = "codeSearch", 
                           top_k: int = 8, hybrid: bool = False, embed_model: str | None = None,
                           config_path: str | None = None) -> List[Tuple[float, str]]:
    """
    Returns list of (score, snippet) pairs based on the specified focus mode.
    """
    if focus_mode == "webSearch":
        perplexica = PerplexicaSearchAdapter(config_path=config_path)
        if perplexica.is_available():
            # Use web search as the primary source
            web_results = perplexica.search_web(query)
            
            if "error" not in web_results and "results" in web_results:
                snippets = []
                for result in web_results["results"][:top_k]:
                    if "pageContent" in result:
                        title = result.get("title", "No Title")
                        source_url = result.get("url", "No URL")
                        content = result["pageContent"][:500]
                        snippet = f"# Web Source: {title}\nURL: {source_url}\n\n{content}..."
                        snippets.append((0.9, snippet))  # High score for web results in web mode
                    elif "content" in result:
                        title = result.get("title", "No Title")
                        source_url = result.get("url", "No URL")
                        content = result["content"][:500]
                        snippet = f"# Web Source: {title}\nURL: {source_url}\n\n{content}..."
                        snippets.append((0.9, snippet))  # High score for web results in web mode
                return snippets
        # Fallback to local search if web search failed
        return retrieve_context(index_dir, query, top_k, hybrid, embed_model, config_path)
    
    elif focus_mode == "librarySearch":
        perplexica = PerplexicaSearchAdapter(config_path=config_path)
        # For library search, we'll try to identify if the query is about a specific library
        # This is a simplified approach - in a more advanced implementation, we'd parse the query better
        import re
        library_pattern = r"library|package|module|dependency|framework"
        
        if re.search(library_pattern, query, re.IGNORECASE) and perplexica.is_available():
            # Use documentation search for library-related queries
            doc_results = perplexica.search_documentation(query)
            
            if "error" not in doc_results and "results" in doc_results:
                snippets = []
                for result in doc_results["results"][:top_k]:
                    if "pageContent" in result:
                        title = result.get("title", "No Title")
                        source_url = result.get("url", "No URL")
                        content = result["pageContent"][:500]
                        snippet = f"# Library Documentation: {title}\nURL: {source_url}\n\n{content}..."
                        snippets.append((0.85, snippet))  # High score for documentation in library mode
                    elif "content" in result:
                        title = result.get("title", "No Title")
                        source_url = result.get("url", "No URL")
                        content = result["content"][:500]
                        snippet = f"# Library Documentation: {title}\nURL: {source_url}\n\n{content}..."
                        snippets.append((0.85, snippet))  # High score for documentation in library mode
                return snippets
        
        # Fallback to local search if documentation search failed
        return retrieve_context(index_dir, query, top_k, hybrid, embed_model, config_path)
    
    elif focus_mode == "hybridSearch":
        # Use the combined approach with config-based settings
        return retrieve_context_with_web(index_dir, query, top_k, web_results_count=3, embed_model=embed_model, config_path=config_path)
    
    else:
        # Default to code search
        return retrieve_context(index_dir, query, top_k, hybrid, embed_model, config_path)



def generate_answer(
    index_dir: str,
    query: str,
    top_k: int = 8,
    hybrid: bool = False,
    embed_model: str | None = None,
    llm_path: str | None = None,
    max_tokens: int = 512,
) -> str:
    snippets = retrieve_context(index_dir, query, top_k=top_k, hybrid=hybrid, embed_model=embed_model)
    # Order by score desc and take text only
    snippets_sorted = [s for _score, s in sorted(snippets, key=lambda x: x[0], reverse=True)]

    system_prompt = (
        "You are a local code assistant. Answer accurately using the provided code/documentation snippets. "
        "Cite file paths inline where relevant. If uncertain, say so."
    )

    if llm_path:
        llm = LocalLLM(model_path=llm_path)
    else:
        # Use transformer model by default
        llm = LocalLLM(model_type="transformer", model_name=os.environ.get("SEMINDEX_TRANSFORMER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    answer = llm.generate(
        system_prompt=system_prompt,
        user_prompt=query,
        context_chunks=snippets_sorted,
        max_tokens=max_tokens,
    )
    return answer


def generate_answer_remote(
    index_dir: str,
    query: str,
    *,
    top_k: int = 8,
    hybrid: bool = False,
    embed_model: str | None = None,
    max_tokens: int = 512,
    system_prompt: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    model: str | None = None,
    stop: list[str] | None = None,
) -> str:
    """Generate an answer using a remote OpenAI-compatible LLM."""

    config: OpenAICompatibleConfig | None = resolve_groq_config(
        api_key=api_key,
        api_base=api_base,
        model=model,
    )
    if config is None:
        raise OpenAICompatibleError(
            "Remote LLM configuration missing API key. "
            "Set `SEMINDEX_REMOTE_API_KEY` or pass `api_key`."
        )

    snippets = retrieve_context(
        index_dir,
        query,
        top_k=top_k,
        hybrid=hybrid,
        embed_model=embed_model,
    )
    snippets_sorted = [
        s for _score, s in sorted(snippets, key=lambda x: x[0], reverse=True)
    ]

    final_system_prompt = system_prompt or (
        "You are a remote code assistant. Answer accurately using the provided "
        "code/documentation snippets. Cite file paths inline where relevant. "
        "If uncertain, say so."
    )

    llm = OpenAICompatibleLLM(config)
    return llm.generate(
        system_prompt=final_system_prompt,
        user_prompt=query,
        context_chunks=snippets_sorted,
        max_tokens=max_tokens,
        stop=stop,
    )


def generate_answer_ollama(
    index_dir: str,
    query: str,
    top_k: int = 8,
    hybrid: bool = False,
    embed_model: str | None = None,
    ollama_model: str | None = None,
    max_tokens: int = 512,
    base_url: str | None = None,
    temperature: float = 0.2,
) -> str:
    """Generate an answer using a local Ollama LLM with retrieved context."""

    # Import here to avoid requiring ollama when not needed
    try:
        from .ollama_llm import OllamaLLM, OllamaError
    except ImportError:
        raise ImportError("Ollama module not available. Install ollama to use Ollama features.")

    # Retrieve context
    snippets = retrieve_context(index_dir, query, top_k=top_k, hybrid=hybrid, embed_model=embed_model)
    # Order by score desc and take text only
    snippets_sorted = [s for _score, s in sorted(snippets, key=lambda x: x[0], reverse=True)]

    system_prompt = (
        "You are an expert code assistant. Answer accurately using the provided code/documentation snippets. "
        "Cite file paths inline where relevant. If uncertain, say so. "
        "Provide detailed explanations and examples when helpful."
    )

    try:
        llm = OllamaLLM(
            model=ollama_model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens
        )
        answer = llm.generate(
            system_prompt=system_prompt,
            user_prompt=query,
            context_chunks=snippets_sorted,
        )
        return answer
    except OllamaError as e:
        raise e
    except ImportError:
        raise ImportError("Ollama module not available. Install ollama to use Ollama features.")
