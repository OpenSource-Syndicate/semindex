import os
from typing import List, Tuple

from .embed import Embedder
from .local_llm import LocalLLM
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


def retrieve_context(index_dir: str, query: str, top_k: int = 8, hybrid: bool = False, embed_model: str | None = None) -> List[Tuple[float, str]]:
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

    llm = LocalLLM(model_path=llm_path)
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
