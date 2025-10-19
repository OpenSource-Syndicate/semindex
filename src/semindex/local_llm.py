import os
from typing import List, Optional, Iterable

# Lightweight CPU-only local LLM via llama.cpp bindings
# Requires a local GGUF file (quantized), e.g., phi-3-mini-4k-instruct.Q4_K_M.gguf
# Default model path can be overridden with env SEMINDEX_LLM_PATH

DEFAULT_LLM_PATH = os.environ.get(
    "SEMINDEX_LLM_PATH",
    os.path.join(".semindex", "models", "phi-3-mini-4k-instruct.Q4_K_M.gguf"),
)


class LocalLLM:
    """
    Simple wrapper around llama-cpp-python for local CPU inference.

    Example:
        llm = LocalLLM()  # ensure the GGUF exists at DEFAULT_LLM_PATH or set SEMINDEX_LLM_PATH
        text = llm.generate("You are a helpful assistant.", "Explain FAISS in two sentences.")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        temperature: float = 0.2,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
    ) -> None:
        from llama_cpp import Llama  # local import

        self.model_path = model_path or DEFAULT_LLM_PATH
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Local LLM model not found at {self.model_path}. "
                "Download a GGUF (e.g., phi-3-mini-4k-instruct Q4) and set SEMINDEX_LLM_PATH."
            )

        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty

        # Auto threads from env or CPU count
        if n_threads is None:
            try:
                import multiprocessing as mp
                n_threads = max(1, mp.cpu_count() - 1)
            except Exception:
                n_threads = 4

        self.client = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,  # 0 for pure CPU
            logits_all=False,
            verbose=False,
        )

    def build_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        context_chunks: Optional[Iterable[str]] = None,
    ) -> str:
        """Build a compact prompt suitable for small instruction-tuned models."""
        ctx = "\n\n".join(chunk.strip() for chunk in (context_chunks or []) if chunk and chunk.strip())
        parts: List[str] = []
        if system_prompt:
            parts.append(f"<|system|>\n{system_prompt.strip()}\n")
        if ctx:
            parts.append(f"<|context|>\n{ctx}\n")
        parts.append(f"<|user|>\n{user_prompt.strip()}\n")
        parts.append("<|assistant|>\n")
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

        result = self.client(
            prompt,
            max_tokens=max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repeat_penalty=self.repeat_penalty,
            stop=stop or ["</s>", "<|user|>", "<|system|>", "<|context|>"],
        )
        # llama-cpp returns a dict with 'choices'
        text = result["choices"][0]["text"] if result and "choices" in result else ""
        return text.strip()
