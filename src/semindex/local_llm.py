import hashlib
import os
import tempfile
from typing import Iterable, List, Optional

import requests

# Lightweight CPU-only local LLM supporting both llama.cpp and transformer models
# For llama.cpp: Requires a local GGUF file (quantized), e.g., phi-3-mini-4k-instruct.Q4_K_M.gguf
# For transformers: Uses HuggingFace models directly
# Default model path can be overridden with env SEMINDEX_LLM_PATH

DEFAULT_LLM_PATH = os.environ.get(
    "SEMINDEX_LLM_PATH",
    os.path.join(".semindex", "models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
)

DEFAULT_LLM_URL = os.environ.get(
    "SEMINDEX_LLM_URL",
    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=1",
)

DEFAULT_LLM_SHA256 = os.environ.get(
    "SEMINDEX_LLM_SHA256",
    "f5c78a7c5d5d657a7eabe76fd2cb51c23b71612f6c1b23c2e9f06d6d61bb64ed",
)

AUTO_DOWNLOAD = os.environ.get("SEMINDEX_LLM_AUTO_DOWNLOAD", "1").lower() not in {
    "0",
    "false",
    "no",
}


class LocalLLM:
    """
    Wrapper for local LLM inference, supporting both llama.cpp and transformer models.

    Example:
        # For llama.cpp models
        llm = LocalLLM(model_type="llama-cpp")  # ensure the GGUF exists at DEFAULT_LLM_PATH
        # For transformer models
        llm = LocalLLM(model_type="transformer", model_name="microsoft/phi-2")
        text = llm.generate("You are a helpful assistant.", "Explain FAISS in two sentences.")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,  # For transformer models
        model_type: str = "llama-cpp",  # "llama-cpp" or "transformer"
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        temperature: float = 0.2,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
        # Transformer-specific parameters
        torch_dtype: Optional[object] = None,  # torch.dtype
        device: Optional[str] = None,
    ) -> None:
        self.model_type = model_type.lower()
        
        if self.model_type == "llama-cpp":
            self._init_llama_cpp(
                model_path, n_ctx, n_threads, n_gpu_layers, 
                temperature, top_p, repeat_penalty
            )
        elif self.model_type == "transformer":
            self._init_transformer(
                model_name, n_ctx, temperature, top_p, 
                repeat_penalty, torch_dtype, device
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'llama-cpp' or 'transformer'")

    def _init_llama_cpp(
        self,
        model_path: Optional[str],
        n_ctx: int,
        n_threads: Optional[int],
        n_gpu_layers: int,
        temperature: float,
        top_p: float,
        repeat_penalty: float,
    ):
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The optional dependency 'llama-cpp-python' is required for LocalLLM with llama-cpp type. "
                "Install it via `pip install llama-cpp-python` or run the tooling with `--no-llm` / "
                "set SEMINDEX_LLM_AUTO_DOWNLOAD=0 to skip local generation."
            ) from exc

        self.model_path = model_path or DEFAULT_LLM_PATH
        self._ensure_model_llama()

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
        
        # Store model type specific methods
        self._generate_method = self._generate_llama_cpp

    def _init_transformer(
        self,
        model_name: Optional[str],
        n_ctx: int,
        temperature: float,
        top_p: float,
        repeat_penalty: float,
        torch_dtype: Optional[object],
        device: Optional[str],
    ):
        try:
            from .transformer_llm import TransformerLLM
        except ImportError as exc:
            raise RuntimeError(
                "The optional dependencies for transformer models are required. "
                "Install them via `pip install transformers torch`."
            ) from exc
        
        # If no model name provided, use TinyLlama as default
        if model_name is None:
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        self.client = TransformerLLM(
            model_name=model_name,
            n_ctx=n_ctx,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repeat_penalty,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        # Store model type specific methods
        self._generate_method = self._generate_transformer

    def _generate_llama_cpp(
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

    def _generate_transformer(
        self,
        system_prompt: str,
        user_prompt: str,
        context_chunks: Optional[Iterable[str]] = None,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> str:
        return self.client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context_chunks=context_chunks,
            max_tokens=max_tokens,
            stop=stop,
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_model_llama(self) -> None:
        if os.path.exists(self.model_path):
            if DEFAULT_LLM_SHA256:
                self._verify_checksum(self.model_path, DEFAULT_LLM_SHA256)
            return

        if not AUTO_DOWNLOAD:
            raise FileNotFoundError(
                f"Local LLM model not found at {self.model_path}. "
                "Set SEMINDEX_LLM_AUTO_DOWNLOAD=1 to allow automatic download or provide the file manually."
            )

        url = DEFAULT_LLM_URL
        directory = os.path.dirname(self.model_path)
        os.makedirs(directory, exist_ok=True)
        print(f"[llm] Downloading TinyLlama GGUF model from {url} ...")
        try:
            self._download_file(url, self.model_path)
        except Exception as exc:
            raise FileNotFoundError(
                f"Failed to download local model from {url}: {exc}"
            ) from exc

        if DEFAULT_LLM_SHA256:
            self._verify_checksum(self.model_path, DEFAULT_LLM_SHA256)

    @staticmethod
    def _download_file(url: str, destination: str, chunk_size: int = 2 ** 20) -> None:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                try:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        tmp.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            percent = downloaded / total * 100
                            print(f"[llm] Downloaded {downloaded // (1024 * 1024)} MiB ({percent:0.1f}%)", end="\r")
                    tmp.flush()
                    os.replace(tmp.name, destination)
                except Exception:
                    os.unlink(tmp.name)
                    raise
        print(f"\n[llm] Model stored at {destination}")

    @staticmethod
    def _verify_checksum(path: str, expected_sha256: str) -> None:
        if not expected_sha256:
            return
        sha256 = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(2 ** 20), b""):
                sha256.update(chunk)
        digest = sha256.hexdigest()
        if digest != expected_sha256.lower():
            raise FileNotFoundError(
                f"Checksum mismatch for {path}. Expected {expected_sha256}, got {digest}."
            )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        context_chunks: Optional[Iterable[str]] = None,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> str:
        return self._generate_method(
            system_prompt, user_prompt, context_chunks, max_tokens, stop
        )