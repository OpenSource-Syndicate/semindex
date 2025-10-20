from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests


class OpenAICompatibleError(RuntimeError):
    """Raised when a remote OpenAI-compatible LLM call fails."""


@dataclass(slots=True)
class OpenAICompatibleConfig:
    api_base: str
    api_key: str
    model: str
    timeout: int = 60


class OpenAICompatibleLLM:
    """Minimal OpenAI-compatible client (Groq, local proxy, etc.)."""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config
        if not self.config.api_base:
            raise OpenAICompatibleError("Missing OpenAI-compatible API base URL")
        if not self.config.api_key:
            raise OpenAICompatibleError("Missing OpenAI-compatible API key")
        self.session = requests.Session()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        context_chunks: Optional[Iterable[str]] = None,
        max_tokens: int = 768,
        stop: Optional[List[str]] = None,
    ) -> str:
        messages: List[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if context_chunks:
            ctx = "\n\n".join(chunk.strip() for chunk in context_chunks if chunk and chunk.strip())
            if ctx:
                messages.append({"role": "system", "content": f"Context:\n{ctx}"})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop

        url = self._normalize_url("/v1/chat/completions")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        response = self.session.post(
            url,
            data=json.dumps(payload),
            headers=headers,
            timeout=self.config.timeout,
        )
        if response.status_code >= 400:
            raise OpenAICompatibleError(
                f"Remote LLM call failed ({response.status_code}): {response.text[:2000]}"
            )
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise OpenAICompatibleError("Remote LLM returned no choices")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not content.strip():
            raise OpenAICompatibleError("Remote LLM returned empty content")
        return content.strip()

    def _normalize_url(self, path: str) -> str:
        base = self.config.api_base.rstrip("/")
        suffix = path.lstrip("/")
        return f"{base}/{suffix}"


def resolve_groq_config(
    *,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[OpenAICompatibleConfig]:
    api_key = (
        api_key
        or os.environ.get("SEMINDEX_REMOTE_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or ""
    )
    api_base = (
        api_base
        or os.environ.get("SEMINDEX_REMOTE_API_BASE")
        or os.environ.get("GROQ_API_BASE")
        or "https://api.groq.com/openai/"
    )
    model = (
        model
        or os.environ.get("SEMINDEX_REMOTE_MODEL")
        or os.environ.get("GROQ_MODEL")
        or "llama-3.3-70b-versatile"
    )
    if not api_key:
        return None
    return OpenAICompatibleConfig(api_base=api_base, api_key=api_key, model=model)
