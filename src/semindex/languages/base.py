from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from ..crawler import iter_files
from ..embed import Embedder
from ..model import Chunk, ChunkingConfig, Symbol


@dataclass
class ParseResult:
    symbols: List[Symbol]
    chunks: List[Chunk]
    calls: List[Tuple[str, str]] | None = None


class LanguageAdapter(ABC):
    """Abstract interface for language-specific indexing adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Lowercase identifier for the language (e.g. 'python')."""

    @property
    @abstractmethod
    def file_extensions(self) -> Sequence[str]:
        """File extensions (including leading dot) supported by this adapter."""

    def discover_files(self, root: str) -> Iterable[str]:
        """Yield files belonging to this language under the provided root."""
        return iter_files(root, self.file_extensions)

    @abstractmethod
    def process_file(
        self,
        path: str,
        source: str,
        embedder: Embedder,
        chunk_config: ChunkingConfig,
    ) -> ParseResult:
        """
        Produce language-aware symbols and chunks for a single file.
        Implementations should enrich symbols with metadata like language,
        namespace, and symbol_type when available.
        """
        raise NotImplementedError
