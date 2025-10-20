from __future__ import annotations

from typing import TYPE_CHECKING, List

from ..ast_py import parse_python_symbols
from ..model import Chunk, ChunkingConfig, Symbol
from .base import LanguageAdapter, ParseResult

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..embed import Embedder


class PythonAdapter(LanguageAdapter):
    name = "python"
    file_extensions = (".py",)

    def process_file(
        self,
        path: str,
        source: str,
        embedder: "Embedder",
        chunk_config: ChunkingConfig,
    ) -> ParseResult:
        from ..chunker import build_chunks_from_symbols, build_semantic_chunks_from_symbols

        symbols, calls = parse_python_symbols(path, source)

        if chunk_config.method == "semantic":
            chunks = build_semantic_chunks_from_symbols(
                source,
                symbols,
                embedder,
                similarity_threshold=chunk_config.similarity_threshold,
            )
        else:
            chunks = build_chunks_from_symbols(source, symbols)

        return ParseResult(symbols=symbols, chunks=chunks, calls=calls)
