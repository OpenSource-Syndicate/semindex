from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from ..embed import Embedder
from ..model import Chunk, ChunkingConfig, Symbol
from .base import LanguageAdapter, ParseResult

try:  # pragma: no cover - optional dependency probe
    from tree_sitter import Language as _TSLanguage  # type: ignore
    from tree_sitter import Parser as _TSParser  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    _TSLanguage = None
    _TSParser = None

TREE_SITTER_AVAILABLE = _TSParser is not None

ParserFactory = Callable[[], "_TSParser"]


class TreeSitterNotAvailable(RuntimeError):
    """Raised when tree-sitter support is requested but not available."""


class TreeSitterAdapter(LanguageAdapter):
    """Base adapter that integrates tree-sitter grammars with the indexing pipeline.

    Subclasses should either supply a compiled tree-sitter language library path or a
    custom ``parser_factory`` that returns a configured ``tree_sitter.Parser``.
    """

    def __init__(
        self,
        language_name: str,
        file_extensions: Sequence[str],
        *,
        language_library: Optional[object] = None,
        parser_factory: Optional[ParserFactory] = None,
    ) -> None:
        self._language_name = language_name
        self._file_extensions = tuple(file_extensions)
        self._language_library = language_library
        self._parser_factory = parser_factory
        self._parser: Optional[_TSParser] = None
        self._ts_language: Optional[_TSLanguage] = None

    # ------------------------------------------------------------------
    # LanguageAdapter API
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._language_name

    @property
    def file_extensions(self) -> Sequence[str]:
        return self._file_extensions

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        """Return True if a parser can be created for this adapter."""

        if self._parser_factory is not None:
            try:
                parser = self._parser_factory()
            except Exception:
                return False
            self._parser = parser
            return True

        if not TREE_SITTER_AVAILABLE:
            return False

        try:
            self._ensure_parser()
        except TreeSitterNotAvailable:
            return False
        except Exception:
            return False
        return True

    # ------------------------------------------------------------------
    # Parsing & symbol extraction
    # ------------------------------------------------------------------
    def process_file(
        self,
        path: str,
        source: str,
        embedder: Embedder,
        chunk_config: ChunkingConfig,
    ) -> ParseResult:
        parser = self._ensure_parser()
        tree = parser.parse(source.encode("utf-8"))

        symbols = self.extract_symbols(path, source, tree)
        if not symbols:
            symbols = [self._default_symbol(path, source)]

        chunks = self.build_chunks(source, symbols, chunk_config)
        return ParseResult(symbols=symbols, chunks=chunks, calls=None)

    # ------------------------------------------------------------------
    # Overridable extension points
    # ------------------------------------------------------------------
    def extract_symbols(self, path: str, source: str, tree) -> List[Symbol]:
        """Return language-aware symbols for the parsed syntax tree.

        Subclasses are expected to override this method to walk the tree and
        produce ``Symbol`` entries. The default implementation yields a single
        file-level symbol covering the entire file.
        """

        return [self._default_symbol(path, source)]

    def build_chunks(
        self,
        source: str,
        symbols: Sequence[Symbol],
        chunk_config: ChunkingConfig,
    ) -> List[Chunk]:
        """Create chunks for the provided symbols.

        The default behavior is a 1:1 mapping between symbols and chunks using
        the textual range indicated by each symbol.
        """

        chunks: List[Chunk] = []
        for sym in symbols:
            text = self._slice_text(source, sym.start_line, sym.end_line)
            chunks.append(Chunk(symbol=sym, text=text))
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_parser(self) -> _TSParser:
        if self._parser is not None:
            return self._parser

        if self._parser_factory is not None:
            parser = self._parser_factory()
            self._parser = parser
            return parser

        if not TREE_SITTER_AVAILABLE or _TSParser is None:
            raise TreeSitterNotAvailable(
                "tree-sitter is not installed. Please install the 'tree_sitter' package "
                "or supply a custom parser_factory."
            )

        language = self._load_language()
        parser = _TSParser()
        parser.set_language(language)
        self._parser = parser
        self._ts_language = language
        return parser

    def _load_language(self) -> _TSLanguage:
        if self._ts_language is not None:
            return self._ts_language

        if _TSLanguage is None:
            raise TreeSitterNotAvailable(
                "tree-sitter Language bindings unavailable. Install the 'tree_sitter'"
            )

        if isinstance(self._language_library, _TSLanguage):
            return self._language_library

        if isinstance(self._language_library, str):
            return _TSLanguage(self._language_library, self._language_name)

        # Attempt to load via tree_sitter_languages helper if available
        if self._language_library is None:
            try:
                from tree_sitter_languages import get_language  # type: ignore
            except ImportError as exc:  # pragma: no cover - requires optional dep
                raise TreeSitterNotAvailable(
                    "No tree-sitter language library provided and 'tree_sitter_languages' "
                    "package is not installed."
                ) from exc

            language = get_language(self._language_name)
            if language is None:
                raise TreeSitterNotAvailable(
                    f"tree_sitter_languages.get_language('{self._language_name}') returned None"
                )
            return language

        raise TreeSitterNotAvailable(
            "Unsupported language_library value. Provide a path to a compiled grammar, "
            "an instance of tree_sitter.Language, or install tree_sitter_languages."
        )

    def _default_symbol(self, path: str, source: str) -> Symbol:
        total_lines = source.count("\n") + 1
        return Symbol(
            path=path,
            name=f"<{self._language_name}-file>",
            kind="file",
            start_line=1,
            end_line=total_lines,
            signature="",
            docstring=None,
            imports=[],
            bases=[],
            language=self._language_name,
            namespace=None,
            symbol_type="file",
        )

    @staticmethod
    def _slice_text(source: str, start_line: int, end_line: int) -> str:
        lines = source.splitlines()
        start_index = max(start_line - 1, 0)
        end_index = min(end_line, len(lines))
        if end_index == 0:
            return source
        return "\n".join(lines[start_index:end_index])


__all__ = [
    "TreeSitterAdapter",
    "TreeSitterNotAvailable",
    "TREE_SITTER_AVAILABLE",
]
