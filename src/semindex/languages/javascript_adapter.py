from __future__ import annotations

from typing import Callable, Optional, Sequence

from ..model import Symbol
from .tree_sitter_adapter import TreeSitterAdapter, TreeSitterNotAvailable

try:  # pragma: no cover - optional dependency probe
    from tree_sitter_languages import get_parser  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    get_parser = None


def _default_parser_factory() -> Callable[[], object] | None:
    if get_parser is None:
        return None

    def _factory() -> object:
        parser = get_parser("javascript")
        if parser is None:
            raise TreeSitterNotAvailable(
                "tree_sitter_languages.get_parser('javascript') returned None"
            )
        return parser

    return _factory


class JavascriptTreeSitterAdapter(TreeSitterAdapter):
    """JavaScript adapter built on top of the generic TreeSitterAdapter."""

    def __init__(self, parser_factory: Optional[Callable[[], object]] = None) -> None:
        if parser_factory is None:
            parser_factory = _default_parser_factory()
        super().__init__(
            "javascript",
            (".js", ".jsx", ".cjs", ".mjs"),
            parser_factory=parser_factory,  # type: ignore[arg-type]
        )

    # NOTE: For now we delegate to the default symbol extraction provided by
    # TreeSitterAdapter, which emits a file-level symbol. Future iterations will
    # populate richer symbols by traversing the tree-sitter AST.

    def _default_symbol(self, path: str, source: str) -> Symbol:
        # Override to adjust naming while reusing base behavior
        total_lines = source.count("\n") + 1
        return Symbol(
            path=path,
            name="<javascript-file>",
            kind="file",
            start_line=1,
            end_line=total_lines,
            signature="",
            docstring=None,
            imports=[],
            bases=[],
            language=self.name,
            namespace=None,
            symbol_type="file",
        )


__all__ = ["JavascriptTreeSitterAdapter"]
