from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from ..model import Symbol
from .tree_sitter_adapter import TreeSitterAdapter, TreeSitterNotAvailable

try:  # pragma: no cover - typing hint only
    from tree_sitter import Node  # type: ignore
except Exception:  # pragma: no cover - no runtime dependency
    Node = "Node"  # type: ignore[misc,assignment]

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

    def extract_symbols(self, path: str, source: str, tree) -> List[Symbol]:
        source_bytes = source.encode("utf-8")
        symbols: List[Symbol] = []
        self._collect_symbols(
            tree.root_node,
            path,
            source_bytes,
            namespace_stack=[],
            symbols=symbols,
        )

        if not symbols:
            return [self._default_symbol(path, source)]

        return symbols

    # ------------------------------------------------------------------
    # Symbol extraction helpers
    # ------------------------------------------------------------------
    def _collect_symbols(
        self,
        node: Node,
        path: str,
        source_bytes: bytes,
        *,
        namespace_stack: List[str],
        symbols: List[Symbol],
    ) -> None:
        node_type = node.type

        if node_type in {"export_statement", "export_declaration"}:
            for child in node.children:
                self._collect_symbols(
                    child,
                    path,
                    source_bytes,
                    namespace_stack=namespace_stack,
                    symbols=symbols,
                )
            return

        if node_type == "class_declaration":
            self._handle_class_declaration(
                node,
                path,
                source_bytes,
                namespace_stack=namespace_stack,
                symbols=symbols,
            )
            return

        if node_type == "function_declaration":
            symbol = self._build_function_symbol(
                node,
                path,
                source_bytes,
                namespace=namespace_stack[-1] if namespace_stack else None,
            )
            if symbol is not None:
                symbols.append(symbol)
            return

        if node_type in {"lexical_declaration", "variable_declaration"}:
            self._handle_variable_declaration(
                node,
                path,
                source_bytes,
                namespace_stack=namespace_stack,
                symbols=symbols,
            )

        for child in node.children:
            self._collect_symbols(
                child,
                path,
                source_bytes,
                namespace_stack=namespace_stack,
                symbols=symbols,
            )

    def _handle_class_declaration(
        self,
        node: Node,
        path: str,
        source_bytes: bytes,
        *,
        namespace_stack: List[str],
        symbols: List[Symbol],
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        class_name = self._node_text(name_node, source_bytes)
        if not class_name:
            return

        start_line, end_line = self._node_line_span(node)
        superclass_node = node.child_by_field_name("superclass")
        bases = []
        if superclass_node is not None:
            base_name = self._node_text(superclass_node, source_bytes)
            if base_name:
                bases.append(base_name)

        symbols.append(
            Symbol(
                path=path,
                name=class_name,
                kind="class",
                start_line=start_line,
                end_line=end_line,
                signature=f"class {class_name}",
                docstring=None,
                imports=[],
                bases=bases,
                language=self.name,
                namespace=None,
                symbol_type="class",
            )
        )

        class_body = node.child_by_field_name("body")
        if class_body is None:
            return

        namespace_stack.append(class_name)
        for member in class_body.children:
            if member.type == "method_definition":
                symbol = self._build_method_symbol(
                    member,
                    path,
                    source_bytes,
                    namespace="::".join(namespace_stack),
                )
                if symbol is not None:
                    symbols.append(symbol)
            else:
                self._collect_symbols(
                    member,
                    path,
                    source_bytes,
                    namespace_stack=namespace_stack,
                    symbols=symbols,
                )
        namespace_stack.pop()

    def _handle_variable_declaration(
        self,
        node: Node,
        path: str,
        source_bytes: bytes,
        *,
        namespace_stack: List[str],
        symbols: List[Symbol],
    ) -> None:
        for child in node.children:
            if child.type != "variable_declarator":
                continue

            name_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")
            if name_node is None or value_node is None:
                continue

            if value_node.type not in {"function", "function_expression", "arrow_function"}:
                continue

            symbol = self._build_function_symbol(
                value_node,
                path,
                source_bytes,
                namespace=namespace_stack[-1] if namespace_stack else None,
                explicit_name=self._node_text(name_node, source_bytes),
            )
            if symbol is not None:
                symbols.append(symbol)

    def _build_method_symbol(
        self,
        node: Node,
        path: str,
        source_bytes: bytes,
        *,
        namespace: str,
    ) -> Optional[Symbol]:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            name_node = node.child_by_field_name("property")
        if name_node is None:
            return None

        method_name = self._node_text(name_node, source_bytes)
        if not method_name:
            return None

        fn_node = node.child_by_field_name("value") or node
        params_text = self._parameters_text(fn_node, source_bytes)
        start_line, end_line = self._node_line_span(node)

        return Symbol(
            path=path,
            name=method_name,
            kind="method",
            start_line=start_line,
            end_line=end_line,
            signature=f"{method_name}{params_text}",
            docstring=None,
            imports=[],
            bases=[],
            language=self.name,
            namespace=namespace,
            symbol_type="method",
        )

    def _build_function_symbol(
        self,
        node: Node,
        path: str,
        source_bytes: bytes,
        *,
        namespace: Optional[str],
        explicit_name: Optional[str] = None,
    ) -> Optional[Symbol]:
        name_node = node.child_by_field_name("name")
        name = explicit_name or (self._node_text(name_node, source_bytes) if name_node else None)
        if not name:
            return None

        params_text = self._parameters_text(node, source_bytes)
        start_line, end_line = self._node_line_span(node)

        return Symbol(
            path=path,
            name=name,
            kind="function",
            start_line=start_line,
            end_line=end_line,
            signature=f"{name}{params_text}",
            docstring=None,
            imports=[],
            bases=[],
            language=self.name,
            namespace=namespace,
            symbol_type="function" if namespace is None else "method",
        )

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _node_text(node: Node, source_bytes: bytes) -> str:
        return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore").strip()

    @staticmethod
    def _node_line_span(node: Node) -> tuple[int, int]:
        start_row, _ = node.start_point
        end_row, _ = node.end_point
        return start_row + 1, end_row + 1

    def _parameters_text(self, node: Node, source_bytes: bytes) -> str:
        params_node = node.child_by_field_name("parameters")
        if params_node is None:
            return "()"

        params_text = self._node_text(params_node, source_bytes)
        if params_text.startswith("(") and params_text.endswith(")"):
            return params_text
        return f"({params_text})"

    def _default_symbol(self, path: str, source: str) -> Symbol:
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
