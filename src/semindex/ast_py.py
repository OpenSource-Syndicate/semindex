import ast
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Symbol:
    path: str
    name: str
    kind: str  # module|class|function|method
    start_line: int
    end_line: int
    signature: str
    docstring: Optional[str]
    imports: List[str]
    bases: List[str]


@dataclass
class Chunk:
    symbol: Symbol
    text: str


def _get_end_lineno(node: ast.AST) -> int:
    # Python 3.8+ ast nodes typically have end_lineno; fallback via body traversal
    end = getattr(node, "end_lineno", None)
    if end is not None:
        return end
    max_end = getattr(node, "lineno", 1)
    for child in ast.walk(node):
        if hasattr(child, "end_lineno") and child.end_lineno:
            max_end = max(max_end, child.end_lineno)
        elif hasattr(child, "lineno"):
            max_end = max(max_end, child.lineno)
    return max_end


def _normalize_source(source: str) -> str:
    """Normalize indentation to avoid IndentationError when text has been partially
    left-stripped (e.g., first line column 0 but following lines still indented).

    Strategy:
    1) textwrap.dedent to remove common indent.
    2) If some lines still share a uniform minimal leading space > 0 (because the first
       line had 0 indent, preventing dedent), remove that minimal indent from those lines.
    """
    s = textwrap.dedent(source)
    lines = s.splitlines()
    # compute minimal indent of lines that start with spaces
    indents = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip() and l.startswith(" ")]
    if indents:
        min_indent = min(indents)
        if min_indent > 0:
            new_lines = []
            prefix = " " * min_indent
            for l in lines:
                if l.startswith(prefix):
                    new_lines.append(l[min_indent:])
                else:
                    new_lines.append(l)
            s = "\n".join(new_lines)
    return s


def parse_python_symbols(path: str, source: str) -> Tuple[List[Symbol], List[Tuple[str, str]]]:
    """
    Returns (symbols, edges_calls_approx)
    - edges_calls_approx: list of (caller_name, callee_name) approximated via ast.Call
    """
    norm = _normalize_source(source)
    tree = ast.parse(norm)

    symbols: List[Symbol] = []
    calls: List[Tuple[str, str]] = []

    module_doc = ast.get_docstring(tree)
    module_symbol = Symbol(
        path=path,
        name="<module>",
        kind="module",
        start_line=1,
        end_line=source.count("\n") + 1,
        signature="",
        docstring=module_doc,
        imports=[],
        bases=[],
    )
    symbols.append(module_symbol)

    imports: List[str] = []

    class FunctionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.scope: List[str] = []

        def _qualname(self, name: str) -> str:
            return ".".join(self.scope + [name]) if self.scope else name

        def visit_Import(self, node: ast.Import):
            for alias in node.names:
                imports.append(alias.name)

        def visit_ImportFrom(self, node: ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                imports.append(f"{mod}.{alias.name}" if mod else alias.name)

        def visit_ClassDef(self, node: ast.ClassDef):
            qn = self._qualname(node.name)
            bases = []
            for b in node.bases:
                try:
                    bases.append(ast.unparse(b))  # type: ignore[attr-defined]
                except Exception:
                    bases.append(getattr(getattr(b, "id", None), "",) or type(b).__name__)
            start = node.lineno
            end = _get_end_lineno(node)
            doc = ast.get_docstring(node)
            sym = Symbol(
                path=path,
                name=qn,
                kind="class",
                start_line=start,
                end_line=end,
                signature=f"class {node.name}(...)",
                docstring=doc,
                imports=[],
                bases=bases,
            )
            symbols.append(sym)
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            self._handle_function(node, is_async=False)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self._handle_function(node, is_async=True)

        def _handle_function(self, node, is_async: bool):
            qn = self._qualname(node.name)
            start = node.lineno
            end = _get_end_lineno(node)
            doc = ast.get_docstring(node)
            args_src = _safe_unparse(node.args)
            prefix = "async def" if is_async else "def"
            kind = "method" if self.scope else "function"
            sym = Symbol(
                path=path,
                name=qn,
                kind=kind,
                start_line=start,
                end_line=end,
                signature=f"{prefix} {node.name}({args_src})",
                docstring=doc,
                imports=[],
                bases=[],
            )
            symbols.append(sym)
            # approximate calls from this function body
            caller = qn
            for call in [n for n in ast.walk(node) if isinstance(n, ast.Call)]:
                callee = _extract_call_name(call.func)
                if callee:
                    calls.append((caller, callee))

            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

    FunctionVisitor().visit(tree)

    # attach module-level imports to module symbol
    symbols[0].imports = list(sorted(set(imports)))

    return symbols, calls


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)  # type: ignore[attr-defined]
    except Exception:
        return "..."


def _extract_call_name(node: ast.AST) -> Optional[str]:
    """Best-effort extraction of a callee name from an AST node.
    Handles common shapes like Name, Attribute, nested Call (e.g., A().m),
    and Subscript by peeling layers and composing a dotted path when possible.
    """
    # foo(...)
    if isinstance(node, ast.Call):
        return _extract_call_name(node.func)
    # obj.attr
    if isinstance(node, ast.Attribute):
        base = _extract_call_name(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    # simple name
    if isinstance(node, ast.Name):
        return node.id
    # subscripted call target like pkg["x"].fn
    if isinstance(node, ast.Subscript):
        return _extract_call_name(node.value)
    return None


essential_ws = {" ", "\t"}

def extract_text_for_symbol(source: str, sym: Symbol) -> str:
    lines = source.splitlines()
    start = max(sym.start_line - 1, 0)
    end = min(sym.end_line, len(lines))
    snippet = "\n".join(lines[start:end])
    # include docstring in text if missing
    doc = sym.docstring or ""
    if doc and doc not in snippet:
        return f"{sym.signature}\n\n\"\"\"{doc}\"\"\"\n{snippet}"
    return snippet if snippet.strip() else sym.signature or sym.name
