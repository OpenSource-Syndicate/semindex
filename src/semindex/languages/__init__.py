from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

from .base import LanguageAdapter, ParseResult
from .tree_sitter_adapter import (  # noqa: F401
    TREE_SITTER_AVAILABLE,
    TreeSitterAdapter,
    TreeSitterNotAvailable,
)

try:  # pragma: no cover - optional dependency probe
    from .javascript_adapter import JavascriptTreeSitterAdapter  # noqa: F401
except Exception:  # pragma: no cover - optional dependency missing
    JavascriptTreeSitterAdapter = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency probe
    from tree_sitter_languages import get_parser as _get_parser  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    _get_parser = None  # type: ignore[assignment]


_TREE_SITTER_LANGUAGE_DEFINITIONS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("java", (".java",), "java"),
    ("typescript", (".ts",), "typescript"),
    ("csharp", (".cs",), "c_sharp"),
    (
        "cpp",
        (".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".ipp", ".ixx"),
        "cpp",
    ),
    ("c", (".c", ".h"), "c"),
    ("go", (".go",), "go"),
    ("php", (".php", ".phtml"), "php"),
    ("shell", (".sh", ".bash", ".ksh", ".zsh"), "bash"),
    ("rust", (".rs",), "rust"),
    ("ruby", (".rb",), "ruby"),
)


def _ensure_module_alias() -> None:
    """Ensure tests can reference ``semindex.languages.__init__`` explicitly."""

    module = sys.modules.setdefault("semindex.languages", sys.modules[__name__])
    # Expose alias attribute even when optional adapter is missing so monkeypatching
    # `semindex.languages.__init__.JavascriptTreeSitterAdapter` works reliably.
    setattr(module, "JavascriptTreeSitterAdapter", JavascriptTreeSitterAdapter)
    # Provide an actual module attribute instead of falling back to the built-in
    # ``module.__init__`` method when resolving via getattr.
    setattr(module, "__init__", module)  # type: ignore[attr-defined]
    sys.modules.setdefault("semindex.languages.__init__", module)

# Registry of adapters keyed by language name
_ADAPTERS: Dict[str, LanguageAdapter] = {}
# Mapping of file extensions (lowercase) to adapter name
_EXTENSION_MAP: Dict[str, str] = {}


def register_adapter(adapter: LanguageAdapter) -> None:
    """Register an adapter and make it available for discovery."""
    _ADAPTERS[adapter.name] = adapter
    for ext in adapter.file_extensions:
        _EXTENSION_MAP[ext.lower()] = adapter.name


def clear_adapters() -> None:
    """Reset the adapter registry (primarily for tests)."""
    _ADAPTERS.clear()
    _EXTENSION_MAP.clear()


def available_adapters() -> Sequence[LanguageAdapter]:
    return tuple(_ADAPTERS.values())


def get_adapter(name: str) -> LanguageAdapter:
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown language adapter '{name}'") from exc


def get_adapter_for_path(path: str) -> LanguageAdapter | None:
    _, ext = os.path.splitext(path)
    adapter_name = _EXTENSION_MAP.get(ext.lower())
    if not adapter_name:
        return None
    return _ADAPTERS[adapter_name]


def collect_index_targets(root: str, language: str) -> List[Tuple[LanguageAdapter, str]]:
    """Return [(adapter, path)] pairs for files to index given the language flag."""
    targets: List[Tuple[LanguageAdapter, str]] = []
    if language == "auto":
        seen: set[str] = set()
        for adapter in available_adapters():
            for path in adapter.discover_files(root):
                if path not in seen:
                    seen.add(path)
                    targets.append((adapter, path))
        return targets

    adapter = get_adapter(language)
    for path in adapter.discover_files(root):
        targets.append((adapter, path))
    return targets


def iter_all_supported_files(root: str) -> Iterable[str]:
    seen: set[str] = set()
    for adapter in available_adapters():
        for path in adapter.discover_files(root):
            if path not in seen:
                seen.add(path)
                yield path


def ensure_default_adapters() -> None:
    """Register built-in adapters. Called lazily to avoid circular imports."""
    if _ADAPTERS:
        return
    from .python_adapter import PythonAdapter  # local import to avoid cycles

    register_adapter(PythonAdapter())

    if JavascriptTreeSitterAdapter is None:
        # Don't register tree-sitter adapters if optional dependencies are not available
        return

    try:
        js_adapter = JavascriptTreeSitterAdapter()
    except Exception:
        _register_tree_sitter_language_adapters()
        return

    try:
        if getattr(js_adapter, "is_available", False) and js_adapter.is_available:
            register_adapter(js_adapter)
    except Exception:
        # If availability check fails, skip registration silently
        pass

    _register_tree_sitter_language_adapters()


def _register_tree_sitter_language_adapters() -> None:
    """Register additional tree-sitter backed adapters when available."""

    for language_name, extensions, parser_key in _TREE_SITTER_LANGUAGE_DEFINITIONS:
        adapter = _build_tree_sitter_adapter(language_name, extensions, parser_key)
        if adapter is None:
            continue

        try:
            if getattr(adapter, "is_available", False) and adapter.is_available:
                register_adapter(adapter)
        except Exception:
            continue


def _build_tree_sitter_adapter(
    language_name: str, extensions: Tuple[str, ...], parser_key: str
) -> LanguageAdapter | None:
    parser_factory = _make_parser_factory(parser_key)

    try:
        return TreeSitterAdapter(
            language_name,
            extensions,
            parser_factory=parser_factory,  # type: ignore[arg-type]
        )
    except Exception:
        return None


def _make_parser_factory(parser_key: str):
    if _get_parser is None:
        return None

    def _factory() -> object:
        parser = _get_parser(parser_key)
        if parser is None:
            raise TreeSitterNotAvailable(
                f"tree_sitter_languages.get_parser('{parser_key}') returned None"
            )
        return parser

    return _factory


__all__ = [
    "LanguageAdapter",
    "ParseResult",
    "clear_adapters",
    "available_adapters",
    "collect_index_targets",
    "ensure_default_adapters",
    "get_adapter",
    "get_adapter_for_path",
    "iter_all_supported_files",
    "TreeSitterAdapter",
    "TreeSitterNotAvailable",
    "TREE_SITTER_AVAILABLE",
]

if JavascriptTreeSitterAdapter is not None:
    __all__.append("JavascriptTreeSitterAdapter")


_ensure_module_alias()
