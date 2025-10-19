from __future__ import annotations

import pytest

from semindex.languages.javascript_adapter import JavascriptTreeSitterAdapter
from semindex.model import ChunkingConfig


class _DummyEmbedder:
    def encode(self, *_args, **_kwargs):  # pragma: no cover - not used
        raise NotImplementedError


@pytest.mark.skipif(
    JavascriptTreeSitterAdapter is None,
    reason="JavaScript adapter not available",
)
def test_javascript_adapter_extracts_symbols(tmp_path):
    adapter = JavascriptTreeSitterAdapter()

    if not getattr(adapter, "is_available", False):
        pytest.skip("tree-sitter JavaScript parser unavailable")

    source = (
        "class Greeter {\n"
        "  constructor(name) { this.name = name; }\n"
        "  hello(person) { return `Hello ${person}`; }\n"
        "}\n"
        "const farewell = (who) => { return `Bye ${who}`; };\n"
        "function ping(value) { return value; }\n"
    )

    path = tmp_path / "sample.js"
    path.write_text(source, encoding="utf-8")

    result = adapter.process_file(
        str(path),
        source,
        embedder=_DummyEmbedder(),
        chunk_config=ChunkingConfig(method="symbol"),
    )

    symbol_map = {sym.name: sym for sym in result.symbols}

    assert "Greeter" in symbol_map
    assert symbol_map["Greeter"].symbol_type == "class"

    assert "hello" in symbol_map
    assert symbol_map["hello"].namespace == "Greeter"
    assert symbol_map["hello"].symbol_type == "method"

    assert "farewell" in symbol_map
    assert symbol_map["farewell"].symbol_type in {"function", "method"}

    assert "ping" in symbol_map
    assert symbol_map["ping"].symbol_type == "function"

    assert len(result.chunks) == len(result.symbols)
