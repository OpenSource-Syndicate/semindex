from __future__ import annotations

from pathlib import Path

import pytest

from semindex.languages import (
    available_adapters,
    clear_adapters,
    collect_index_targets,
    ensure_default_adapters,
    get_adapter_for_path,
    register_adapter,
)
from semindex.languages.base import LanguageAdapter, ParseResult
from semindex.model import ChunkingConfig


class _StubAdapter(LanguageAdapter):
    def __init__(self, name: str, exts: tuple[str, ...]):
        self._name = name
        self._exts = exts

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return self._name

    @property
    def file_extensions(self) -> tuple[str, ...]:  # pragma: no cover - trivial
        return self._exts

    def process_file(self, path, source, embedder, chunk_config) -> ParseResult:  # pragma: no cover
        raise NotImplementedError


def test_clear_and_register_adapter_updates_extension_map(tmp_path: Path):
    clear_adapters()
    stub = _StubAdapter("dummy", (".foo", ".bar"))
    register_adapter(stub)

    dummy_file = tmp_path / "example.foo"
    dummy_file.write_text("// stub", encoding="utf-8")

    adapter = get_adapter_for_path(str(dummy_file))
    assert adapter is stub


def test_ensure_default_adapters_python_only_without_optional(monkeypatch):
    monkeypatch.setattr("semindex.languages.__init__.JavascriptTreeSitterAdapter", None)
    clear_adapters()
    ensure_default_adapters()

    names = {adapter.name for adapter in available_adapters()}
    assert names == {"python"}


def test_ensure_default_adapters_registers_optional_when_available(monkeypatch, tmp_path: Path):
    class FakeJs(LanguageAdapter):
        @property
        def name(self) -> str:
            return "javascript"

        @property
        def file_extensions(self) -> tuple[str, ...]:
            return (".js",)

        @property
        def is_available(self) -> bool:  # type: ignore[override]
            return True

        def process_file(self, path, source, embedder, chunk_config) -> ParseResult:  # pragma: no cover
            raise NotImplementedError

    monkeypatch.setattr("semindex.languages.__init__.JavascriptTreeSitterAdapter", FakeJs)
    clear_adapters()
    ensure_default_adapters()

    adapters = {adapter.name: adapter for adapter in available_adapters()}
    assert "python" in adapters
    assert "javascript" in adapters

    js_file = tmp_path / "main.js"
    js_file.write_text("function hello() { return 42; }", encoding="utf-8")

    targets = collect_index_targets(str(tmp_path), "javascript")
    assert targets
    assert all(adapter.name == "javascript" for adapter, _path in targets)
