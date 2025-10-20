from __future__ import annotations

import importlib.util
import sys
import json
from pathlib import Path
from typing import Dict

import pytest

from semindex.docs import MermaidGraph


def _load_gen_docs_module(_: Path) -> object:
    # Resolve repository root from this test file location
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "gen_docs.py"
    assert script_path.exists(), f"Missing gen_docs.py at {script_path}"
    mod_name = f"gen_docs_{hash(str(script_path)) & 0xFFFF:x}"
    spec = importlib.util.spec_from_file_location(mod_name, str(script_path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # Register before exec_module to fix dataclass __module__ lookup
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def _dummy_graphs() -> Dict[str, MermaidGraph]:
    return {
        "pipeline": MermaidGraph(title="p", diagram="graph TD\n A[Node A]-->B[Node B]"),
        "module": MermaidGraph(title="m", diagram="graph TD\n A[Node A]-->B[Node B]"),
        "adapter": MermaidGraph(title="a", diagram="graph TD\n A[Node A]-->B[Node B]", metadata={"adapter_count": 2}),
    }


def _dummy_stats() -> Dict[str, object]:
    return {
        "repo_root": ".",
        "index_dir": ".semindex",
        "indexed_files": 10,
        "symbols": 123,
        "symbols_by_language": {"python": 120, "js": 3},
        "docs_indexed": [{"package": "pkg", "pages": 5}],
    }


def test_run_no_llm_static_sections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: sandbox wiki dir
    repo_root = tmp_path
    (repo_root / "wiki" / "templates").mkdir(parents=True)
    # minimal required templates
    (repo_root / "wiki" / "templates" / "overview.prompt").write_text("Context:\n{{context}}\n\nOverview.")
    (repo_root / "wiki" / "templates" / "architecture.prompt").write_text("Context:\n{{context}}\n\nArchitecture.")
    (repo_root / "wiki" / "templates" / "adapters.prompt").write_text("Context:\n{{context}}\n\nAdapters.")

    gen_docs = _load_gen_docs_module(repo_root)
    # Monkeypatch graphs/stats
    monkeypatch.setattr(gen_docs, "build_repo_statistics", lambda **_: _dummy_stats())
    monkeypatch.setattr(gen_docs, "_gather_graphs", lambda **_: _dummy_graphs())
    # Avoid saving mermaid files
    monkeypatch.setattr(gen_docs, "save_mermaid", lambda diagram, path: None)

    code = gen_docs.run([
        "--repo-root", str(repo_root),
        "--no-llm",
        "--force",
    ])
    assert code == 0

    # Files written
    assert (repo_root / "wiki" / "Overview.md").exists()
    assert (repo_root / "wiki" / "Architecture.md").exists()
    assert (repo_root / "wiki" / "Adapters.md").exists()

    # Manifest created
    manifest_path = repo_root / "wiki" / "_manifest.json"
    data = json.loads(manifest_path.read_text())
    assert set(data.keys()) >= {"overview", "architecture", "adapters"}


def test_run_auto_plan_creates_extra_sections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path
    (repo_root / "wiki" / "templates").mkdir(parents=True)
    # required templates + planner ones
    (repo_root / "wiki" / "templates" / "overview.prompt").write_text("Context:\n{{context}}\n\nOverview.")
    (repo_root / "wiki" / "templates" / "architecture.prompt").write_text("Context:\n{{context}}\n\nArchitecture.")
    (repo_root / "wiki" / "templates" / "adapters.prompt").write_text("Context:\n{{context}}\n\nAdapters.")
    (repo_root / "wiki" / "templates" / "indexing.prompt").write_text("Context:\n{{context}}\n\nIndexing.")
    (repo_root / "wiki" / "templates" / "languages.prompt").write_text("Context:\n{{context}}\n\nLanguages.")

    gen_docs = _load_gen_docs_module(repo_root)
    monkeypatch.setattr(gen_docs, "build_repo_statistics", lambda **_: _dummy_stats())
    monkeypatch.setattr(gen_docs, "_gather_graphs", lambda **_: _dummy_graphs())
    monkeypatch.setattr(gen_docs, "save_mermaid", lambda diagram, path: None)

    code = gen_docs.run([
        "--repo-root", str(repo_root),
        "--no-llm",
        "--force",
        "--auto-plan",
    ])
    assert code == 0

    # Extra generated
    assert (repo_root / "wiki" / "Indexing.md").exists()
    assert (repo_root / "wiki" / "Languages.md").exists()
    # Auto plan manifest
    assert (repo_root / "wiki" / "_auto_plan.json").exists()


def test_flag_validation_conflicts() -> None:
    gen_docs = _load_gen_docs_module(Path(__file__).parent)
    # argparse uses SystemExit on parser.error
    with pytest.raises(SystemExit):
        gen_docs.run(["--no-llm", "--remote-llm"])  # invalid combination
    with pytest.raises(SystemExit):
        gen_docs.run(["--max-sections", "0"])  # non-positive
    with pytest.raises(SystemExit):
        gen_docs.run(["--max-sections", "3"])  # requires --auto-plan
