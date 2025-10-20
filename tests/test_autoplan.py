from pathlib import Path
from semindex.docs import MermaidGraph
from semindex.docs.autoplan import generate_plan


def test_generate_plan_minimal_repo(tmp_path: Path) -> None:
    graphs = {
        "pipeline": MermaidGraph(title="t", diagram="graph TD\n A-->B"),
        "adapter": MermaidGraph(title="a", diagram="graph TD\n A-->B", metadata={"adapter_count": 0}),
    }
    stats = {"indexed_files": 0, "symbols": 0}

    plan = generate_plan(repo_root=tmp_path, graphs=graphs, stats=stats)

    names = [s.name for s in plan]
    assert "overview" in names  # always present
    assert "architecture" not in names  # requires symbols
    assert "adapters" not in names  # requires adapters
    assert "indexing" not in names  # requires indexed files
    assert "languages" not in names  # requires language stats


def test_generate_plan_rich_repo(tmp_path: Path) -> None:
    graphs = {
        "pipeline": MermaidGraph(title="p", diagram="graph TD\n A-->B"),
        "module": MermaidGraph(title="m", diagram="graph TD\n A-->B"),
        "adapter": MermaidGraph(title="a", diagram="graph TD\n A-->B", metadata={"adapter_count": 3}),
    }
    stats = {
        "indexed_files": 42,
        "symbols": 1000,
        "symbols_by_language": {"python": 900, "js": 100},
        "docs_indexed": [{"package": "semindex", "pages": 10}],
    }

    plan = generate_plan(repo_root=tmp_path, graphs=graphs, stats=stats)
    names = [s.name for s in plan]
    # Overview always present
    assert "overview" in names
    # With symbols -> include architecture
    assert "architecture" in names
    # With adapters -> include adapters
    assert "adapters" in names
    # With indexed files -> include indexing
    assert "indexing" in names
    # With languages -> include languages
    assert "languages" in names

    # Ensure extra_context carries planner metadata
    section = next(s for s in plan if s.name == "languages")
    assert section.extra_context and "planner" in section.extra_context
