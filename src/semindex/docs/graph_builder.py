from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..languages import available_adapters, ensure_default_adapters
from ..store import ensure_db
from .mermaid import normalize_mermaid, validate_mermaid


DB_NAME = "semindex.db"


@dataclass(slots=True)
class MermaidGraph:
    title: str
    diagram: str
    description: str = ""
    metadata: Optional[Dict[str, object]] = None

    def to_markdown(self) -> str:
        block = [f"### {self.title}"]
        if self.description:
            block.append(self.description.strip())
        block.append("```mermaid")
        block.append(self.diagram.strip())
        block.append("```")
        return "\n\n".join(block)


def _sanitize_id(label: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label)
    if not sanitized:
        sanitized = "node"
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def _load_symbols(index_dir: Path) -> Sequence[Tuple[str, str]]:
    db_path = index_dir / DB_NAME
    if not db_path.exists():
        return []
    ensure_db(str(db_path))
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT DISTINCT path, imports FROM symbols WHERE kind = 'module' OR symbol_type = 'module'"
        )
        return [(row[0], row[1] or "") for row in cur.fetchall()]
    finally:
        con.close()


def build_module_graph(
    repo_root: Path | str = Path.cwd(),
    index_dir: Path | str = Path(".semindex"),
    *,
    max_edges: int = 80,
) -> MermaidGraph:
    repo_root = Path(repo_root).resolve()
    index_dir = Path(index_dir)
    symbols = _load_symbols(index_dir)
    nodes: Set[str] = set()
    edges: Set[Tuple[str, str]] = set()
    for path, imports in symbols:
        module = _path_to_module(repo_root, Path(path))
        nodes.add(module)
        for raw in (s.strip() for s in imports.split(",")):
            if not raw:
                continue
            nodes.add(raw)
            edges.add((module, raw))
    if not nodes:
        diagram = _fallback_module_graph()
    else:
        if not edges:
            diagram = _fallback_module_edges()
        else:
            diagram = _render_graph(nodes, edges, max_edges=max_edges)
    diagram = normalize_mermaid(diagram)
    validate_mermaid(diagram)
    return MermaidGraph(
        title="Module dependency graph",
        description="Shows module-level imports discovered from the index database.",
        diagram=diagram,
        metadata={
            "node_count": len(nodes),
            "edge_count": len(edges),
            "index_dir": str(index_dir),
        },
    )


def build_adapter_graph() -> MermaidGraph:
    ensure_default_adapters()
    adapters = list(available_adapters())
    if not adapters:
        diagram = "graph TD\n    A[No adapters registered]"
    else:
        lines = ["graph TD"]
        for adapter in adapters:
            adapter_id = _sanitize_id(adapter.name)
            lines.append(f"    {adapter_id}[{adapter.name}]")
            for ext in sorted(set(adapter.file_extensions)):
                ext_id = _sanitize_id(f"{adapter.name}_{ext}")
                lines.append(f"    {adapter_id} --> {ext_id}[`{ext}`]")
        diagram = "\n".join(lines)
    diagram = normalize_mermaid(diagram)
    validate_mermaid(diagram)
    return MermaidGraph(
        title="Adapter registry",
        description="Language adapters and their file-extension coverage.",
        diagram=diagram,
        metadata={"adapter_count": len(adapters)},
    )


def build_pipeline_graph() -> MermaidGraph:
    diagram = normalize_mermaid(
        """
        graph LR
            A[Repository crawler] --> B[Language adapter]
            B --> C[Chunker]
            C --> D[Embedder]
            D --> E[FAISS index]
            D --> F[SQLite metadata]
            E --> G[Vector search]
            F --> G
            F --> H[Keyword search]
            D --> I[Docs indexer]
            I --> J[Docs FAISS]
            J --> G
        """
    )
    validate_mermaid(diagram)
    return MermaidGraph(
        title="Indexing pipeline",
        description="High-level data flow for indexing code, metadata, and documentation.",
        diagram=diagram,
    )


def build_repo_statistics(
    repo_root: Path | str = Path.cwd(),
    index_dir: Path | str = Path(".semindex"),
) -> Dict[str, object]:
    repo_root = Path(repo_root).resolve()
    index_dir = Path(index_dir)
    db_path = index_dir / DB_NAME
    stats: Dict[str, object] = {
        "repo_root": str(repo_root),
        "index_dir": str(index_dir),
        "indexed_files": 0,
        "symbols": 0,
    }
    if not db_path.exists():
        return stats
    ensure_db(str(db_path))
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM files")
        stats["indexed_files"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM symbols")
        stats["symbols"] = cur.fetchone()[0]
        cur.execute(
            "SELECT language, COUNT(*) FROM symbols WHERE language IS NOT NULL GROUP BY language"
        )
        stats["symbols_by_language"] = {row[0]: row[1] for row in cur.fetchall()}
        cur.execute(
            "SELECT package, COUNT(*) FROM doc_pages GROUP BY package ORDER BY COUNT(*) DESC"
        )
        stats["docs_indexed"] = [
            {"package": row[0], "pages": row[1]} for row in cur.fetchall()
        ]
    finally:
        con.close()
    return stats


def _render_graph(nodes: Iterable[str], edges: Iterable[Tuple[str, str]], *, max_edges: int) -> str:
    lines = ["graph TD"]
    limited_edges = list(edges)[:max_edges]
    nodes_seen: Set[str] = set()
    for src, dst in limited_edges:
        src_id = _sanitize_id(src)
        dst_id = _sanitize_id(dst)
        lines.append(f"    {src_id}[{src}] --> {dst_id}[{dst}]")
        nodes_seen.add(src)
        nodes_seen.add(dst)
    remaining = set(nodes) - nodes_seen
    for node in sorted(remaining)[: max(0, max_edges - len(limited_edges))]:
        node_id = _sanitize_id(node)
        lines.append(f"    {node_id}[{node}]")
    if len(list(edges)) > max_edges:
        lines.append("    meta[More edges not shown]")
    return "\n".join(lines)


def _path_to_module(repo_root: Path, file_path: Path) -> str:
    try:
        rel = file_path.resolve().relative_to(repo_root)
    except Exception:
        rel = file_path
    stem_parts = list(rel.with_suffix("").parts)
    if not stem_parts:
        return file_path.stem
    return ".".join(part for part in stem_parts if part)


def _fallback_module_graph() -> str:
    return "\n".join(
        [
            "graph TD",
            "    A[No modules indexed yet] --> B[Run semindex index to populate modules]",
        ]
    )


def _fallback_module_edges() -> str:
    return "\n".join(
        [
            "graph TD",
            "    A[Module import relationships not detected] --> B[Run indexing to populate imports]",
        ]
    )


def export_graph_metadata(graphs: Sequence[MermaidGraph]) -> str:
    payload = [
        {
            "title": g.title,
            "description": g.description,
            "metadata": g.metadata or {},
        }
        for g in graphs
    ]
    return json.dumps(payload, indent=2)
