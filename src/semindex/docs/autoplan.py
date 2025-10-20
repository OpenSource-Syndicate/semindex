from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .graph_builder import MermaidGraph


@dataclass(slots=True)
class PlannedSection:
    """Represents a documentation section proposed by the auto planner."""

    name: str
    title: str
    template: str
    output: str
    graphs: Sequence[str]
    extra_context: Optional[Dict[str, object]] = None
    rationale: Optional[str] = None


def _planner_metadata(
    *,
    rationale: str,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "mode": "auto",
        "rationale": rationale,
    }
    if extra:
        payload.update(extra)
    return {"planner": payload}


def _top_languages(stats: Dict[str, object], limit: int = 5) -> Sequence[Tuple[str, int]]:
    languages = stats.get("symbols_by_language") or {}
    if not isinstance(languages, dict):
        return []
    ordered = sorted(languages.items(), key=lambda item: item[1], reverse=True)
    return ordered[:limit]


def generate_plan(
    *,
    repo_root: Path,
    graphs: Dict[str, MermaidGraph],
    stats: Dict[str, object],
    max_sections: Optional[int] = None,
) -> List[PlannedSection]:
    """Produce a list of documentation sections tailored to the repository."""

    plan: List[PlannedSection] = []

    def add_section(section: PlannedSection) -> None:
        if max_sections is not None and len(plan) >= max_sections:
            return
        plan.append(section)

    graph_names = set(graphs.keys())
    indexed_files = int(stats.get("indexed_files", 0) or 0)
    symbol_count = int(stats.get("symbols", 0) or 0)

    # Always include a high-level overview.
    rationale = "Baseline executive summary for all repositories."
    add_section(
        PlannedSection(
            name="overview",
            title="Project Overview",
            template="overview.prompt",
            output="Overview.md",
            graphs=tuple(name for name in ("pipeline",) if name in graph_names),
            extra_context=_planner_metadata(
                rationale=rationale,
                extra={
                    "indexed_files": indexed_files,
                    "symbols": symbol_count,
                },
            ),
            rationale=rationale,
        )
    )

    # Architecture deep-dive is useful once code has been indexed.
    if symbol_count:
        rationale = "Codebase contains indexed modules requiring architectural documentation."
        add_section(
            PlannedSection(
                name="architecture",
                title="Architecture Deep-Dive",
                template="architecture.prompt",
                output="Architecture.md",
                graphs=tuple(
                    name
                    for name in ("module", "pipeline")
                    if name in graph_names
                ),
                extra_context=_planner_metadata(
                    rationale=rationale,
                    extra={
                        "indexed_files": indexed_files,
                        "symbols": symbol_count,
                    },
                ),
                rationale=rationale,
            )
        )

    # Adapter coverage section adds value if adapters exist.
    adapter_graph = graphs.get("adapter")
    adapter_count = 0
    if adapter_graph and adapter_graph.metadata:
        adapter_count = int(adapter_graph.metadata.get("adapter_count", 0) or 0)
    if adapter_count:
        rationale = "Repository registers language adapters that should be documented."
        add_section(
            PlannedSection(
                name="adapters",
                title="Language Adapter Guide",
                template="adapters.prompt",
                output="Adapters.md",
                graphs=("adapter",),
                extra_context=_planner_metadata(
                    rationale=rationale,
                    extra={
                        "adapter_count": adapter_count,
                    },
                ),
                rationale=rationale,
            )
        )

    # Indexing workflow section when data has been ingested.
    if indexed_files:
        rationale = "Indexed files detected; documenting ingestion and maintenance guidance."
        add_section(
            PlannedSection(
                name="indexing",
                title="Indexing Workflow Playbook",
                template="indexing.prompt",
                output="Indexing.md",
                graphs=tuple(name for name in ("pipeline",) if name in graph_names),
                extra_context=_planner_metadata(
                    rationale=rationale,
                    extra={
                        "indexed_files": indexed_files,
                        "docs_indexed": stats.get("docs_indexed", []),
                    },
                ),
                rationale=rationale,
            )
        )

    # Language coverage section if multiple languages detected.
    top_languages = _top_languages(stats)
    if top_languages:
        rationale = "Multiple languages detected in the index warranting dedicated coverage."
        add_section(
            PlannedSection(
                name="languages",
                title="Language Coverage Report",
                template="languages.prompt",
                output="Languages.md",
                graphs=tuple(name for name in ("adapter",) if name in graph_names),
                extra_context=_planner_metadata(
                    rationale=rationale,
                    extra={
                        "languages": [
                            {"language": lang, "symbols": count} for lang, count in top_languages
                        ],
                    },
                ),
                rationale=rationale,
            )
        )

    return plan
