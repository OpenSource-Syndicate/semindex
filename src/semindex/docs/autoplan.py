from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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


@dataclass(slots=True)
class PlanningRule:
    """A rule that determines whether a section should be included in the plan."""

    name: str
    title: str
    template: str
    output: str
    required_graphs: Sequence[str]
    condition: Callable[[Dict[str, object]], bool]
    rationale_fn: Callable[[Dict[str, object]], str]
    context_fn: Callable[[Dict[str, object]], Dict[str, object]]


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
    """Extract top languages by symbol count from statistics."""
    languages = stats.get("symbols_by_language") or {}
    if not isinstance(languages, dict):
        return []
    ordered = sorted(languages.items(), key=lambda item: item[1], reverse=True)
    return ordered[:limit]


def _validate_graphs(graphs: Dict[str, MermaidGraph], required: Sequence[str]) -> bool:
    """Check if all required graphs are present."""
    return all(name in graphs for name in required)


def _query_index(db_path: str, query: str) -> List[Tuple]:
    """Execute a query against the index database."""
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        result = cur.execute(query).fetchall()
        con.close()
        return result
    except Exception:
        return []


def _discover_key_modules(db_path: str, limit: int = 5) -> List[str]:
    """Discover key modules by symbol count."""
    query = """
    SELECT path, COUNT(*) as count
    FROM symbols
    WHERE kind = 'module' OR symbol_type = 'module'
    GROUP BY path
    ORDER BY count DESC
    LIMIT ?
    """
    results = _query_index(db_path, f"SELECT path FROM symbols WHERE kind = 'module' OR symbol_type = 'module' LIMIT {limit}")
    return [row[0] for row in results]


def _discover_key_classes(db_path: str, limit: int = 10) -> List[Tuple[str, str]]:
    """Discover key classes by their presence and complexity."""
    query = f"""
    SELECT name, path
    FROM symbols
    WHERE kind = 'class'
    ORDER BY (SELECT COUNT(*) FROM symbols s2 WHERE s2.namespace = symbols.name) DESC
    LIMIT {limit}
    """
    results = _query_index(db_path, query)
    return results


def _discover_key_functions(db_path: str, limit: int = 10) -> List[Tuple[str, str]]:
    """Discover key functions by their usage (callers/callees)."""
    query = f"""
    SELECT s.name, s.path
    FROM symbols s
    WHERE s.kind = 'function'
    ORDER BY (SELECT COUNT(*) FROM calls WHERE caller_id = s.id OR callee_symbol_id = s.id) DESC
    LIMIT {limit}
    """
    results = _query_index(db_path, query)
    return results


def _discover_patterns(db_path: str) -> List[str]:
    """Discover architectural patterns from the codebase."""
    patterns = []
    
    # Check for test files
    test_count = len(_query_index(db_path, "SELECT COUNT(*) FROM files WHERE path LIKE '%test%'"))
    if test_count > 0:
        patterns.append("testing")
    
    # Check for configuration files
    config_count = len(_query_index(db_path, "SELECT COUNT(*) FROM files WHERE path LIKE '%config%' OR path LIKE '%.yaml%' OR path LIKE '%.json%'"))
    if config_count > 0:
        patterns.append("configuration")
    
    # Check for API/handler patterns
    api_count = len(_query_index(db_path, "SELECT COUNT(*) FROM symbols WHERE name LIKE '%handler%' OR name LIKE '%route%' OR name LIKE '%endpoint%'"))
    if api_count > 0:
        patterns.append("api")
    
    # Check for database patterns
    db_count = len(_query_index(db_path, "SELECT COUNT(*) FROM symbols WHERE name LIKE '%query%' OR name LIKE '%model%' OR name LIKE '%schema%'"))
    if db_count > 0:
        patterns.append("data_layer")
    
    return patterns


def _build_default_rules(graphs: Dict[str, MermaidGraph]) -> List[PlanningRule]:
    """Build the default set of planning rules.
    
    Args:
        graphs: Available graphs to extract metadata from
    """
    def get_adapter_count(stats: Dict[str, object]) -> int:
        """Extract adapter count from graphs metadata."""
        adapter_graph = graphs.get("adapter")
        if adapter_graph and adapter_graph.metadata:
            return int(adapter_graph.metadata.get("adapter_count", 0) or 0)
        return 0
    
    return [
        # Overview section: always included
        PlanningRule(
            name="overview",
            title="Project Overview",
            template="overview.prompt",
            output="Overview.md",
            required_graphs=("pipeline",),
            condition=lambda _: True,  # Always included
            rationale_fn=lambda _: "Baseline executive summary for all repositories.",
            context_fn=lambda stats: {
                "indexed_files": int(stats.get("indexed_files", 0) or 0),
                "symbols": int(stats.get("symbols", 0) or 0),
            },
        ),
        # Architecture section: requires indexed symbols
        PlanningRule(
            name="architecture",
            title="Architecture Deep-Dive",
            template="architecture.prompt",
            output="Architecture.md",
            required_graphs=("module", "pipeline"),
            condition=lambda stats: int(stats.get("symbols", 0) or 0) > 0,
            rationale_fn=lambda _: "Codebase contains indexed modules requiring architectural documentation.",
            context_fn=lambda stats: {
                "indexed_files": int(stats.get("indexed_files", 0) or 0),
                "symbols": int(stats.get("symbols", 0) or 0),
            },
        ),
        # Adapters section: requires registered adapters
        PlanningRule(
            name="adapters",
            title="Language Adapter Guide",
            template="adapters.prompt",
            output="Adapters.md",
            required_graphs=("adapter",),
            condition=lambda stats: get_adapter_count(stats) > 0,
            rationale_fn=lambda _: "Repository registers language adapters that should be documented.",
            context_fn=lambda stats: {
                "adapter_count": get_adapter_count(stats),
            },
        ),
        # Indexing section: requires indexed files
        PlanningRule(
            name="indexing",
            title="Indexing Workflow Playbook",
            template="indexing.prompt",
            output="Indexing.md",
            required_graphs=("pipeline",),
            condition=lambda stats: int(stats.get("indexed_files", 0) or 0) > 0,
            rationale_fn=lambda _: "Indexed files detected; documenting ingestion and maintenance guidance.",
            context_fn=lambda stats: {
                "indexed_files": int(stats.get("indexed_files", 0) or 0),
                "docs_indexed": stats.get("docs_indexed", []),
            },
        ),
        # Languages section: requires multiple languages
        PlanningRule(
            name="languages",
            title="Language Coverage Report",
            template="languages.prompt",
            output="Languages.md",
            required_graphs=("adapter",),
            condition=lambda stats: len(_top_languages(stats)) > 0,
            rationale_fn=lambda _: "Multiple languages detected in the index warranting dedicated coverage.",
            context_fn=lambda stats: {
                "languages": [
                    {"language": lang, "symbols": count} for lang, count in _top_languages(stats)
                ],
            },
        ),
    ]


def generate_plan(
    *,
    repo_root: Path,
    graphs: Dict[str, MermaidGraph],
    stats: Dict[str, object],
    max_sections: Optional[int] = None,
    rules: Optional[List[PlanningRule]] = None,
    index_dir: Optional[str] = None,
) -> List[PlannedSection]:
    """Produce a list of documentation sections tailored to the repository.
    
    Generates sections based on:
    1. Default rules (overview, architecture, etc.)
    2. Index-discovered content (key modules, classes, functions, patterns)
    
    Args:
        repo_root: Root directory of the repository
        graphs: Dictionary of available MermaidGraph objects
        stats: Repository statistics from graph builder
        max_sections: Maximum number of sections to include (None = unlimited)
        rules: Custom planning rules (None = use defaults)
        index_dir: Path to index directory for querying discovered content
    
    Returns:
        List of PlannedSection objects in priority order
    """
    if rules is None:
        rules = _build_default_rules(graphs)
    
    plan: List[PlannedSection] = []
    
    # First, add sections from rules
    for rule in rules:
        # Check if we've hit the max sections limit
        if max_sections is not None and len(plan) >= max_sections:
            break
        
        # Check if all required graphs are available
        if not _validate_graphs(graphs, rule.required_graphs):
            continue
        
        # Check if the condition is met
        if not rule.condition(stats):
            continue
        
        # Build the section
        rationale = rule.rationale_fn(stats)
        context = rule.context_fn(stats)
        
        section = PlannedSection(
            name=rule.name,
            title=rule.title,
            template=rule.template,
            output=rule.output,
            graphs=tuple(name for name in rule.required_graphs if name in graphs),
            extra_context=_planner_metadata(rationale=rationale, extra=context),
            rationale=rationale,
        )
        plan.append(section)
    
    # Then, add index-discovered sections if index_dir is provided
    if index_dir:
        from ..store import DB_NAME
        db_path = str(Path(index_dir) / DB_NAME)
        
        # Discover key modules
        key_modules = _discover_key_modules(db_path)
        if key_modules and (max_sections is None or len(plan) < max_sections):
            section = PlannedSection(
                name="key_modules",
                title="Key Modules",
                template="key_modules.prompt",
                output="KeyModules.md",
                graphs=tuple(name for name in ("module",) if name in graphs),
                extra_context=_planner_metadata(
                    rationale="Core modules discovered from codebase analysis.",
                    extra={"modules": key_modules},
                ),
                rationale="Core modules discovered from codebase analysis.",
            )
            plan.append(section)
        
        # Discover key classes
        key_classes = _discover_key_classes(db_path)
        if key_classes and (max_sections is None or len(plan) < max_sections):
            section = PlannedSection(
                name="key_classes",
                title="Key Classes",
                template="key_classes.prompt",
                output="KeyClasses.md",
                graphs=tuple(name for name in ("module",) if name in graphs),
                extra_context=_planner_metadata(
                    rationale="Important classes discovered through structural analysis.",
                    extra={"classes": [{"name": name, "path": path} for name, path in key_classes]},
                ),
                rationale="Important classes discovered through structural analysis.",
            )
            plan.append(section)
        
        # Discover key functions
        key_functions = _discover_key_functions(db_path)
        if key_functions and (max_sections is None or len(plan) < max_sections):
            section = PlannedSection(
                name="key_functions",
                title="Key Functions",
                template="key_functions.prompt",
                output="KeyFunctions.md",
                graphs=tuple(name for name in ("module",) if name in graphs),
                extra_context=_planner_metadata(
                    rationale="Critical functions identified by call graph analysis.",
                    extra={"functions": [{"name": name, "path": path} for name, path in key_functions]},
                ),
                rationale="Critical functions identified by call graph analysis.",
            )
            plan.append(section)
        
        # Discover patterns
        patterns = _discover_patterns(db_path)
        if patterns and (max_sections is None or len(plan) < max_sections):
            section = PlannedSection(
                name="patterns",
                title="Architectural Patterns",
                template="patterns.prompt",
                output="Patterns.md",
                graphs=tuple(name for name in ("module", "pipeline") if name in graphs),
                extra_context=_planner_metadata(
                    rationale="Architectural patterns detected in the codebase.",
                    extra={"patterns": patterns},
                ),
                rationale="Architectural patterns detected in the codebase.",
            )
            plan.append(section)
    
    return plan
