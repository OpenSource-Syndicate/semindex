from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from semindex.docs import (
    MermaidGraph,
    PlannedSection,
    build_adapter_graph,
    build_module_graph,
    build_pipeline_graph,
    build_repo_statistics,
    export_graph_metadata,
    generate_plan,
)
from semindex.docs.mermaid import save_mermaid, validate_mermaid
from semindex.local_llm import LocalLLM
from semindex.remote_llm import OpenAICompatibleLLM, resolve_groq_config


@dataclass(slots=True)
class SectionSpec:
    name: str
    title: str
    template: str
    output: str
    graphs: Sequence[str]
    extra_context: Optional[Dict[str, object]] = None


GRAPH_BUILDERS = {
    "module": build_module_graph,
    "adapter": build_adapter_graph,
    "pipeline": build_pipeline_graph,
}


SECTION_SPECS: Sequence[SectionSpec] = (
    SectionSpec(
        name="overview",
        title="Project Overview",
        template="overview.prompt",
        output="Overview.md",
        graphs=("pipeline",),
    ),
    SectionSpec(
        name="architecture",
        title="Architecture Deep-Dive",
        template="architecture.prompt",
        output="Architecture.md",
        graphs=("module", "pipeline"),
    ),
    SectionSpec(
        name="adapters",
        title="Language Adapter Guide",
        template="adapters.prompt",
        output="Adapters.md",
        graphs=("adapter",),
    ),
)


def _get_unified_template() -> str:
    """Return a unified template that works for all section types."""
    return """# {title}

## Overview
This section covers {section_name} aspects of the project.

## Details
{details}

## Context
```json
{context_json}
```
"""


def _load_template(repo_root: Path, template_name: str) -> str:
    """Load template from file, or use unified template as fallback."""
    template_path = repo_root / "wiki" / "templates" / template_name
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    # Return unified template as fallback
    return _get_unified_template()


def _hash_payload(payload: Dict[str, object]) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _load_manifest(manifest_path: Path) -> Dict[str, Dict[str, object]]:
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def _save_manifest(manifest_path: Path, manifest: Dict[str, Dict[str, object]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _slugify(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    slug = "-".join(filter(None, slug.split("-")))
    return slug or "diagram"


def _ensure_wiki(repo_root: Path) -> Path:
    wiki_path = repo_root / "wiki"
    wiki_path.mkdir(exist_ok=True)
    (wiki_path / "diagrams").mkdir(exist_ok=True)
    (wiki_path / "templates").mkdir(exist_ok=True)
    return wiki_path


def _gather_graphs(repo_root: Path, index_dir: Path) -> Dict[str, MermaidGraph]:
    graphs = {
        "module": build_module_graph(repo_root=repo_root, index_dir=index_dir),
        "adapter": build_adapter_graph(),
        "pipeline": build_pipeline_graph(),
    }
    return graphs


def _compose_prompt(template: str, context: Dict[str, object], spec_name: str = "", spec_title: str = "") -> str:
    """Compose a prompt by filling in template variables.
    
    Supports both:
    - Old style: {{context}} placeholder replacement
    - New style: {title}, {section_name}, {details}, {context_json} f-string formatting
    """
    context_json = json.dumps(context, indent=2, ensure_ascii=False)
    
    # Try old-style replacement first
    if "{{context}}" in template:
        return template.replace("{{context}}", context_json)
    
    # Use new-style f-string formatting
    details = context.get("description", "No details available.")
    try:
        return template.format(
            title=spec_title or context.get("title", "Section"),
            section_name=spec_name or "project",
            details=details,
            context_json=context_json,
        )
    except KeyError:
        # If template has unexpected placeholders, return as-is
        return template


def _generate_content(
    llm: Optional[object],
    system_prompt: str,
    user_prompt: str,
    *,
    fallback: str,
    max_tokens: int,
) -> str:
    if llm is None:
        return fallback
    try:
        if isinstance(llm, LocalLLM):
            text = llm.generate(system_prompt, user_prompt, max_tokens=max_tokens)
        else:
            text = llm.generate(system_prompt, user_prompt, max_tokens=max_tokens)
        if text:
            return text
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"[WARN] LLM generation failed: {exc}", file=sys.stderr)
    return fallback


def _default_fallback(title: str, context: Dict[str, object]) -> str:
    lines = [f"# {title}", "", "This section could not be generated automatically."]
    lines.append("Context snapshot:")
    lines.append("```json")
    lines.append(json.dumps(context, indent=2, ensure_ascii=False))
    lines.append("```")
    return "\n".join(lines)


def _write_section(
    wiki_path: Path,
    spec: SectionSpec,
    body: str,
    graphs: Sequence[MermaidGraph],
) -> None:
    output_path = wiki_path / spec.output
    parts = [body.strip()]
    for graph in graphs:
        parts.append(graph.to_markdown())
    output = "\n\n".join(parts).strip() + "\n"
    output_path.write_text(output, encoding="utf-8")
    for graph in graphs:
        slug = _slugify(f"{spec.name}-{graph.title}")
        save_mermaid(graph.diagram, wiki_path / "diagrams" / f"{slug}.mmd")


def _build_context(
    repo_root: Path,
    spec: SectionSpec,
    graphs: Dict[str, MermaidGraph],
    stats: Dict[str, object],
) -> Dict[str, object]:
    graph_meta = {name: graphs[name].metadata for name in spec.graphs}
    context: Dict[str, object] = {
        "title": spec.title,
        "stats": stats,
        "graphs": graph_meta,
    }
    if spec.extra_context:
        context.update(spec.extra_context)
    return context


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate developer documentation using a local or remote LLM.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root (default: cwd)")
    parser.add_argument("--index-dir", type=Path, default=Path(".semindex"), help="Index directory")
    parser.add_argument("--section", nargs="*", help="Limit generation to specific sections (resolved after planning)")
    parser.add_argument("--force", action="store_true", help="Regenerate sections even if the manifest is unchanged")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM generation and use fallback text")
    parser.add_argument("--remote-llm", action="store_true", help="Force the use of a remote OpenAI-compatible LLM and fail if unavailable")
    parser.add_argument("--remote-api-key", help="Override the remote LLM API key (otherwise use environment variables)")
    parser.add_argument("--remote-api-base", help="Override the remote LLM API base URL")
    parser.add_argument("--remote-model", help="Override the remote LLM model name")
    parser.add_argument("--auto-plan", action="store_true", help="Dynamically select documentation sections based on repository analysis")
    parser.add_argument("--max-sections", type=int, help="Maximum sections to emit when using --auto-plan")
    parser.add_argument("--max-tokens", type=int, default=768, help="Maximum tokens per section for the local LLM")
    parser.add_argument("--dump-graphs", action="store_true", help="Export graph metadata to wiki/graphs.json")
    args = parser.parse_args(argv)

    if args.no_llm and args.remote_llm:
        parser.error("--remote-llm cannot be combined with --no-llm")
    if args.max_sections is not None and args.max_sections <= 0:
        parser.error("--max-sections must be a positive integer")
    if args.max_sections is not None and not args.auto_plan:
        parser.error("--max-sections requires --auto-plan")

    repo_root = args.repo_root.resolve()
    index_dir = (repo_root / args.index_dir).resolve()

    wiki_path = _ensure_wiki(repo_root)
    manifest_path = wiki_path / "_manifest.json"
    manifest = _load_manifest(manifest_path)
    planner_manifest_path = wiki_path / "_auto_plan.json"

    stats = build_repo_statistics(repo_root=repo_root, index_dir=index_dir)
    graphs = _gather_graphs(repo_root=repo_root, index_dir=index_dir)

    if args.dump_graphs:
        metadata_path = wiki_path / "graphs.json"
        metadata_path.write_text(export_graph_metadata(graphs.values()), encoding="utf-8")

    available_specs: List[SectionSpec]
    planner_sections: List[PlannedSection] = []
    if args.auto_plan:
        planner_sections = generate_plan(
            repo_root=repo_root,
            graphs=graphs,
            stats=stats,
            max_sections=args.max_sections,
        )
        available_specs = [
            SectionSpec(
                name=section.name,
                title=section.title,
                template=section.template,
                output=section.output,
                graphs=tuple(section.graphs),
                extra_context=section.extra_context,
            )
            for section in planner_sections
        ]
        if not available_specs:
            print(
                "[WARN] Auto planner did not return any sections; falling back to default SECTION_SPECS.",
                file=sys.stderr,
            )
            available_specs = list(SECTION_SPECS)
            planner_sections = []
        else:
            plan_payload = [
                {
                    "name": section.name,
                    "title": section.title,
                    "template": section.template,
                    "output": section.output,
                    "graphs": list(section.graphs),
                    "rationale": section.rationale,
                    "extra_context": section.extra_context or {},
                }
                for section in planner_sections
            ]
            planner_manifest_path.write_text(
                json.dumps(plan_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[info] Auto planner selected {len(available_specs)} section(s)", file=sys.stderr)
    else:
        available_specs = list(SECTION_SPECS)

    name_to_spec = {spec.name: spec for spec in available_specs}
    requested_sections = list(args.section or [])
    if requested_sections:
        missing = [name for name in requested_sections if name not in name_to_spec]
        if missing:
            parser.error(f"Unknown section(s): {', '.join(missing)}")
        selected_sections = [name_to_spec[name] for name in requested_sections]
    else:
        selected_sections = available_specs

    llm: Optional[object] = None
    remote_error: Optional[str] = None
    if not args.no_llm:
        remote_config = resolve_groq_config(
            api_key=args.remote_api_key,
            api_base=args.remote_api_base,
            model=args.remote_model,
        )
        if remote_config is not None:
            try:
                llm = OpenAICompatibleLLM(remote_config)
                print("[info] Using remote Groq-compatible LLM", file=sys.stderr)
            except Exception as exc:
                print(f"[WARN] Failed to initialize remote LLM: {exc}", file=sys.stderr)
                remote_error = str(exc)
        elif args.remote_llm:
            remote_error = (
                "Remote LLM requested but no API key was provided. Set GROQ_API_KEY or pass --remote-api-key."
            )
        if llm is None:
            if args.remote_llm:
                message = "[ERROR] Remote LLM requested via --remote-llm but initialization failed."
                if remote_error:
                    message += f" Reason: {remote_error}"
                print(message, file=sys.stderr)
                return 1
            try:
                llm = LocalLLM()
                print("[info] Using local GGUF LLM", file=sys.stderr)
            except FileNotFoundError as exc:
                print(f"[WARN] Local LLM unavailable: {exc}", file=sys.stderr)
            except Exception as exc:  # pragma: no cover - defensive guard
                print(f"[WARN] Failed to initialize LocalLLM: {exc}", file=sys.stderr)

    did_write = False
    for spec in selected_sections:
        context = _build_context(repo_root, spec, graphs, stats)
        template_text = _load_template(repo_root, spec.template)
        user_prompt = _compose_prompt(template_text, context, spec_name=spec.name, spec_title=spec.title)
        payload_hash = _hash_payload({
            "template": template_text,
            "context": context,
        })

        manifest_entry = manifest.get(spec.name)
        if not args.force and manifest_entry and manifest_entry.get("payload_hash") == payload_hash:
            print(f"[SKIP] {spec.name} unchanged")
            continue

        fallback = _default_fallback(spec.title, context)
        system_prompt = (
            "You are an expert technical writer creating in-depth developer documentation "
            "for the semindex project. Emphasize architecture, data flow, and actionable guidance."
        )
        body = _generate_content(
            llm,
            system_prompt,
            user_prompt,
            fallback=fallback,
            max_tokens=args.max_tokens,
        )

        used_graphs = [graphs[name] for name in spec.graphs]
        for graph in used_graphs:
            validate_mermaid(graph.diagram)
        _write_section(wiki_path, spec, body, used_graphs)

        manifest[spec.name] = {
            "payload_hash": payload_hash,
            "output": spec.output,
        }
        did_write = True
        print(f"[OK] Generated {spec.output}")

    if did_write:
        _save_manifest(manifest_path, manifest)

    return 0


if __name__ == "__main__":
    sys.exit(run())
