from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


MERMAID_BLOCK_PATTERN = re.compile(r"```mermaid\s+([\s\S]+?)```", re.IGNORECASE)
NODE_PATTERN = re.compile(r"\bgraph\s+(TB|TD|LR|RL|BT)\b", re.IGNORECASE)
EDGE_PATTERN = re.compile(r"-->|===|---|-.->|==>")


class MermaidValidationError(ValueError):
    """Raised when a Mermaid diagram fails validation."""


@dataclass(slots=True)
class MermaidFragment:
    source_path: Path
    code: str


def extract_mermaid_blocks(markdown: str) -> Iterable[MermaidFragment]:
    for match in MERMAID_BLOCK_PATTERN.finditer(markdown):
        yield MermaidFragment(source_path=Path(""), code=match.group(1))


def normalize_mermaid(code: str) -> str:
    lines = [line.rstrip() for line in code.strip().splitlines() if line.strip()]
    return "\n".join(lines)


def validate_mermaid(code: str) -> None:
    if not code:
        raise MermaidValidationError("Mermaid diagram is empty")
    header = NODE_PATTERN.search(code)
    if not header:
        raise MermaidValidationError("Missing graph direction header (e.g., 'graph TD')")
    if "[" not in code and "(" not in code:
        raise MermaidValidationError("No nodes detected in Mermaid diagram")
    edges = EDGE_PATTERN.findall(code)
    if not edges:
        raise MermaidValidationError("No edges detected in Mermaid diagram")


def lint_mermaid_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    for fragment in extract_mermaid_blocks(text):
        try:
            validate_mermaid(fragment.code)
        except MermaidValidationError as exc:
            relative = path
            errors.append(f"{relative}: {exc}")
    return errors


def lint_mermaid_directory(directory: Path) -> list[str]:
    errors: list[str] = []
    for path in directory.rglob("*.md"):
        errors.extend(lint_mermaid_file(path))
    return errors


def save_mermaid(graph_code: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(graph_code.strip() + "\n", encoding="utf-8")


def export_mermaid(graph_code: str, destination: Path) -> None:
    data = {"diagram": graph_code}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2), encoding="utf-8")
