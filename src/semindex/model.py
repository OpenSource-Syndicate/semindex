from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Symbol:
    path: str
    name: str
    kind: str  # e.g. module|class|function|method
    start_line: int
    end_line: int
    signature: str
    docstring: Optional[str]
    imports: List[str]
    bases: List[str]
    language: Optional[str] = None
    namespace: Optional[str] = None
    symbol_type: Optional[str] = None

    def __hash__(self) -> int:
        # Hash only immutable identifying fields
        return hash(
            (
                self.path,
                self.name,
                self.kind,
                self.start_line,
                self.end_line,
                self.signature,
                self.language,
            )
        )


@dataclass
class Chunk:
    symbol: Symbol
    text: str


@dataclass
class ChunkingConfig:
    method: str  # symbol | semantic | language-specific identifier
    similarity_threshold: float = 0.7
