from typing import List
from .ast_py import Symbol, Chunk, extract_text_for_symbol


def build_chunks_from_symbols(source: str, symbols: List[Symbol]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for s in symbols:
        if s.kind in {"function", "method", "class"}:
            text = extract_text_for_symbol(source, s)
            if text.strip():
                chunks.append(Chunk(symbol=s, text=text))
    # Fallback: if only module exists, add file-level chunk from module
    if not chunks and symbols:
        mod = symbols[0]
        chunks.append(Chunk(symbol=mod, text=extract_text_for_symbol(source, mod)))
    return chunks
