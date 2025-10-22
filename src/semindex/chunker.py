import ast
import textwrap
from typing import List, Optional
from dataclasses import dataclass

import numpy as np

from .embed import Embedder
from .model import Chunk, Symbol
from .ast_py import extract_text_for_symbol


@dataclass
class SemanticChunk:
    """Represents a semantically coherent chunk of code"""
    symbol: Symbol
    text: str
    start_idx: int  # Start line index in the source
    end_idx: int    # End line index in the source
    embedding: Optional[np.ndarray] = None  # Embedding of this chunk


def build_chunks_from_symbols(source: str, symbols: List[Symbol]) -> List[Chunk]:
    """Build chunks based on symbol boundaries (original approach)"""
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


def build_semantic_chunks_from_symbols(
    source: str, 
    symbols: List[Symbol], 
    embedder: Embedder,
    similarity_threshold: float = 0.7
) -> List[Chunk]:
    """
    Build chunks based on semantic coherence using a CAST-like algorithm.
    
    :param source: Source code string
    :param symbols: List of parsed symbols
    :param embedder: Embedder instance for generating embeddings
    :param similarity_threshold: Threshold for semantic similarity (0.0-1.0)
    :return: List of semantically coherent chunks
    """
    lines = source.splitlines()
    
    # Early return if no symbols
    if not symbols:
        return []
    
    # Get embeddings for all symbols at once for efficiency
    symbol_texts = [extract_text_for_symbol(source, symbol) for symbol in symbols]
    
    # Filter out empty texts
    valid_symbols = []
    valid_texts = []
    for symbol, text in zip(symbols, symbol_texts):
        if text.strip():
            valid_symbols.append(symbol)
            valid_texts.append(text)
    
    if not valid_symbols:
        return []
    
    # Batch encode all symbol texts
    try:
        all_embeddings = embedder.encode(valid_texts, batch_size=32)
    except Exception as e:
        print(f"[WARN] Could not generate embeddings for symbols: {e}")
        # Fall back to basic chunking
        return build_chunks_from_symbols(source, symbols)
    
    # Convert to numpy array for vectorized operations
    embedding_matrix = np.array(all_embeddings)
    
    # Use vectorized cosine similarity calculation for efficiency
    # This calculates similarities between consecutive symbols
    if len(embedding_matrix) > 1:
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized_embeddings = embedding_matrix / norms
        
        # Calculate cosine similarities between consecutive symbols
        consecutive_similarities = np.sum(normalized_embeddings[:-1] * normalized_embeddings[1:], axis=1)
    else:
        consecutive_similarities = np.array([])
    
    # Group symbols based on similarity thresholds
    semantic_chunks = []
    current_chunk_symbols = [valid_symbols[0]] if valid_symbols else []
    current_chunk_start = valid_symbols[0].start_line - 1 if valid_symbols else 0
    current_chunk_end = valid_symbols[0].end_line if valid_symbols else 0
    
    # Process symbols in sequence
    for i in range(1, len(valid_symbols)):
        symbol = valid_symbols[i]
        
        # Check if this symbol should be grouped with the current chunk
        if i - 1 < len(consecutive_similarities) and consecutive_similarities[i - 1] >= similarity_threshold:
            # Add to current chunk
            current_chunk_symbols.append(symbol)
            current_chunk_end = max(current_chunk_end, symbol.end_line)
        else:
            # Create a chunk with the accumulated symbols
            if current_chunk_symbols:
                chunk_text = extract_chunk_text(lines, current_chunk_start, current_chunk_end)
                
                semantic_chunks.append(
                    SemanticChunk(
                        symbol=get_representative_symbol(current_chunk_symbols),
                        text=chunk_text,
                        start_idx=current_chunk_start,
                        end_idx=current_chunk_end
                    )
                )
            
            # Start a new chunk with the current symbol
            current_chunk_symbols = [symbol]
            current_chunk_start = symbol.start_line - 1  # Convert to 0-indexed
            current_chunk_end = symbol.end_line
    
    # Add the final chunk if there are symbols left
    if current_chunk_symbols:
        chunk_text = extract_chunk_text(lines, current_chunk_start, current_chunk_end)
        
        semantic_chunks.append(
            SemanticChunk(
                symbol=get_representative_symbol(current_chunk_symbols),
                text=chunk_text,
                start_idx=current_chunk_start,
                end_idx=current_chunk_end
            )
        )
    
    # Convert SemanticChunk to regular Chunk for compatibility
    chunks = []
    for sem_chunk in semantic_chunks:
        # Create a new symbol that represents the full chunk
        representative_symbol = sem_chunk.symbol
        new_symbol = Symbol(
            path=representative_symbol.path,
            name=representative_symbol.name,
            kind=representative_symbol.kind,
            start_line=sem_chunk.start_idx + 1,  # Convert back to 1-indexed
            end_line=sem_chunk.end_idx,
            signature=representative_symbol.signature,
            docstring=representative_symbol.docstring,
            imports=representative_symbol.imports,
            bases=representative_symbol.bases,
            language=representative_symbol.language,
            namespace=representative_symbol.namespace,
            symbol_type=representative_symbol.symbol_type,
        )
        chunks.append(Chunk(symbol=new_symbol, text=sem_chunk.text))
    
    return chunks


def get_representative_symbol(symbols: List[Symbol]) -> Symbol:
    """
    Get a representative symbol from a list of symbols.
    Uses the first symbol as the representative.
    
    :param symbols: List of symbols
    :return: Representative symbol
    """
    return symbols[0]


def extract_chunk_text(lines: List[str], start_idx: int, end_idx: int) -> str:
    """
    Extract text from lines between start and end indices.
    
    :param lines: List of source code lines
    :param start_idx: Start line index (0-based)
    :param end_idx: End line index (1-based)
    :return: Extracted text
    """
    # Ensure indices are within bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(lines), end_idx)
    
    return "\n".join(lines[start_idx:end_idx])


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    :param vec1: First vector
    :param vec2: Second vector
    :return: Cosine similarity score
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def average_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Calculate the average of a list of embeddings.
    
    :param embeddings: List of embedding vectors
    :return: Average embedding vector
    """
    if not embeddings:
        return np.array([])
    
    # Stack embeddings and calculate mean
    stacked = np.stack(embeddings)
    return np.mean(stacked, axis=0)
