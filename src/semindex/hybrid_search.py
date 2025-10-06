"""
Hybrid search module that combines dense vector search and keyword search
using Reciprocal Rank Fusion (RRF).
"""
import os
from typing import List, Tuple, Optional
import numpy as np
from .store import search as faiss_search, get_symbol_by_id
from .keyword_search import KeywordSearcher


def reciprocal_rank_fusion(
    dense_results: List[Tuple[float, int, tuple]], 
    keyword_results: List, 
    top_k: int = 10,
    k: float = 60.0  # RRF smoothing constant
) -> List[Tuple[float, int, tuple]]:
    """
    Apply Reciprocal Rank Fusion to combine dense and keyword search results.
    
    :param dense_results: Results from dense vector search (score, symbol_id, symbol_info)
    :param keyword_results: Results from keyword search (KeywordResult objects)
    :param top_k: Number of top results to return
    :param k: RRF smoothing constant (default 60 is commonly used)
    :return: Combined results sorted by RRF score
    """
    # Create a dictionary to track combined scores for each symbol ID
    combined_scores = {}
    
    # Add scores from dense results
    for rank, (score, symbol_id, symbol_info) in enumerate(dense_results):
        # Use the rank (position) for RRF calculation
        rrf_score = 1.0 / (k + rank)
        combined_scores[symbol_id] = combined_scores.get(symbol_id, 0) + rrf_score
    
    # Add scores from keyword results
    for rank, keyword_result in enumerate(keyword_results):
        symbol_id = keyword_result.symbol_id
        rrf_score = 1.0 / (k + rank)
        combined_scores[symbol_id] = combined_scores.get(symbol_id, 0) + rrf_score
    
    # Sort by combined RRF scores in descending order
    sorted_results = sorted(
        combined_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:top_k]
    
    # Fetch the full symbol information for the top results
    # This requires access to the database connection
    return sorted_results


def hybrid_search(
    index_dir: str,
    query_vec: np.ndarray,
    query_text: str,
    top_k: int = 10,
    dense_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> List[Tuple[float, int, tuple]]:
    """
    Perform hybrid search combining dense vector and keyword search.
    
    :param index_dir: Directory where index files are stored
    :param query_vec: Query vector for dense search
    :param query_text: Query text for keyword search
    :param top_k: Number of top results to return
    :param dense_weight: Weight for dense search results (default 0.7)
    :param keyword_weight: Weight for keyword search results (default 0.3)
    :return: Combined search results
    """
    from .store import DB_NAME
    import sqlite3
    
    # Paths to index files
    db_path = os.path.join(index_dir, DB_NAME)
    
    # Perform dense search (vector similarity)
    con = sqlite3.connect(db_path)
    try:
        dense_results = faiss_search(
            os.path.join(index_dir, "index.faiss"), 
            con, 
            query_vec, 
            top_k * 2  # Get more results for fusion
        )
    finally:
        con.close()
    
    # Perform keyword search
    keyword_searcher = KeywordSearcher(index_dir)
    keyword_results = keyword_searcher.search(query_text, top_k=top_k * 2)
    
    # Apply Reciprocal Rank Fusion to combine results
    fused_results = reciprocal_rank_fusion(
        dense_results[:top_k*2], 
        keyword_results[:top_k*2], 
        top_k=top_k
    )
    
    # Fetch full symbol information for the top results
    con = sqlite3.connect(db_path)
    try:
        final_results = []
        for symbol_id, rrf_score in fused_results:
            symbol_info = get_symbol_by_id(con, symbol_id)
            if symbol_info:
                final_results.append((rrf_score, symbol_id, symbol_info))
    finally:
        con.close()
    
    return final_results