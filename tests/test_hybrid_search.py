"""
Tests for hybrid search functionality.
"""
import os
import tempfile
import pytest
import numpy as np
from unittest.mock import Mock, patch
from semindex.hybrid_search import reciprocal_rank_fusion, hybrid_search
from semindex.keyword_search import KeywordResult


def test_reciprocal_rank_fusion_basic():
    """Test basic reciprocal rank fusion functionality."""
    # Mock dense results: (score, symbol_id, symbol_info)
    dense_results = [
        (0.9, 1, ("path1", "func1", "function", 1, 10, "def func1()")),
        (0.8, 2, ("path2", "func2", "function", 10, 20, "def func2()")),
        (0.7, 3, ("path3", "func3", "function", 20, 30, "def func3()")),
    ]
    
    # Mock keyword results: KeywordResult objects
    keyword_results = [
        KeywordResult(2, "path2", "func2", "function", 2.5),  # Same as dense result 1
        KeywordResult(4, "path4", "func4", "function", 2.0),  # New
        KeywordResult(1, "path1", "func1", "function", 1.8),  # Same as dense result 0
    ]
    
    # Apply RRF
    result = reciprocal_rank_fusion(dense_results, keyword_results, top_k=3)
    
    # Check that we have results
    assert len(result) <= 3
    assert all(len(item) == 2 for item in result)  # (symbol_id, rrf_score)
    
    # Check that symbol IDs from both result sets are included
    symbol_ids = [item[0] for item in result]
    assert 1 in symbol_ids
    assert 2 in symbol_ids
    assert 4 in symbol_ids  # New from keyword search


def test_reciprocal_rank_fusion_empty_inputs():
    """Test RRF with empty inputs."""
    result = reciprocal_rank_fusion([], [], top_k=5)
    assert result == []
    
    # Only dense results
    dense_results = [(0.9, 1, ("path1", "func1", "function", 1, 10, "def func1()"))]
    result = reciprocal_rank_fusion(dense_results, [], top_k=5)
    assert len(result) == 1
    assert result[0][0] == 1
    
    # Only keyword results
    keyword_results = [KeywordResult(1, "path1", "func1", "function", 2.5)]
    result = reciprocal_rank_fusion([], keyword_results, top_k=5)
    assert len(result) == 1
    assert result[0][0] == 1


def test_reciprocal_rank_fusion_with_custom_k():
    """Test RRF with custom k value."""
    dense_results = [(0.9, 1, ("path1", "func1", "function", 1, 10, "def func1()"))]
    keyword_results = [KeywordResult(1, "path1", "func1", "function", 2.5)]
    
    result = reciprocal_rank_fusion(dense_results, keyword_results, top_k=1, k=10.0)
    assert len(result) <= 1


@patch('semindex.hybrid_search.faiss_search')
@patch('semindex.hybrid_search.KeywordSearcher')
def test_hybrid_search_integration(mock_keyword_searcher, mock_faiss_search):
    """Test the full hybrid search integration."""
    # Mock FAISS search results
    mock_faiss_search.return_value = [
        (0.9, 1, ("path1", "func1", "function", 1, 10, "def func1()")),
        (0.8, 2, ("path2", "func2", "function", 10, 20, "def func2()")),
    ]
    
    # Mock keyword searcher
    mock_keyword_instance = Mock()
    mock_keyword_instance.search.return_value = [
        KeywordResult(2, "path2", "func2", "function", 2.5),
        KeywordResult(3, "path3", "func3", "function", 2.0),
    ]
    mock_keyword_searcher.return_value = mock_keyword_instance
    
    # Mock query vector
    query_vec = np.array([[0.1, 0.2, 0.3, 0.4]])
    
    # Mock database connection
    with patch('sqlite3.connect') as mock_connect:
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = ("path1", "func1", "function", 1, 10, "def func1()")
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Call hybrid search
        index_dir = tempfile.mkdtemp()
        results = hybrid_search(index_dir, query_vec, "test query", top_k=2)
        
        # Check that both search methods were called
        assert mock_faiss_search.called
        assert mock_keyword_instance.search.called
        
        # Results should be returned
        assert isinstance(results, list)
        assert len(results) <= 2  # top_k is 2


def test_hybrid_search_empty_results():
    """Test hybrid search with empty results from one source."""
    with patch('semindex.hybrid_search.faiss_search') as mock_faiss:
        mock_faiss.return_value = []
        
        with patch('semindex.hybrid_search.KeywordSearcher') as mock_keyword_searcher:
            mock_keyword_instance = Mock()
            mock_keyword_instance.search.return_value = []
            mock_keyword_searcher.return_value = mock_keyword_instance
            
            # Mock database connection
            with patch('sqlite3.connect') as mock_connect:
                mock_conn = Mock()
                mock_connect.return_value.__enter__.return_value = mock_conn
                
                query_vec = np.array([[0.1, 0.2, 0.3, 0.4]])
                index_dir = tempfile.mkdtemp()
                
                results = hybrid_search(index_dir, query_vec, "test query", top_k=5)
                
                # Should return an empty list
                assert results == []


def test_reciprocal_rank_fusion_edge_cases():
    """Test RRF with various edge cases."""
    # Test with duplicate entries
    dense_results = [
        (0.9, 1, ("path1", "func1", "function", 1, 10, "def func1()")),
        (0.8, 2, ("path2", "func2", "function", 10, 20, "def func2()")),
    ]
    
    keyword_results = [
        KeywordResult(1, "path1", "func1", "function", 2.5),  # Same as dense result 0
        KeywordResult(3, "path3", "func3", "function", 2.0),
    ]
    
    result = reciprocal_rank_fusion(dense_results, keyword_results, top_k=5)
    assert len(result) <= 3  # Should have 3 unique results
    symbol_ids = [item[0] for item in result]
    assert 1 in symbol_ids  # Common result
    assert 2 in symbol_ids  # From dense only
    assert 3 in symbol_ids  # From keyword only