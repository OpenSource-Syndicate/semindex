"""
Tests for semantic chunking functionality.
"""
import tempfile
import os
from unittest.mock import Mock
import numpy as np
from semindex.chunker import (
    build_semantic_chunks_from_symbols,
    cosine_similarity,
    average_embeddings,
    get_representative_symbol
)
from semindex.ast_py import Symbol


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    # Two identical vectors should have similarity 1.0
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(vec1, vec2) == 1.0
    
    # Two orthogonal vectors should have similarity 0.0
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 1.0, 0.0])
    assert abs(cosine_similarity(vec1, vec2)) < 1e-10  # Essentially 0
    
    # Two opposite vectors should have similarity -1.0
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([-1.0, 0.0, 0.0])
    assert cosine_similarity(vec1, vec2) == -1.0
    
    # Test with zero vector
    vec1 = np.array([0.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(vec1, vec2) == 0.0


def test_average_embeddings():
    """Test averaging of embeddings."""
    embeddings = [
        np.array([1.0, 2.0, 3.0]),
        np.array([3.0, 4.0, 5.0]),
        np.array([5.0, 6.0, 7.0])
    ]
    avg = average_embeddings(embeddings)
    expected = np.array([3.0, 4.0, 5.0])  # Mean of each dimension
    np.testing.assert_array_equal(avg, expected)
    
    # Test with empty list
    assert len(average_embeddings([])) == 0
    
    # Test with single embedding
    single = [np.array([1.0, 2.0, 3.0])]
    result = average_embeddings(single)
    np.testing.assert_array_equal(result, single[0])


def test_get_representative_symbol():
    """Test getting representative symbol from list."""
    sym1 = Symbol(
        path="test.py",
        name="func1",
        kind="function",
        start_line=1,
        end_line=5,
        signature="def func1():",
        docstring="First function",
        imports=[],
        bases=[]
    )
    
    sym2 = Symbol(
        path="test.py",
        name="func2",
        kind="function",
        start_line=6,
        end_line=10,
        signature="def func2():",
        docstring="Second function",
        imports=[],
        bases=[]
    )
    
    symbols = [sym1, sym2]
    representative = get_representative_symbol(symbols)
    
    # Should return the first symbol
    assert representative == sym1


def test_build_semantic_chunks_from_symbols():
    """Test that the function can be called without raising an exception."""
    # Create mock symbols
    symbols = [
        Symbol(
            path="test.py",
            name="func1",
            kind="function",
            start_line=1,
            end_line=5,
            signature="def func1():",
            docstring="First function",
            imports=[],
            bases=[]
        ),
        Symbol(
            path="test.py",
            name="func2",
            kind="function",
            start_line=6,
            end_line=10,
            signature="def func2():",
            docstring="Second function (similar to func1)",
            imports=[],
            bases=[]
        ),
        Symbol(
            path="test.py",
            name="class1",
            kind="class",
            start_line=11,
            end_line=15,
            signature="class Class1:",
            docstring="A class",
            imports=[],
            bases=[]
        )
    ]
    
    # Create mock embedder
    mock_embedder = Mock()
    # Properly mocking the encode method to return the expected numpy array format
    def mock_encode(texts):
        # Create embeddings as a 2D numpy array: (num_texts, embedding_dim)
        num_texts = len(texts)
        embedding_dim = 5  # arbitrary dimension
        # Return array of shape (num_texts, embedding_dim)
        embeddings = np.random.random((num_texts, embedding_dim)).astype(np.float32)
        return embeddings
    
    mock_embedder.encode = Mock(side_effect=mock_encode)
    
    source = """def func1():
    '''First function'''
    pass

def func2():
    '''Second function (similar to func1)'''
    pass

class Class1:
    '''A class'''
    pass
"""
    
    # Just make sure the function doesn't crash
    chunks = build_semantic_chunks_from_symbols(
        source=source,
        symbols=symbols,
        embedder=mock_embedder,
        similarity_threshold=0.5
    )
    
    # Should return a list (even if empty due to low threshold)
    assert isinstance(chunks, list)


def test_build_semantic_chunks_with_very_low_threshold():
    """Test semantic chunking with very low threshold (everything grouped)."""
    symbols = [
        Symbol(
            path="test.py",
            name="func1",
            kind="function",
            start_line=1,
            end_line=5,
            signature="def func1():",
            docstring="First function",
            imports=[],
            bases=[]
        ),
        Symbol(
            path="test.py",
            name="func2",
            kind="function",
            start_line=6,
            end_line=10,
            signature="def func2():",
            docstring="Second function",
            imports=[],
            bases=[]
        )
    ]
    
    # Create mock embedder with identical embeddings (max similarity)
    mock_embedder = Mock()
    # The encode method should return a 2D array where each row is an embedding for each text
    def mock_encode(texts):
        # Return the same embedding for all texts
        num_texts = len(texts)
        embedding_dim = 3  # Fixed dimension
        # Return array of shape (num_texts, embedding_dim) with identical values
        embeddings = np.ones((num_texts, embedding_dim), dtype=np.float32)
        return embeddings
    
    mock_embedder.encode = Mock(side_effect=mock_encode)
    
    source = """
def func1():
    '''First function'''
    pass

def func2():
    '''Second function'''
    pass
"""
    
    chunks = build_semantic_chunks_from_symbols(
        source=source,
        symbols=symbols,
        embedder=mock_embedder,
        similarity_threshold=0.0  # Accept any similarity
    )
    
    # With identical embeddings and low threshold, should potentially group
    assert len(chunks) > 0


def test_build_semantic_chunks_with_very_high_threshold():
    """Test semantic chunking with very high threshold (nothing grouped)."""
    symbols = [
        Symbol(
            path="test.py",
            name="func1",
            kind="function",
            start_line=1,
            end_line=5,
            signature="def func1():",
            docstring="First function",
            imports=[],
            bases=[]
        ),
        Symbol(
            path="test.py",
            name="func2",
            kind="function",
            start_line=6,
            end_line=10,
            signature="def func2():",
            docstring="Second function",
            imports=[],
            bases=[]
        )
    ]
    
    # Create mock embedder with very different embeddings (low similarity)
    mock_embedder = Mock()
    # The encode method should return a 2D array where each row is an embedding for each text
    def mock_encode(texts):
        # Return different embeddings for each text
        num_texts = len(texts)
        embedding_dim = 3
        # Create embeddings with different values for each text
        embeddings = np.random.random((num_texts, embedding_dim)).astype(np.float32)
        # Make them distinctive by scaling differently
        for i in range(num_texts):
            embeddings[i] *= (i + 1)  # Scale each embedding differently
        return embeddings
    
    mock_embedder.encode = Mock(side_effect=mock_encode)
    
    source = """
def func1():
    '''First function'''
    pass

def func2():
    '''Second function'''
    pass
"""
    
    chunks = build_semantic_chunks_from_symbols(
        source=source,
        symbols=symbols,
        embedder=mock_embedder,
        similarity_threshold=0.99  # Very high threshold
    )
    
    # With different embeddings and high threshold, 
    # each symbol should likely form its own chunk
    assert len(chunks) > 0