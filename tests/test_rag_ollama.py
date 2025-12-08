"""
Tests for Ollama integration in RAG module.
"""
import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.semindex.rag import generate_answer_ollama


class TestRagOllama:
    """Test RAG Ollama integration."""
    
    @patch('src.semindex.rag.retrieve_context')
    @patch('src.semindex.ollama_llm.OllamaLLM')
    def test_generate_answer_ollama_success(self, mock_ollama_class, mock_retrieve_context):
        """Test successful Ollama answer generation."""
        # Mock context retrieval
        mock_retrieve_context.return_value = [
            (0.9, "# File: test.py:1-5\ndef hello():\n    print('Hello')"),
            (0.8, "# File: test.py:6-10\ndef world():\n    print('World')")
        ]
        
        # Mock Ollama instance
        mock_ollama_instance = Mock()
        mock_ollama_instance.generate.return_value = "This is the explanation for the code."
        mock_ollama_class.return_value = mock_ollama_instance
        
        # Call the function
        result = generate_answer_ollama(
            index_dir="/fake/path",
            query="Explain the hello function",
            top_k=5,
            ollama_model="llama3"
        )
        
        # Verify the result
        assert result == "This is the explanation for the code."
        
        # Verify Ollama was called with correct parameters
        mock_ollama_class.assert_called_once()
        mock_ollama_instance.generate.assert_called_once()
        
        # Get the call arguments
        call_args = mock_ollama_instance.generate.call_args
        assert call_args is not None
        
        # Verify context chunks were passed correctly
        context_chunks = call_args.kwargs.get('context_chunks', [])
        assert len(context_chunks) == 2
        assert "# File: test.py:1-5" in context_chunks[0]
        assert "def hello():" in context_chunks[0]
    
    @patch('src.semindex.rag.retrieve_context')
    def test_generate_answer_ollama_import_error(self, mock_retrieve_context):
        """Test Ollama answer generation when Ollama is not available."""
        # Mock context retrieval
        mock_retrieve_context.return_value = [
            (0.9, "# File: test.py:1-5\ndef hello():\n    print('Hello')")
        ]

        # Test import error by temporarily making the module unavailable
        with patch.dict('sys.modules', {'src.semindex.ollama_llm': None}):
            with pytest.raises(ImportError):
                generate_answer_ollama(
                    index_dir="/fake/path",
                    query="Explain the hello function",
                    top_k=5
                )
    
    @patch('src.semindex.rag.retrieve_context')
    @patch('src.semindex.ollama_llm.OllamaLLM')
    def test_generate_answer_ollama_ollama_error(self, mock_ollama_class, mock_retrieve_context):
        """Test Ollama answer generation when Ollama throws an error."""
        # Mock context retrieval
        mock_retrieve_context.return_value = [
            (0.9, "# File: test.py:1-5\ndef hello():\n    print('Hello')")
        ]
        
        # Mock Ollama to raise an OllamaError
        mock_ollama_instance = Mock()
        mock_ollama_instance.generate.side_effect = Exception("Ollama server error")
        mock_ollama_class.return_value = mock_ollama_instance
        
        # Should raise OllamaError (wrapped exception)
        with pytest.raises(Exception, match="Ollama server error"):
            generate_answer_ollama(
                index_dir="/fake/path",
                query="Explain the hello function",
                top_k=5
            )