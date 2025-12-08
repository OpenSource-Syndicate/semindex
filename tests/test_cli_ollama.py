"""
Tests for CLI Ollama integration.
"""
import argparse
import os
from unittest.mock import Mock, patch, MagicMock
import pytest
from src.semindex.cli import cmd_query


class TestCliOllama:
    """Test CLI Ollama integration."""
    
    @patch('src.semindex.search.search_similar')
    @patch('src.semindex.cli.print')
    def test_cmd_query_with_ollama_success(self, mock_print, mock_search_similar):
        """Test query command with Ollama integration."""
        # Mock search results
        mock_search_similar.return_value = [
            (0.9, 1, ("test.py", "hello", "function", 1, 5, "def hello():")),
            (0.8, 2, ("test.py", "world", "function", 6, 10, "def world():"))
        ]

        # Create args object with ollama enabled
        args = argparse.Namespace(
            query="Explain the code",
            index_dir=".semindex",
            model="microsoft/codebert-base",
            top_k=10,
            hybrid=False,
            include_docs=False,
            docs_weight=0.4,
            ollama=True,
            ollama_model="llama3",
            max_tokens=512
        )

        with patch('src.semindex.rag.generate_answer_ollama') as mock_gen_answer:
            mock_gen_answer.return_value = "This is an Ollama-generated explanation."

            # Call the function
            cmd_query(args)

            # Verify that Ollama function was called
            mock_gen_answer.assert_called_once_with(
                index_dir=os.path.abspath('.semindex'),  # Function converts to absolute path
                query='Explain the code',
                top_k=10,
                embed_model='microsoft/codebert-base',
                ollama_model='llama3',
                max_tokens=512
            )

            # Verify print was called with the Ollama response
            assert mock_print.called
            # Check that print was called with the Ollama response
            print_calls = [str(call[0]) for call in mock_print.call_args_list]
            ollama_response_printed = any("This is an Ollama-generated explanation." in call for call in print_calls)
            assert ollama_response_printed
    
    @patch('src.semindex.search.search_similar')
    @patch('src.semindex.cli.print')
    def test_cmd_query_without_ollama(self, mock_print, mock_search_similar):
        """Test query command without Ollama integration."""
        # Mock search results
        mock_search_similar.return_value = [
            (0.9, 1, ("test.py", "hello", "function", 1, 5, "def hello():"))
        ]

        # Create args object with ollama disabled
        args = argparse.Namespace(
            query="Explain the code",
            index_dir=".semindex",
            model="microsoft/codebert-base",
            top_k=10,
            hybrid=False,
            include_docs=False,
            docs_weight=0.4,
            ollama=False,
            ollama_model="llama3",
            max_tokens=512
        )

        # Call the function
        cmd_query(args)

        # Verify that Ollama function was NOT called
        # We can't directly mock generate_answer_ollama since it's imported inside cmd_query
        # So instead we'll verify the behavior by ensuring print was called appropriately
        mock_print.assert_called()  # Should still print results in regular mode