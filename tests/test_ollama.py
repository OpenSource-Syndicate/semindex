"""
Tests for Ollama LLM integration.
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.semindex.ollama_llm import OllamaLLM, OllamaError, get_available_models, pull_model


class TestOllamaLLM:
    """Test OllamaLLM class functionality."""
    
    @patch('src.semindex.ollama_llm.requests.Session.get')
    def test_initialization_success(self, mock_get):
        """Test successful initialization of OllamaLLM."""
        # Mock successful connection check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        llm = OllamaLLM(model="llama3", base_url="http://localhost:11434")
        
        assert llm.model == "llama3"
        assert llm.base_url == "http://localhost:11434"
        assert llm.temperature == 0.2  # default value
        
    @patch('src.semindex.ollama_llm.requests.Session.get')
    def test_initialization_failure(self, mock_get):
        """Test initialization failure when Ollama is not accessible."""
        # Mock failed connection check
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        with pytest.raises(OllamaError):
            OllamaLLM(model="llama3", base_url="http://invalid-url:12345")
    
    def test_build_prompt_basic(self):
        """Test basic prompt building."""
        llm = OllamaLLM.__new__(OllamaLLM)  # Create without calling __init__
        llm.base_url = "http://localhost:11434"
        llm.model = "llama3"
        
        messages = llm.build_prompt(
            system_prompt="You are a helpful assistant.",
            user_prompt="Hello, world!",
            context_chunks=None
        )
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello, world!"
    
    def test_build_prompt_with_context(self):
        """Test prompt building with context chunks."""
        llm = OllamaLLM.__new__(OllamaLLM)  # Create without calling __init__
        llm.base_url = "http://localhost:11434"
        llm.model = "llama3"
        
        messages = llm.build_prompt(
            system_prompt="You are a code assistant.",
            user_prompt="Explain this function.",
            context_chunks=["def example():\n    pass", "x = example()"]
        )
        
        # Should have system prompt, context, and user prompt
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a code assistant."
        assert messages[1]["role"] == "system"
        assert "def example():\n    pass" in messages[1]["content"]
        assert "x = example()" in messages[1]["content"]
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "Explain this function."
    
    @patch('src.semindex.ollama_llm.requests.Session')
    @patch('src.semindex.ollama_llm.OllamaLLM._check_connection', return_value=True)
    def test_generate_success(self, mock_check_connection, mock_session_class):
        """Test successful text generation."""
        # Mock the API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "content": "This is a generated response."
            }
        }

        # Create a mock session instance
        mock_session_instance = Mock()
        mock_session_instance.post.return_value = mock_response
        mock_session_class.return_value = mock_session_instance

        # Create the actual OllamaLLM instance (this will call __init__)
        llm = OllamaLLM(model="llama3")

        result = llm.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say hello.",
            context_chunks=None
        )

        assert result == "This is a generated response."
        mock_session_instance.post.assert_called_once()
    
    @patch('src.semindex.ollama_llm.requests.Session')
    @patch('src.semindex.ollama_llm.OllamaLLM._check_connection', return_value=True)
    def test_generate_api_error(self, mock_check_connection, mock_session_class):
        """Test generation when API returns error status."""
        # Mock the API response with error status
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        # Create a mock session instance
        mock_session_instance = Mock()
        mock_session_instance.post.return_value = mock_response
        mock_session_class.return_value = mock_session_instance

        # Create the actual OllamaLLM instance (this will call __init__)
        llm = OllamaLLM(model="llama3")

        with pytest.raises(OllamaError):
            llm.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt="Say hello.",
                context_chunks=None
            )


def test_get_available_models():
    """Test getting available models from Ollama."""
    with patch('src.semindex.ollama_llm.requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3", "model": "llama3:latest"},
                {"name": "codellama", "model": "codellama:7b"}
            ]
        }
        mock_get.return_value = mock_response
        
        models = get_available_models("http://localhost:11434")
        
        assert len(models) == 2
        assert models[0]["name"] == "llama3"
        assert models[1]["name"] == "codellama"


def test_pull_model():
    """Test pulling a model from Ollama."""
    with patch('src.semindex.ollama_llm.requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        result = pull_model("llama3", "http://localhost:11434")
        
        assert result is True
        mock_post.assert_called_once()