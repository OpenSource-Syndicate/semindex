"""
Test suite for Perplexica integration and config functionality
"""
import os
import tempfile
from pathlib import Path
import toml
import pytest
from unittest.mock import Mock, patch

from semindex.config import Config, get_config
from semindex.perplexica_adapter import PerplexicaSearchAdapter
from semindex.rag import retrieve_context_by_mode, retrieve_context_with_web
from semindex.ai_implementation_assistant import AIImplementationAssistant


def test_config_creation():
    """Test that Config can be created and loaded properly"""
    # Test with a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        config_content = {
            "GENERAL": {
                "SIMILARITY_MEASURE": "cosine",
                "KEEP_ALIVE": "5m"
            },
            "MODELS": {
                "OPENAI": {
                    "API_KEY": "test-key"
                },
                "GROQ": {
                    "API_KEY": "gsk_test"
                }
            },
            "API_ENDPOINTS": {
                "PERPLEXICA": {
                    "API_BASE": "http://test-server:3000",
                    "CHAT_MODEL": {
                        "provider": "openai",
                        "name": "gpt-4-turbo"
                    },
                    "EMBEDDING_MODEL": {
                        "provider": "openai",
                        "name": "text-embedding-3-small"
                    }
                }
            }
        }
        toml.dump(config_content, f)
        config_path = f.name

    try:
        config = Config(config_path)
        
        # Test that config values can be retrieved
        assert config.get("API_ENDPOINTS.PERPLEXICA.API_BASE") == "http://test-server:3000"
        assert config.get("MODELS.OPENAI.API_KEY") == "test-key"
        assert config.get("API_ENDPOINTS.PERPLEXICA.CHAT_MODEL.name") == "gpt-4-turbo"
        
        # Test default values
        assert config.get("GENERAL.SIMILARITY_MEASURE") == "cosine"
    finally:
        os.unlink(config_path)


def test_perplexica_search_adapter_config_integration():
    """Test that PerplexicaSearchAdapter properly uses config values"""
    # Create a temporary config file with custom Perplexica settings
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        config_content = {
            "API_ENDPOINTS": {
                "PERPLEXICA": {
                    "API_BASE": "http://custom-server:4000",
                    "CHAT_MODEL": {
                        "provider": "custom_provider",
                        "name": "custom-model"
                    },
                    "EMBEDDING_MODEL": {
                        "provider": "custom_provider",
                        "name": "custom-embedding"
                    }
                }
            }
        }
        toml.dump(config_content, f)
        config_path = f.name

    try:
        # Initialize adapter with config
        adapter = PerplexicaSearchAdapter(config_path=config_path)
        
        # Check that the adapter uses config values
        assert adapter.api_base == "http://custom-server:4000"
        assert adapter.chat_model["name"] == "custom-model"
        assert adapter.embedding_model["name"] == "custom-embedding"
    finally:
        os.unlink(config_path)


def test_perplexica_search_adapter_default_config():
    """Test that PerplexicaSearchAdapter uses default values when config is not provided"""
    # Initialize adapter without config (should use default which reads from env/defaults)
    adapter = PerplexicaSearchAdapter()
    
    # Default values should be present
    assert hasattr(adapter, 'api_base')
    assert hasattr(adapter, 'chat_model')
    assert hasattr(adapter, 'embedding_model')


@patch('semindex.perplexica_adapter.requests.get')
def test_perplexica_is_available(mock_get):
    """Test the is_available method of PerplexicaSearchAdapter"""
    mock_get.return_value.status_code = 200
    
    adapter = PerplexicaSearchAdapter()
    assert adapter.is_available() == True
    
    mock_get.return_value.status_code = 404
    assert adapter.is_available() == False


@patch('semindex.perplexica_adapter.requests.post')
def test_perplexica_search_method(mock_post):
    """Test the search method of PerplexicaSearchAdapter"""
    # Mock the response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {"title": "Test Result", "pageContent": "Test content", "url": "http://example.com"}
        ]
    }
    mock_post.return_value = mock_response
    
    adapter = PerplexicaSearchAdapter()
    result = adapter.search("test query", focus_mode="webSearch")
    
    # Verify the request was made with proper payload
    assert "results" in result
    assert len(result["results"]) == 1
    
    # Check that the payload used config models by default
    call_args = mock_post.call_args
    payload = call_args[1]['json']
    
    assert "chatModel" in payload
    assert "embeddingModel" in payload
    assert "focusMode" in payload
    assert payload["focusMode"] == "webSearch"


def test_ai_implementation_assistant_config_integration():
    """Test that AIImplementationAssistant properly uses config for Perplexica adapter"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        config_content = {
            "API_ENDPOINTS": {
                "PERPLEXICA": {
                    "API_BASE": "http://test-server:5000",
                    "CHAT_MODEL": {
                        "provider": "test_provider",
                        "name": "test-model"
                    },
                    "EMBEDDING_MODEL": {
                        "provider": "test_provider",
                        "name": "test-embedding"
                    }
                }
            }
        }
        toml.dump(config_content, f)
        config_path = f.name

    try:
        # Initialize AI assistant with config
        ai_assistant = AIImplementationAssistant(index_dir=".semindex", config_path=config_path)
        
        # Check that the Perplexica adapter uses config values
        assert ai_assistant.perplexica_adapter.api_base == "http://test-server:5000"
        assert ai_assistant.perplexica_adapter.chat_model["name"] == "test-model"
    finally:
        os.unlink(config_path)


def test_config_get_perplexica_config():
    """Test the get_perplexica_config method of Config"""
    config = Config()
    
    # Test default values
    perplexica_config = config.get_perplexica_config()
    assert "api_base" in perplexica_config
    assert "chat_model" in perplexica_config
    assert "embedding_model" in perplexica_config


def test_config_get_methods():
    """Test the various get methods of Config"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        config_content = {
            "MODELS": {
                "OPENAI": {
                    "API_KEY": "test-openai-key"
                },
                "GROQ": {
                    "API_KEY": "test-groq-key"
                }
            }
        }
        toml.dump(config_content, f)
        config_path = f.name

    try:
        config = Config(config_path)
        
        # Test specific get methods
        openai_config = config.get_openai_config()
        assert openai_config["api_key"] == "test-openai-key"
        
        groq_config = config.get_groq_config()
        assert groq_config["api_key"] == "test-groq-key"
    finally:
        os.unlink(config_path)


@patch('semindex.perplexica_adapter.PerplexicaSearchAdapter.is_available')
@patch('semindex.rag.retrieve_context')
def test_retrieve_context_by_mode_config(mock_retrieve_context, mock_is_available):
    """Test retrieve_context_by_mode with config-based parameters"""
    # Mock the local search to return some results
    mock_retrieve_context.return_value = [(0.9, "test snippet")]
    mock_is_available.return_value = False  # Perplexica not available, so should fallback to local
    
    # Test with a temporary config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        config_content = {
            "API_ENDPOINTS": {
                "PERPLEXICA": {
                    "API_BASE": "http://test-server:6000"
                }
            }
        }
        toml.dump(config_content, f)
        config_path = f.name

    try:
        results = retrieve_context_by_mode(
            index_dir=".semindex",
            query="test query",
            focus_mode="webSearch",  # This would normally hit Perplexica
            config_path=config_path
        )
        
        # Should fallback to local results since Perplexica is not available
        assert len(results) == 1
        assert results[0][1] == "test snippet"
    finally:
        os.unlink(config_path)


@patch('semindex.perplexica_adapter.PerplexicaSearchAdapter.is_available')
@patch('semindex.rag.retrieve_context')
def test_retrieve_context_with_web_config(mock_retrieve_context, mock_is_available):
    """Test retrieve_context_with_web with config-based parameters"""
    # Mock the local search to return some results
    mock_retrieve_context.return_value = [(0.8, "local snippet")]
    mock_is_available.return_value = False  # Perplexica not available
    
    # Test with a config path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        config_content = {
            "API_ENDPOINTS": {
                "PERPLEXICA": {
                    "API_BASE": "http://test-server:7000"
                }
            }
        }
        toml.dump(config_content, f)
        config_path = f.name

    try:
        results = retrieve_context_with_web(
            index_dir=".semindex",
            query="test query",
            config_path=config_path
        )
        
        # Should return local results since Perplexica is not available
        assert len(results) == 1
        assert results[0][1] == "local snippet"
    finally:
        os.unlink(config_path)


def test_global_config_instance():
    """Test the global config instance functionality"""
    # Test that get_config returns the same instance
    config1 = get_config()
    config2 = get_config()
    
    assert config1 is config2
    
    # Test with a specific config path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        config_content = {"TEST": {"VALUE": "test"}}
        toml.dump(config_content, f)
        config_path = f.name

    try:
        config3 = get_config(config_path=config_path)
        assert config3 is not config1  # Different instance when path is specified
        assert config3.get("TEST.VALUE") == "test"
    finally:
        os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__])