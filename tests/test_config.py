"""
Test suite for configuration module
"""
import os
import tempfile
from pathlib import Path
import toml
import pytest

from semindex.config import Config, get_config


def test_config_initialization():
    """Test basic config initialization"""
    config = Config()
    
    # Should have default values
    assert config.get("GENERAL.SIMILARITY_MEASURE") == "cosine"
    assert config.get("GENERAL.KEEP_ALIVE") == "5m"


def test_config_file_creation():
    """Test that config file is created if it doesn't exist"""
    # Use a temporary file path
    with tempfile.NamedTemporaryFile(delete=False, suffix='.toml') as temp_file:
        temp_path = temp_file.name
    
    # Remove the file so config will create it
    os.unlink(temp_path)
    
    # Initialize config with this path - it should NOT automatically create the file,
    # but it will have default values loaded
    config = Config(temp_path)
    
    # The config object should have default values
    assert config.get("GENERAL.SIMILARITY_MEASURE") == "cosine"
    
    # But the file itself is not automatically created until save_config is called
    # So we need to verify it's created when save is called
    config.save_config(config.config_data, temp_path)
    assert Path(temp_path).exists()
    
    # Clean up
    os.unlink(temp_path)
    
    # Verify we can read values
    assert config.get("GENERAL.SIMILARITY_MEASURE") == "cosine"
    
    # Clean up
    if Path(temp_path).exists():
        os.unlink(temp_path)


def test_config_get_and_set():
    """Test getting and setting config values"""
    config = Config()
    
    # Test getting a value that doesn't exist (should return default)
    assert config.get("NONEXISTENT.KEY", "default") == "default"
    
    # Test setting and getting a value
    config.set("TEST.KEY", "test_value")
    assert config.get("TEST.KEY") == "test_value"
    
    # Test nested setting
    config.set("TEST.NESTED.KEY", "nested_value")
    assert config.get("TEST.NESTED.KEY") == "nested_value"


def test_config_perplexica_settings():
    """Test Perplexica-specific config settings"""
    config = Config()
    
    # Test default Perplexica config
    perplexica_config = config.get_perplexica_config()
    assert "api_base" in perplexica_config
    assert "chat_model" in perplexica_config
    assert "embedding_model" in perplexica_config
    
    # Verify expected structure
    assert isinstance(perplexica_config["api_base"], str)
    assert isinstance(perplexica_config["chat_model"], dict)
    assert isinstance(perplexica_config["embedding_model"], dict)


def test_config_model_settings():
    """Test various model config settings"""
    config = Config()
    
    # Test that various model configs can be retrieved
    openai_config = config.get_openai_config()
    groq_config = config.get_groq_config()
    ollama_config = config.get_ollama_config()
    
    # Verify structure
    assert "api_key" in openai_config
    assert "api_key" in groq_config
    assert "api_url" in ollama_config


def test_config_with_custom_file():
    """Test config with a custom file"""
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        config_content = {
            "GENERAL": {
                "SIMILARITY_MEASURE": "dot_product",
                "KEEP_ALIVE": "10m"
            },
            "MODELS": {
                "OPENAI": {
                    "API_KEY": "custom-openai-key"
                },
                "GROQ": {
                    "API_KEY": "custom-groq-key"
                }
            },
            "API_ENDPOINTS": {
                "PERPLEXICA": {
                    "API_BASE": "http://custom-server:8000",
                    "CHAT_MODEL": {
                        "provider": "custom_provider",
                        "name": "custom-chat-model"
                    },
                    "EMBEDDING_MODEL": {
                        "provider": "custom_provider",
                        "name": "custom-embedding-model"
                    }
                }
            }
        }
        toml.dump(config_content, f)
        config_path = f.name

    try:
        # Load config from file
        config = Config(config_path)
        
        # Verify custom values
        assert config.get("GENERAL.SIMILARITY_MEASURE") == "dot_product"
        assert config.get("GENERAL.KEEP_ALIVE") == "10m"
        assert config.get("MODELS.OPENAI.API_KEY") == "custom-openai-key"
        assert config.get("MODELS.GROQ.API_KEY") == "custom-groq-key"
        assert config.get("API_ENDPOINTS.PERPLEXICA.API_BASE") == "http://custom-server:8000"
        assert config.get("API_ENDPOINTS.PERPLEXICA.CHAT_MODEL.name") == "custom-chat-model"
        assert config.get("API_ENDPOINTS.PERPLEXICA.EMBEDDING_MODEL.name") == "custom-embedding-model"
    finally:
        # Clean up
        os.unlink(config_path)


def test_config_save_functionality():
    """Test that config can be saved to file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        config_content = {
            "TEST": {
                "VALUE": "test_value"
            }
        }
        toml.dump(config_content, f)
        config_path = f.name

    try:
        # Load existing config
        config = Config(config_path)
        
        # Modify config
        config.set("MODIFIED.KEY", "modified_value")
        
        # Save to same file
        config.save_config(config.config_data, config_path)
        
        # Reload and verify changes
        config2 = Config(config_path)
        assert config2.get("MODIFIED.KEY") == "modified_value"
    finally:
        os.unlink(config_path)


def test_config_file_not_found():
    """Test config with non-existent file path"""
    # Use a path that doesn't exist
    non_existent_path = "/non/existent/path/config.toml"
    config = Config(non_existent_path)
    
    # Should have default values since the file doesn't exist
    assert config.get("GENERAL.SIMILARITY_MEASURE") == "cosine"
    assert config.config_data  # Should have default values loaded


def test_config_default_config_structure():
    """Test that default config has expected structure"""
    config = Config()
    
    # Verify the structure exists
    assert "GENERAL" in config.config_data
    assert "MODELS" in config.config_data
    assert "API_ENDPOINTS" in config.config_data
    
    # Verify API_ENDPOINTS has PERPLEXICA section
    api_endpoints = config.config_data.get("API_ENDPOINTS", {})
    assert "PERPLEXICA" in api_endpoints
    assert api_endpoints["PERPLEXICA"]["API_BASE"] == "http://localhost:3000"


def test_config_model_specific_getters():
    """Test model-specific getters"""
    config = Config()
    
    # Test all model getters have expected structure
    openai_config = config.get_openai_config()
    assert isinstance(openai_config, dict)
    assert "api_key" in openai_config
    
    groq_config = config.get_groq_config()
    assert isinstance(groq_config, dict)
    assert "api_key" in groq_config
    
    ollama_config = config.get_ollama_config()
    assert isinstance(ollama_config, dict)
    assert "api_url" in ollama_config


def test_config_perplexica_config_structure():
    """Test that get_perplexica_config returns expected structure"""
    config = Config()
    perplexica_config = config.get_perplexica_config()
    
    # Verify structure
    assert "api_base" in perplexica_config
    assert "chat_model" in perplexica_config
    assert "embedding_model" in perplexica_config
    
    # Verify model config structure
    assert "provider" in perplexica_config["chat_model"]
    assert "name" in perplexica_config["chat_model"]
    assert "provider" in perplexica_config["embedding_model"]
    assert "name" in perplexica_config["embedding_model"]


if __name__ == "__main__":
    pytest.main([__file__])