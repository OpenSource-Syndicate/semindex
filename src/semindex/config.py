"""
Configuration module for semindex
Handles loading and accessing configuration from config.toml file
"""
import os
import toml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration class for loading and accessing settings from config.toml"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            # If a specific config path is provided, use it even if it doesn't exist yet
            self.config_path = config_path
            self.config_data = self._load_config()
        else:
            # If no config path provided, find or create a default one
            self.config_path = self._find_config_file()
            self.config_data = self._load_config()
    
    def _find_config_file(self) -> str:
        """Find the config.toml file in common locations"""
        possible_paths = [
            "config.toml",
            "conf.toml",
            ".semindex/config.toml",
            str(Path.home() / ".semindex" / "config.toml"),
            str(Path.cwd() / "config.toml")
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        # If no config file is found, create a default one in current directory
        default_config = self._get_default_config()
        self.save_config(default_config, "config.toml")
        return "config.toml"
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            "GENERAL": {
                "SIMILARITY_MEASURE": "cosine",
                "KEEP_ALIVE": "5m"
            },
            "PERFORMANCE": {
                "MAX_WORKERS": 4,
                "BATCH_SIZE": 16,
                "CACHE_SIZE": 10000,
                "MAX_MEMORY_MB": 2048,
                "ENABLE_CACHING": True,
                "ENABLE_PARALLEL_PROCESSING": True
            },
            "MODELS": {
                "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",  # Recommended faster model
                "CODE_LLM_MODEL": "microsoft/Phi-3-mini-4k-instruct",  # Recommended for code tasks
                "GENERAL_LLM_MODEL": "microsoft/Phi-3-mini-4k-instruct"  # Recommended for general tasks
            },
            "MODELS": {
                "OPENAI": {
                    "API_KEY": ""
                },
                "GROQ": {
                    "API_KEY": ""
                },
                "ANTHROPIC": {
                    "API_KEY": ""
                },
                "GEMINI": {
                    "API_KEY": ""
                },
                "CUSTOM_OPENAI": {
                    "API_KEY": "",
                    "API_URL": "",
                    "MODEL_NAME": ""
                },
                "OLLAMA": {
                    "API_URL": "http://host.docker.internal:11434"
                },
                "DEEPSEEK": {
                    "API_KEY": ""
                },
                "AIMLAPI": {
                    "API_KEY": ""
                },
                "LM_STUDIO": {
                    "API_URL": ""
                },
                "LEMONADE": {
                    "API_URL": "",
                    "API_KEY": ""
                }
            },
            "API_ENDPOINTS": {
                "SEARXNG": "",
                "PERPLEXICA": {
                    "API_BASE": "http://localhost:3000",
                    "CHAT_MODEL": {
                        "provider": "openai",
                        "name": "gpt-3.5-turbo"
                    },
                    "EMBEDDING_MODEL": {
                        "provider": "openai", 
                        "name": "text-embedding-ada-002"
                    }
                }
            }
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from the TOML file"""
        # Get the default configuration as base
        default_config = self._get_default_config()
        
        if not os.path.exists(self.config_path):
            # If the config file doesn't exist, use default config
            print(f"Config file {self.config_path} not found, using default config")
            return default_config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                loaded_config = toml.load(f)
            
            # Merge loaded config with defaults (loaded config takes precedence)
            merged_config = self._merge_configs(default_config, loaded_config)
            return merged_config
        except Exception as e:
            print(f"Error loading config file {self.config_path}: {e}")
            # Return default config if there's an error
            return default_config
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Merge default config with loaded config, with loaded taking precedence"""
        import copy
        merged = copy.deepcopy(default)
        
        for key, value in loaded.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = self._merge_configs(merged[key], value)
            else:
                # Use loaded value if it's not a nested dict or if key doesn't exist in defaults
                merged[key] = value
        
        return merged
    
    def save_config(self, config_data: Dict[str, Any], path: str = None) -> None:
        """Save configuration to a TOML file"""
        save_path = path or self.config_path
        with open(save_path, 'w', encoding='utf-8') as f:
            toml.dump(config_data, f)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using a dot-separated path
        Example: get("MODELS.OPENAI.API_KEY")
        """
        keys = key_path.split('.')
        value = self.config_data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using a dot-separated path
        Example: set("MODELS.OPENAI.API_KEY", "new_api_key")
        """
        keys = key_path.split('.')
        config_ref = self.config_data
        
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        config_ref[keys[-1]] = value
    
    def get_perplexica_config(self) -> Dict[str, Any]:
        """Get Perplexica-specific configuration"""
        return {
            "api_base": self.get("API_ENDPOINTS.PERPLEXICA.API_BASE", "http://localhost:3000"),
            "chat_model": self.get("API_ENDPOINTS.PERPLEXICA.CHAT_MODEL", {
                "provider": "openai",
                "name": "gpt-3.5-turbo"
            }),
            "embedding_model": self.get("API_ENDPOINTS.PERPLEXICA.EMBEDDING_MODEL", {
                "provider": "openai", 
                "name": "text-embedding-ada-002"
            })
        }
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI-specific configuration"""
        return {
            "api_key": self.get("MODELS.OPENAI.API_KEY", ""),
        }
    
    def get_groq_config(self) -> Dict[str, Any]:
        """Get Groq-specific configuration"""
        return {
            "api_key": self.get("MODELS.GROQ.API_KEY", ""),
        }
    
    def get_ollama_config(self) -> Dict[str, Any]:
        """Get Ollama-specific configuration"""
        return {
            "api_url": self.get("MODELS.OLLAMA.API_URL", "http://localhost:11434"),
        }


# Global configuration instance (for default config without specific path)
_config_instance = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global configuration instance"""
    global _config_instance
    if config_path is None:
        # Use the shared default instance
        if _config_instance is None:
            _config_instance = Config(config_path=None)
        return _config_instance
    else:
        # Create a new instance for specific config path
        return Config(config_path)