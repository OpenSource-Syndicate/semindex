import os
import hashlib
from typing import List
import numpy as np
import psutil

from .cache import cache_manager
from .config import get_config

# Use configuration to get default model, fallback to environment variable or default
def _get_default_model():
    config = get_config()
    model_from_config = config.get("MODELS.EMBEDDING_MODEL")
    if model_from_config and model_from_config.strip():
        return model_from_config
    return os.environ.get("SEMINDEX_MODEL", "BAAI/bge-small-en-v1.5")  # Better performance and speed than codebert-base

DEFAULT_MODEL = _get_default_model()


def _get_optimal_batch_size(device: str, model_name: str, texts_count: int) -> int:
    """Determine optimal batch size based on system resources and model characteristics.
    
    Args:
        device: Device being used ('cpu', 'cuda', etc.)
        model_name: Name of the model being used
        texts_count: Number of texts to process
        
    Returns:
        Optimal batch size for processing
    """
    try:
        # Get system memory information
        memory = psutil.virtual_memory()
        available_memory_mb = memory.available / (1024 * 1024)
        
        # Base batch size based on device
        if device == "cuda":
            # For GPU, check CUDA memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                    # Use 80% of GPU memory to account for overhead
                    available_memory_mb = min(available_memory_mb, gpu_memory_mb * 0.8)
            except:
                pass
            
            # GPU typically can handle larger batches
            base_batch_size = 32
        else:
            # CPU batch sizing based on available memory
            base_batch_size = 16
            
        # Adjust for model size (approximate)
        if "large" in model_name.lower() or "3b" in model_name.lower() or "7b" in model_name.lower():
            # Larger models need smaller batches
            base_batch_size = max(1, base_batch_size // 2)
        elif "small" in model_name.lower() or "mini" in model_name.lower() or "tiny" in model_name.lower():
            # Smaller models can handle larger batches
            base_batch_size = min(64, base_batch_size * 2)
            
        # Further adjust based on available memory
        if available_memory_mb < 1024:  # Less than 1GB
            base_batch_size = max(1, base_batch_size // 4)
        elif available_memory_mb < 2048:  # Less than 2GB
            base_batch_size = max(1, base_batch_size // 2)
        elif available_memory_mb > 8192:  # More than 8GB
            base_batch_size = min(128, base_batch_size * 2)
            
        # For very small text sets, use smaller batches to reduce overhead
        if texts_count < 10:
            base_batch_size = min(base_batch_size, 8)
        elif texts_count < 100:
            base_batch_size = min(base_batch_size, 32)
            
        # Ensure minimum batch size of 1
        return max(1, int(base_batch_size))
        
    except Exception:
        # Fallback to reasonable default
        return 16


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        # Defer heavy imports to keep module import light and test-friendly
        import torch  # local import
        from transformers import AutoTokenizer, AutoModel  # local import

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens if they don't exist (important for many models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Cache for this specific model instance
        self.model_cache_key = f"embedder_{model_name}"

    def encode(self, texts: List[str], batch_size: int = None, max_length: int = 512) -> np.ndarray:
        """Encode texts to embeddings with adaptive batch sizing.
        
        If batch_size is None, automatically determines optimal batch size based on 
        system resources and model characteristics.
        """
        # Automatically determine optimal batch size if not provided
        if batch_size is None:
            batch_size = _get_optimal_batch_size(self.device, self.model_name, len(texts))
        
        # Check for cached embeddings first
        uncached_texts = []
        uncached_indices = []
        results = [None] * len(texts)
        
        for i, text in enumerate(texts):
            cached_result = cache_manager.get_embedding(text)
            if cached_result is not None:
                results[i] = cached_result
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process only uncached texts
        if uncached_texts:
            torch = self.torch
            with torch.no_grad():
                vecs = []
                for i in range(0, len(uncached_texts), batch_size):
                    batch = uncached_texts[i : i + batch_size]
                    
                    # Add special tokens for code-specific models if needed
                    processed_batch = self._preprocess_texts(batch)
                    
                    tokens = self.tokenizer(
                        processed_batch,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                        # Use the pad token we set above
                    )
                    tokens = {k: v.to(self.device) for k, v in tokens.items()}
                    outputs = self.model(**tokens)
                    
                    # Use different pooling strategies based on the model
                    if self._is_sentence_transformer_model():
                        # For sentence transformer style models, use CLS token or mean pooling
                        last_hidden = outputs.last_hidden_state  # [B, T, H]
                        # Use CLS token if available (first token), otherwise mean pooling
                        if self._has_cls_pooling():
                            emb = last_hidden[:, 0, :].cpu().numpy()  # Use CLS token
                        else:
                            # Apply attention mask for proper mean pooling
                            mask = tokens["attention_mask"].unsqueeze(-1)  # [B, T, 1]
                            summed = (last_hidden * mask).sum(dim=1)
                            counts = mask.sum(dim=1).clamp(min=1)
                            emb = (summed / counts).cpu().numpy()
                    else:
                        # Default behavior for other models
                        last_hidden = outputs.last_hidden_state  # [B, T, H]
                        mask = tokens["attention_mask"].unsqueeze(-1)  # [B, T, 1]
                        summed = (last_hidden * mask).sum(dim=1)
                        counts = mask.sum(dim=1).clamp(min=1)
                        emb = (summed / counts).cpu().numpy()
                    
                    # L2 normalize for cosine with IndexFlatIP
                    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
                    vecs.append(emb)
            
            uncached_results = (
                np.vstack(vecs)
                if vecs
                else np.zeros((0, getattr(self.model.config, "hidden_size", 768)), dtype=np.float32)
            )
            
            # Cache the results and update results array
            if len(uncached_results) > 0:
                for idx, result in zip(uncached_indices, uncached_results):
                    results[idx] = result
                    # Cache the individual result
                    cache_manager.cache_embedding(texts[idx], result)

        # Stack and return all results
        return (
            np.vstack(results)
            if results and any(r is not None for r in results)
            else np.zeros((0, getattr(self.model.config, "hidden_size", 768)), dtype=np.float32)
        )

    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess texts specifically for code embedding models.
        Applies special formatting to help models better understand code.
        """
        processed = []
        for text in texts:
            # Detect if text looks like code based on common patterns
            if self._looks_like_code(text):
                # Apply code-specific preprocessing
                formatted_text = self._format_code_text(text)
            else:
                # For non-code text, just strip whitespace
                formatted_text = text.strip()
            processed.append(formatted_text)
        return processed

    def _looks_like_code(self, text: str) -> bool:
        """
        Determine if text appears to be code based on common patterns.
        """
        # Check for common code patterns
        code_indicators = [
            'def ', 'class ', 'import ', 'from ', 'return ',
            'if ', 'else ', 'for ', 'while ', 'try:', 'except',
            'def(', 'class(', 'import(', 'from(', 'return(',
            '{', '}', '[', ']', '(', ')', ':',
            ' = ', ' == ', ' != ', ' < ', ' > ', ' <= ', ' >= ',
            '.length', '.append', '.push', '.pop', '.map', '.filter',
            'function ', 'var ', 'let ', 'const ', '=>', '=>{',
            'public ', 'private ', 'protected ', 'static ', 'void ',
            '#include', 'using namespace', 'namespace ',
            'SELECT ', 'INSERT ', 'UPDATE ', 'DELETE ', 'FROM ',
            self._has_indentation(text),
        ]
        
        # Count indicators that suggest this is code
        code_score = 0
        for indicator in code_indicators:
            if isinstance(indicator, bool):
                if indicator:
                    code_score += 1
            else:
                if indicator in text:
                    code_score += 1
        
        # If more than 1 indicator, likely code
        return code_score > 1

    def _has_indentation(self, text: str) -> bool:
        """
        Check if text has Python-style indentation (or similar).
        """
        lines = text.split('\n')
        for line in lines:
            stripped = line.lstrip()
            if stripped and len(line) > len(stripped):  # Has leading whitespace
                return True
        return False

    def _format_code_text(self, text: str) -> str:
        """
        Apply special formatting to code text to help models understand it better.
        """
        # For models that benefit from explicit code marking
        if self._should_add_code_prefix():
            return f"def {text}" if text.strip().startswith("def ") else f"code: {text}"
        
        # For other models, preserve code structure but clean it up
        return self._clean_code_text(text)

    def _clean_code_text(self, text: str) -> str:
        """
        Clean code text while preserving its structure.
        """
        # Normalize whitespace while preserving indentation
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Preserve important whitespace but remove trailing spaces
            cleaned_lines.append(line.rstrip())
        
        return '\n'.join(cleaned_lines).strip()

    def _should_add_code_prefix(self) -> bool:
        """
        Determine if the model benefits from code-specific prefixes.
        """
        model_name_lower = self.model_name.lower()
        
        # Models that benefit from code prefixes
        has_prefix_support = any(pattern in model_name_lower for pattern in [
            'codet5', 'codebert', 'graphcodebert'
        ])
        
        return has_prefix_support

    def _is_sentence_transformer_model(self) -> bool:
        """
        Check if the model is a sentence transformer style model.
        This can be extended to detect specific model architectures.
        """
        # For now, just return based on model name patterns
        model_name_lower = self.model_name.lower()
        return any(pattern in model_name_lower for pattern in ['bge', 'gte', 'minilm', 'all-mpnet', 'paraphrase'])

    def _has_cls_pooling(self) -> bool:
        """
        Check if model has built-in CLS pooling.
        This can be extended to check model configurations.
        """
        model_name_lower = self.model_name.lower()
        # Models like BERT variants often use CLS token
        return any(pattern in model_name_lower for pattern in ['bert', 'roberta', 'bge'])
