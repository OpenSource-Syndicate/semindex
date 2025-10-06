import os
from typing import List
import numpy as np

DEFAULT_MODEL = os.environ.get("SEMINDEX_MODEL", "microsoft/codebert-base")


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

    def encode(self, texts: List[str], batch_size: int = 16, max_length: int = 512) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            vecs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                
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
        return (
            np.vstack(vecs)
            if vecs
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
