import os
from typing import List
import numpy as np

DEFAULT_MODEL = os.environ.get("SEMINDEX_MODEL", "BAAI/bge-small-en-v1.5")


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        # Defer heavy imports to keep module import light and test-friendly
        import torch  # local import
        from transformers import AutoTokenizer, AutoModel  # local import

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: List[str], batch_size: int = 16, max_length: int = 512) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            vecs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                outputs = self.model(**tokens)
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
