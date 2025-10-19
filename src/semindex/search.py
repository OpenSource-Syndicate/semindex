from typing import List, Tuple
import numpy as np
from .store import search as faiss_search
from .embed import Embedder
import os


def search_similar(index_dir: str, query_vec: np.ndarray, top_k: int = 10):
    from .store import DB_NAME, FAISS_INDEX
    import os
    db_path = os.path.join(index_dir, DB_NAME)
    index_path = os.path.join(index_dir, FAISS_INDEX)
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        return faiss_search(index_path, con, query_vec, top_k)
    finally:
        con.close()


class Searcher:
    """
    High-level search API.
    Usage:
        searcher = Searcher(index_dir=".semindex")
        results = searcher.query("how is user auth implemented?", hybrid=True)
    """

    def __init__(self, index_dir: str = ".semindex", model: str | None = None) -> None:
        self.index_dir = os.path.abspath(index_dir)
        self.model = model or os.environ.get("SEMINDEX_MODEL", "microsoft/codebert-base")
        self.embedder = Embedder(model_name=self.model)

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        *,
        hybrid: bool = False,
        include_docs: bool = False,
        docs_weight: float = 0.4,
    ) -> List[Tuple[float, int, tuple]]:
        qvec = self.embedder.encode([query_text])

        # primary search: dense or hybrid
        if hybrid:
            try:
                from .hybrid_search import hybrid_search as _hybrid_search
                results = _hybrid_search(self.index_dir, qvec, query_text, top_k=top_k)
            except Exception:
                # Fallback to dense-only search
                results = search_similar(self.index_dir, qvec, top_k=top_k)
        else:
            results = search_similar(self.index_dir, qvec, top_k=top_k)

        # optional docs retrieval
        doc_results: List[Tuple[float, int, tuple]] = []
        if include_docs:
            try:
                from .doc_search import search_docs
                doc_results = search_docs(self.index_dir, qvec, top_k=top_k)
            except Exception:
                doc_results = []

        if not doc_results:
            return results

        # merge code + docs with simple weighted normalization (mirrors CLI)
        merged: List[Tuple[float, tuple]] = []
        code_scores = [r[0] for r in results] or [1.0]
        doc_scores = [r[0] for r in doc_results] or [1.0]
        max_code = max(code_scores) if code_scores else 1.0
        max_doc = max(doc_scores) if doc_scores else 1.0
        for r in results:
            merged.append(((1.0 - docs_weight) * (r[0] / (max_code or 1.0)), ("code", r)))
        for r in doc_results:
            merged.append((docs_weight * (r[0] / (max_doc or 1.0)), ("doc", r)))
        merged.sort(key=lambda x: x[0], reverse=True)
        merged = merged[:top_k]

        # return unified list: code results are already in (score, id, info)
        final: List[Tuple[float, int, tuple]] = []
        for _score, (rtype, r) in merged:
            final.append(r)
        return final
