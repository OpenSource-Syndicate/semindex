import os
from typing import List, Tuple
import numpy as np
import sqlite3

from .store import DB_NAME, DOCS_FAISS_INDEX
from .store import search_docs as _search_docs


def search_docs(index_dir: str, query_vec: np.ndarray, top_k: int = 10):
    db_path = os.path.join(index_dir, DB_NAME)
    index_path = os.path.join(index_dir, DOCS_FAISS_INDEX)
    con = sqlite3.connect(db_path)
    try:
        return _search_docs(index_path, con, query_vec, top_k)
    finally:
        con.close()
