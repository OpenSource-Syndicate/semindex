from typing import List, Tuple
import numpy as np
from .store import search as faiss_search


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
