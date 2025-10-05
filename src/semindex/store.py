import os
import hashlib
import sqlite3
from contextlib import contextmanager
from typing import Iterable, List, Tuple

import faiss
import numpy as np

DB_NAME = "semindex.db"
FAISS_INDEX = "index.faiss"


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as rf:
        for chunk in iter(lambda: rf.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_db(db_path: str):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            hash TEXT NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            start_line INTEGER,
            end_line INTEGER,
            signature TEXT,
            docstring TEXT,
            imports TEXT,
            bases TEXT
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vectors (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol_id INTEGER NOT NULL,
            FOREIGN KEY(symbol_id) REFERENCES symbols(id)
        );
        """
    )
    con.commit()
    con.close()


@contextmanager
def db_conn(db_path: str):
    con = sqlite3.connect(db_path)
    try:
        yield con
    finally:
        con.commit()
        con.close()


def reset_index(index_dir: str, dim: int):
    os.makedirs(index_dir, exist_ok=True)
    # fresh FAISS index and vectors mapping
    index = faiss.IndexFlatIP(dim)
    faiss.write_index(index, os.path.join(index_dir, FAISS_INDEX))
    with db_conn(os.path.join(index_dir, DB_NAME)) as con:
        cur = con.cursor()
        cur.execute("DELETE FROM vectors;")
        cur.execute("DELETE FROM symbols;")
        # keep files table for incremental decisions later if needed


def add_symbols(con: sqlite3.Connection, symbols: List[Tuple[str, str, str, int, int, str, str, str, str]]) -> List[int]:
    cur = con.cursor()
    cur.executemany(
        """
        INSERT INTO symbols (path, name, kind, start_line, end_line, signature, docstring, imports, bases)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        symbols,
    )
    return [r[0] for r in cur.execute("SELECT last_insert_rowid();").fetchall()]


def add_vectors(index_path: str, con: sqlite3.Connection, symbol_ids: List[int], vectors: np.ndarray):
    index = faiss.read_index(index_path)
    index.add(vectors.astype(np.float32))
    faiss.write_index(index, index_path)
    cur = con.cursor()
    cur.executemany("INSERT INTO vectors (symbol_id) VALUES (?);", [(sid,) for sid in symbol_ids])


def search(index_path: str, con: sqlite3.Connection, query_vec: np.ndarray, top_k: int = 10):
    index = faiss.read_index(index_path)
    if index.ntotal == 0:
        return []
    D, I = index.search(query_vec.astype(np.float32), top_k)
    # map FAISS row to symbol_id in insertion order
    cur = con.cursor()
    rows = cur.execute("SELECT rowid, symbol_id FROM vectors ORDER BY rowid ASC;").fetchall()
    id_map = [sid for (_rowid, sid) in rows]
    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(id_map):
            continue
        sid = id_map[idx]
        sym = cur.execute(
            "SELECT path, name, kind, start_line, end_line, signature FROM symbols WHERE id=?;",
            (sid,),
        ).fetchone()
        if sym:
            results.append((score, sid, sym))
    return results
