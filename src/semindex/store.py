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


def file_sha256_from_content(content: bytes) -> str:
    """
    Calculate SHA256 hash from file content (bytes).
    
    :param content: File content as bytes
    :return: SHA256 hash as string
    """
    h = hashlib.sha256()
    h.update(content)
    return h.hexdigest()


def get_changed_files(repo_path: str, db_path: str) -> List[str]:
    """
    Get list of files that have changed since last indexing.
    
    :param repo_path: Repository path
    :param db_path: Database path
    :return: List of changed file paths
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get existing file hashes from the database
    cursor.execute("SELECT path, hash FROM files")
    existing_hashes = {row[0]: row[1] for row in cursor.fetchall()}
    
    conn.close()
    
    changed_files = []
    
    # Iterate through all Python files in the repo
    from .crawler import iter_python_files
    for file_path in iter_python_files(repo_path):
        # Calculate current hash of the file
        current_hash = file_sha256(file_path)
        
        # Check if file exists in DB and if the hash has changed
        if file_path in existing_hashes:
            if existing_hashes[file_path] != current_hash:
                changed_files.append(file_path)
        else:
            # File is new, needs to be indexed
            changed_files.append(file_path)
    
    return changed_files


def get_all_files_in_db(db_path: str) -> List[str]:
    """
    Get all file paths stored in the database.
    
    :param db_path: Database path
    :return: List of file paths
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT path FROM files")
    files = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return files


def remove_file_from_index(db_path: str, index_path: str, file_path: str):
    """
    Remove all symbols associated with a file from the index.
    
    :param db_path: Database path
    :param index_path: FAISS index path
    :param file_path: Path of file to remove
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all symbol IDs associated with the file
    cursor.execute("SELECT id FROM symbols WHERE path = ?", (file_path,))
    symbol_ids = [row[0] for row in cursor.fetchall()]
    
    # Remove symbols from SQLite
    cursor.execute("DELETE FROM symbols WHERE path = ?", (file_path,))
    cursor.execute("DELETE FROM files WHERE path = ?", (file_path,))
    
    conn.commit()
    conn.close()
    
    # Note: Removing from FAISS is more complex as it requires
    # rebuilding the index without those vectors, which we're not doing here
    # for simplicity. In a production system, you might want to implement
    # a marking system or use a different FAISS index type that supports deletion.


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


def get_all_symbols_for_keyword_index(con: sqlite3.Connection) -> List[dict]:
    """
    Retrieve all symbols for keyword indexing.
    
    :param con: SQLite connection
    :return: List of symbols as dictionaries
    """
    cur = con.cursor()
    cur.execute("""
        SELECT id, path, name, kind, start_line, end_line, signature, docstring, imports, bases
        FROM symbols
    """)
    
    symbols = []
    for row in cur.fetchall():
        symbol = {
            "id": row[0],
            "path": row[1],
            "name": row[2],
            "kind": row[3],
            "start_line": row[4],
            "end_line": row[5],
            "signature": row[6],
            "docstring": row[7],
            "imports": row[8],
            "bases": row[9]
        }
        symbols.append(symbol)
    
    return symbols


def get_symbol_by_id(con: sqlite3.Connection, symbol_id: int):
    """
    Retrieve a symbol by its ID.
    
    :param con: SQLite connection
    :param symbol_id: The symbol ID to retrieve
    :return: Symbol information
    """
    cur = con.cursor()
    cur.execute("""
        SELECT path, name, kind, start_line, end_line, signature
        FROM symbols
        WHERE id=?
    """, (symbol_id,))
    
    return cur.fetchone()
