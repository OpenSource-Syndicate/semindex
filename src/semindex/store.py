import os
import hashlib
import sqlite3
from contextlib import contextmanager
from typing import Iterable, List, Tuple, Optional

import faiss
import numpy as np

from .languages import ensure_default_adapters, get_adapter, iter_all_supported_files


DB_NAME = "semindex.db"
FAISS_INDEX = "index.faiss"
DOCS_FAISS_INDEX = "docs.faiss"


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as rf:
        for chunk in iter(lambda: rf.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256_from_content(content: bytes) -> str:
    """Calculate SHA256 hash from file content (bytes)."""

    h = hashlib.sha256()
    h.update(content)
    return h.hexdigest()


def get_changed_files(repo_path: str, db_path: str, language: str = "auto") -> List[str]:
    """Return files that have changed since the last indexing run."""

    ensure_default_adapters()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT path, hash FROM files")
        existing_hashes = {row[0]: row[1] for row in cursor.fetchall()}
    finally:
        conn.close()

    if language == "auto":
        file_iter: Iterable[str] = iter_all_supported_files(repo_path)
    else:
        adapter = get_adapter(language)
        file_iter = adapter.discover_files(repo_path)

    changed_files: List[str] = []
    for file_path in file_iter:
        current_hash = file_sha256(file_path)
        if file_path in existing_hashes:
            if existing_hashes[file_path] != current_hash:
                changed_files.append(file_path)
        else:
            changed_files.append(file_path)

    return changed_files


def get_all_files_in_db(db_path: str) -> List[str]:
    """Return all file paths currently tracked in the index database."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT path FROM files")
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def remove_file_from_index(db_path: str, index_path: str, file_path: str):
    """Remove all artifacts corresponding to a file from SQLite and FAISS."""

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM symbols WHERE path = ?", (file_path,))
    symbol_ids = [row[0] for row in cursor.fetchall()]

    if symbol_ids:
        cursor.executemany(
            "DELETE FROM calls WHERE caller_id = ? OR callee_symbol_id = ?",
            [(sid, sid) for sid in symbol_ids],
        )

    cursor.execute("DELETE FROM symbols WHERE path = ?", (file_path,))
    cursor.execute("DELETE FROM files WHERE path = ?", (file_path,))

    conn.commit()
    conn.close()

    # Removing vectors from FAISS still requires rebuilding the index; omitted here.


def ensure_db(db_path: str):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            hash TEXT NOT NULL,
            language TEXT
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
            bases TEXT,
            language TEXT,
            namespace TEXT,
            symbol_type TEXT
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vectors (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol_id INTEGER NOT NULL,
            FOREIGN KEY(symbol_id) REFERENCES symbols(id) ON DELETE CASCADE
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS calls (
            caller_id INTEGER NOT NULL,
            callee_name TEXT NOT NULL,
            callee_symbol_id INTEGER,
            callee_path TEXT,
            FOREIGN KEY(caller_id) REFERENCES symbols(id) ON DELETE CASCADE,
            FOREIGN KEY(callee_symbol_id) REFERENCES symbols(id) ON DELETE SET NULL
        );
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_calls_unique
        ON calls (caller_id, callee_name, IFNULL(callee_symbol_id, -1), IFNULL(callee_path, ''));
        """
    )

    _ensure_column(cur, "files", "language", "ALTER TABLE files ADD COLUMN language TEXT")
    _ensure_column(cur, "symbols", "language", "ALTER TABLE symbols ADD COLUMN language TEXT")
    _ensure_column(cur, "symbols", "namespace", "ALTER TABLE symbols ADD COLUMN namespace TEXT")
    _ensure_column(cur, "symbols", "symbol_type", "ALTER TABLE symbols ADD COLUMN symbol_type TEXT")

    # Documentation tables
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS doc_packages (
            name TEXT,
            version TEXT,
            PRIMARY KEY (name, version)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS doc_pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            package TEXT,
            version TEXT,
            url TEXT,
            title TEXT,
            checksum TEXT,
            last_indexed TIMESTAMP
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS doc_vectors (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id INTEGER NOT NULL,
            FOREIGN KEY(page_id) REFERENCES doc_pages(id)
        );
        """
    )

    con.commit()
    con.close()


@contextmanager
def db_conn(db_path: str):
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON;")
    try:
        yield con
    finally:
        con.commit()
        con.close()


def reset_index(index_dir: str, dim: int):
    os.makedirs(index_dir, exist_ok=True)

    index = faiss.IndexFlatIP(dim)
    faiss.write_index(index, os.path.join(index_dir, FAISS_INDEX))

    with db_conn(os.path.join(index_dir, DB_NAME)) as con:
        cur = con.cursor()
        cur.execute("DELETE FROM vectors;")
        cur.execute("DELETE FROM symbols;")
        cur.execute("DELETE FROM calls;")

    # Also reset docs index
    faiss.write_index(faiss.IndexFlatIP(dim), os.path.join(index_dir, DOCS_FAISS_INDEX))
    with db_conn(os.path.join(index_dir, DB_NAME)) as con:
        cur = con.cursor()
        cur.execute("DELETE FROM doc_vectors;")
        cur.execute("DELETE FROM doc_pages;")
        cur.execute("DELETE FROM doc_packages;")


def add_symbols(
    con: sqlite3.Connection,
    symbols: List[Tuple[str, str, str, int, int, str, str, str, str, str, str, str]],
) -> List[int]:
    cur = con.cursor()
    cur.executemany(
        """
        INSERT INTO symbols (
            path,
            name,
            kind,
            start_line,
            end_line,
            signature,
            docstring,
            imports,
            bases,
            language,
            namespace,
            symbol_type
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


def add_calls(
    con: sqlite3.Connection,
    call_rows: Iterable[Tuple[int, str, Optional[int], Optional[str]]],
) -> None:
    cur = con.cursor()
    cur.executemany(
        """
        INSERT OR IGNORE INTO calls (caller_id, callee_name, callee_symbol_id, callee_path)
        VALUES (?, ?, ?, ?);
        """,
        list(call_rows),
    )


def add_doc_vectors(index_path: str, con: sqlite3.Connection, page_ids: List[int], vectors: np.ndarray):
    index = faiss.read_index(index_path)
    index.add(vectors.astype(np.float32))
    faiss.write_index(index, index_path)
    cur = con.cursor()
    cur.executemany("INSERT INTO doc_vectors (page_id) VALUES (?);", [(pid,) for pid in page_ids])


def search(index_path: str, con: sqlite3.Connection, query_vec: np.ndarray, top_k: int = 10):
    index = faiss.read_index(index_path)
    if index.ntotal == 0:
        return []

    D, I = index.search(query_vec.astype(np.float32), top_k)

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


def search_docs(index_path: str, con: sqlite3.Connection, query_vec: np.ndarray, top_k: int = 10):
    index = faiss.read_index(index_path)
    if index.ntotal == 0:
        return []

    D, I = index.search(query_vec.astype(np.float32), top_k)

    cur = con.cursor()
    rows = cur.execute("SELECT rowid, page_id FROM doc_vectors ORDER BY rowid ASC;").fetchall()
    id_map = [pid for (_rowid, pid) in rows]

    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(id_map):
            continue
        pid = id_map[idx]
        page = cur.execute(
            "SELECT package, version, url, title FROM doc_pages WHERE id=?;",
            (pid,),
        ).fetchone()
        if page:
            results.append((score, pid, page))
    return results


def get_all_symbols_for_keyword_index(con: sqlite3.Connection) -> List[dict]:
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, path, name, kind, start_line, end_line, signature, docstring, imports, bases, language, namespace, symbol_type
        FROM symbols
        """
    )

    symbols: List[dict] = []
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
            "bases": row[9],
            "language": row[10],
            "namespace": row[11],
            "symbol_type": row[12],
        }
        symbols.append(symbol)
    return symbols


def get_symbol_by_id(con: sqlite3.Connection, symbol_id: int):
    cur = con.cursor()
    cur.execute(
        """
        SELECT path, name, kind, start_line, end_line, signature
        FROM symbols
        WHERE id=?
        """,
        (symbol_id,),
    )
    return cur.fetchone()


def get_doc_page_by_id(con: sqlite3.Connection, page_id: int):
    cur = con.cursor()
    cur.execute(
        """
        SELECT package, version, url, title
        FROM doc_pages
        WHERE id=?
        """,
        (page_id,),
    )
    return cur.fetchone()


def get_callers(con: sqlite3.Connection, symbol_id: int) -> List[Tuple[int, str, Optional[int], Optional[str]]]:
    cur = con.cursor()
    cur.execute(
        """
        SELECT caller_id, symbols.name, calls.callee_symbol_id, calls.callee_path
        FROM calls
        JOIN symbols ON symbols.id = calls.caller_id
        WHERE calls.callee_symbol_id = ?
        ORDER BY symbols.name
        """,
        (symbol_id,),
    )
    return cur.fetchall()


def get_callees(con: sqlite3.Connection, symbol_id: int) -> List[Tuple[int, str, Optional[int], Optional[str]]]:
    cur = con.cursor()
    cur.execute(
        """
        SELECT calls.callee_symbol_id, calls.callee_name, calls.callee_symbol_id, calls.callee_path
        FROM calls
        WHERE caller_id = ?
        ORDER BY calls.callee_name
        """,
        (symbol_id,),
    )
    return cur.fetchall()


def _ensure_column(cur: sqlite3.Cursor, table: str, column: str, ddl: str) -> None:
    cur.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cur.fetchall()}
    if column not in existing:
        cur.execute(ddl)
