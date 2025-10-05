import argparse
import os
from typing import List

from .crawler import iter_python_files, read_text
from .ast_py import parse_python_symbols
from .chunker import build_chunks_from_symbols
from .embed import Embedder
from .store import (
    DB_NAME,
    FAISS_INDEX,
    ensure_db,
    db_conn,
    reset_index,
    add_vectors,
)


def cmd_index(args: argparse.Namespace):
    repo = os.path.abspath(args.repo)
    index_dir = os.path.abspath(args.index_dir)
    os.makedirs(index_dir, exist_ok=True)

    db_path = os.path.join(index_dir, DB_NAME)
    index_path = os.path.join(index_dir, FAISS_INDEX)
    ensure_db(db_path)

    embedder = Embedder(model_name=args.model)

    # fresh index for now (simple and deterministic)
    reset_index(index_dir, dim=embedder.model.config.hidden_size)

    all_symbol_rows = []
    all_texts: List[str] = []

    for path in iter_python_files(repo):
        try:
            source = read_text(path)
            symbols, _calls = parse_python_symbols(path, source)
            chunks = build_chunks_from_symbols(source, symbols)
            for ch in chunks:
                all_symbol_rows.append(
                    (
                        ch.symbol.path,
                        ch.symbol.name,
                        ch.symbol.kind,
                        ch.symbol.start_line,
                        ch.symbol.end_line,
                        ch.symbol.signature,
                        ch.symbol.docstring or "",
                        ",".join(ch.symbol.imports or []),
                        ",".join(ch.symbol.bases or []),
                    )
                )
                all_texts.append(ch.text)
        except Exception as e:
            if args.verbose:
                print(f"[WARN] Failed to process {path}: {e}")

    # persist symbols and vectors
    with db_conn(db_path) as con:
        cur = con.cursor()
        # bulk insert symbols
        cur.executemany(
            """
            INSERT INTO symbols (path, name, kind, start_line, end_line, signature, docstring, imports, bases)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            all_symbol_rows,
        )
        # fetch last N symbol ids in insertion order
        last_id = cur.execute("SELECT ifnull(MAX(id),0) FROM symbols;").fetchone()[0]
        count = len(all_symbol_rows)
        first_id = last_id - count + 1 if count > 0 else 0
        symbol_ids = list(range(first_id, last_id + 1)) if count > 0 else []

        if all_texts:
            vecs = embedder.encode(all_texts, batch_size=args.batch)
            add_vectors(index_path, con, symbol_ids, vecs)

    print(f"Indexed {len(all_texts)} chunks from repository {repo}")


def cmd_query(args: argparse.Namespace):
    from .search import search_similar

    index_dir = os.path.abspath(args.index_dir)
    embedder = Embedder(model_name=args.model)
    qvec = embedder.encode([args.query])
    results = search_similar(index_dir, qvec, top_k=args.top_k)
    if not results:
        print("No results.")
        return
    for score, sid, (path, name, kind, start, end, sig) in results:
        print(f"score={score:.4f} | {kind} {name} @ {path}:{start}-{end}")
        if sig:
            print(f"  sig: {sig}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="semindex", description="Local Python semantic indexer (CPU-only)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="Index a Python repository")
    p_idx.add_argument("repo", help="Path to repository root")
    p_idx.add_argument("--index-dir", default=".semindex", help="Path to store index files (FAISS + SQLite)")
    p_idx.add_argument("--model", default=os.environ.get("SEMINDEX_MODEL", "BAAI/bge-small-en-v1.5"))
    p_idx.add_argument("--batch", type=int, default=16)
    p_idx.add_argument("--verbose", action="store_true")
    p_idx.set_defaults(func=cmd_index)

    p_q = sub.add_parser("query", help="Query the semantic index")
    p_q.add_argument("query", help="Natural language or code query")
    p_q.add_argument("--index-dir", default=".semindex")
    p_q.add_argument("--model", default=os.environ.get("SEMINDEX_MODEL", "BAAI/bge-small-en-v1.5"))
    p_q.add_argument("--top-k", type=int, default=10)
    p_q.set_defaults(func=cmd_query)

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)
