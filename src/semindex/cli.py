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
    get_changed_files,
    file_sha256_from_content,
)


def cmd_index(args: argparse.Namespace):
    repo = os.path.abspath(args.repo)
    index_dir = os.path.abspath(args.index_dir)
    os.makedirs(index_dir, exist_ok=True)

    db_path = os.path.join(index_dir, DB_NAME)
    index_path = os.path.join(index_dir, FAISS_INDEX)
    ensure_db(db_path)

    embedder = Embedder(model_name=args.model)

    # Check if incremental indexing is enabled
    incremental = args.incremental
    
    # Determine which files to process
    if incremental and os.path.exists(db_path) and os.path.exists(index_path):
        # Get list of changed files
        files_to_process = get_changed_files(repo, db_path)
        print(f"Found {len(files_to_process)} changed/new files for incremental indexing")
    else:
        # Process all files (fresh index or incremental not enabled)
        files_to_process = list(iter_python_files(repo))
        # Reset index for fresh start
        reset_index(index_dir, dim=embedder.model.config.hidden_size)
        print(f"Performing fresh indexing of {len(files_to_process)} files")

    all_symbol_rows = []
    all_texts: List[str] = []
    file_hashes = []

    # Determine which chunking method to use
    use_semantic_chunking = args.chunking == 'semantic'
    
    for path in files_to_process:
        try:
            source = read_text(path)
            current_hash = file_sha256_from_content(source.encode('utf-8'))
            
            symbols, _calls = parse_python_symbols(path, source)
            
            if use_semantic_chunking:
                # Use semantic chunking with the embedder
                chunks = build_semantic_chunks_from_symbols(source, symbols, embedder, 
                                                           similarity_threshold=getattr(args, 'similarity_threshold', 0.7))
            else:
                # Use original symbol-based chunking
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
                
            # Record file hash for incremental indexing
            file_hashes.append((path, current_hash))
        except Exception as e:
            if args.verbose:
                print(f"[WARN] Failed to process {path}: {e}")

    # persist symbols and vectors
    with db_conn(db_path) as con:
        cur = con.cursor()
        
        # If incremental, remove old entries for changed files before adding new ones
        if incremental and files_to_process:
            for file_path in files_to_process:
                cur.execute("DELETE FROM symbols WHERE path = ?", (file_path,))
        
        # bulk insert symbols
        cur.executemany(
            """
            INSERT INTO symbols (path, name, kind, start_line, end_line, signature, docstring, imports, bases)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            all_symbol_rows,
        )
        
        # update file hashes in the files table
        if file_hashes:
            cur.executemany(
                """
                INSERT OR REPLACE INTO files (path, hash) VALUES (?, ?)
                """,
                file_hashes
            )
        
        # fetch last N symbol ids in insertion order (only for new entries)
        last_id = cur.execute("SELECT ifnull(MAX(id),0) FROM symbols;").fetchone()[0]
        count = len(all_symbol_rows)
        first_id = last_id - count + 1 if count > 0 else 0
        symbol_ids = list(range(first_id, last_id + 1)) if count > 0 else []

        if all_texts:
            vecs = embedder.encode(all_texts, batch_size=args.batch)
            add_vectors(index_path, con, symbol_ids, vecs)

    # Also index in Elasticsearch for keyword search (only for changed files in incremental mode)
    try:
        from .keyword_search import KeywordSearcher
        from .store import get_all_symbols_for_keyword_index
        
        keyword_searcher = KeywordSearcher(index_dir)
        if not os.path.exists(os.path.join(index_dir, "index.faiss")) or not incremental:
            # Create index if it doesn't exist (fresh indexing)
            keyword_searcher.create_index()
        
        # Get all symbols from SQLite and index in Elasticsearch
        with db_conn(db_path) as con:
            all_symbols = get_all_symbols_for_keyword_index(con)
            # Add content to each symbol based on the source files
            for symbol in all_symbols:
                try:
                    with open(symbol["path"], 'r', encoding='utf-8') as f:
                        source = f.read()
                        # Extract the content for this specific symbol
                        start_line = max(symbol["start_line"] - 1, 0)
                        end_line = min(symbol["end_line"], source.count("\n") + 1)
                        content = "\n".join(source.splitlines()[start_line:end_line])
                        symbol["content"] = content
                except Exception as e:
                    print(f"[WARN] Could not read content for {symbol['path']}: {e}")
                    symbol["content"] = ""
            
            keyword_searcher.bulk_index_symbols(all_symbols)
        
        print(f"Also indexed in keyword search: {len(all_symbol_rows)} symbols")
    except ImportError:
        print("[WARN] Elasticsearch not available, skipping keyword indexing")
    except Exception as e:
        print(f"[WARN] Could not connect to Elasticsearch: {e}, skipping keyword indexing")

    print(f"Indexed {len(all_texts)} chunks from repository {repo}")
    if incremental:
        print(f"Processed {len(files_to_process)} changed/new files")


def cmd_query(args: argparse.Namespace):
    index_dir = os.path.abspath(args.index_dir)
    embedder = Embedder(model_name=args.model)
    qvec = embedder.encode([args.query])
    
    # Check if hybrid search is requested
    if args.hybrid:
        try:
            from .hybrid_search import hybrid_search
            results = hybrid_search(index_dir, qvec, args.query, top_k=args.top_k)
        except ImportError:
            print("[WARN] Hybrid search module not available, falling back to vector search")
            from .search import search_similar
            results = search_similar(index_dir, qvec, top_k=args.top_k)
    else:
        from .search import search_similar
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
    p_idx.add_argument("--model", default=os.environ.get("SEMINDEX_MODEL", "microsoft/codebert-base"))
    p_idx.add_argument("--batch", type=int, default=16)
    p_idx.add_argument("--verbose", action="store_true")
    p_idx.add_argument("--chunking", choices=['symbol', 'semantic'], default='symbol',
                       help="Chunking method to use: 'symbol' for original function/class-based, 'semantic' for semantic-aware chunking")
    p_idx.add_argument("--similarity-threshold", type=float, default=0.7,
                       help="Similarity threshold for semantic chunking (0.0-1.0)")
    p_idx.add_argument("--incremental", action="store_true",
                       help="Perform incremental indexing by only processing changed files")
    p_idx.set_defaults(func=cmd_index)

    p_q = sub.add_parser("query", help="Query the semantic index")
    p_q.add_argument("query", help="Natural language or code query")
    p_q.add_argument("--index-dir", default=".semindex")
    p_q.add_argument("--model", default=os.environ.get("SEMINDEX_MODEL", "microsoft/codebert-base"))
    p_q.add_argument("--top-k", type=int, default=10)
    p_q.add_argument("--hybrid", action="store_true", help="Use hybrid search (combines vector and keyword search)")
    p_q.set_defaults(func=cmd_query)

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)
