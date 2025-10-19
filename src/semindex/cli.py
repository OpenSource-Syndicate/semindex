import argparse
import os
from typing import List, Sequence, Tuple

from .ast_py import parse_python_symbols as _parse_python_symbols
from .chunker import build_chunks_from_symbols as _build_chunks_from_symbols
from .crawler import iter_python_files as _iter_python_files, read_text
from .embed import Embedder
from .languages import (
    LanguageAdapter,
    collect_index_targets,
    ensure_default_adapters,
    get_adapter,
    get_adapter_for_path,
)
from .languages.base import ParseResult
from .model import Chunk, ChunkingConfig, Symbol
from .store import (
    DB_NAME,
    FAISS_INDEX,
    add_vectors,
    ensure_db,
    db_conn,
    reset_index,
    get_changed_files,
    file_sha256_from_content,
)
from .store import DOCS_FAISS_INDEX


# Re-export helpers expected by tests for monkeypatching.
iter_python_files = _iter_python_files
parse_python_symbols = _parse_python_symbols
build_chunks_from_symbols = _build_chunks_from_symbols


def _default_chunks_for_symbols(source: str, symbols: List[Symbol]) -> List[Chunk]:
    chunks = []
    for symbol in symbols:
        if symbol.kind in {"function", "method", "class"}:
            start = max(symbol.start_line - 1, 0)
            end = symbol.end_line
            snippet = "\n".join(source.splitlines()[start:end]) or symbol.signature or symbol.name
            chunks.append(Chunk(symbol=symbol, text=snippet))

    if not chunks and symbols:
        chunks.append(Chunk(symbol=symbols[0], text=source))

    return chunks


def _make_module_symbol(path: str, source: str, language_name: str) -> Symbol:
    return Symbol(
        path=path,
        name=os.path.splitext(os.path.basename(path))[0],
        kind="module",
        start_line=1,
        end_line=source.count("\n") + 1,
        signature="",
        docstring=None,
        imports=[],
        bases=[],
        language=language_name,
        namespace=None,
        symbol_type="module",
    )


def _resolve_targets(
    paths: Sequence[str],
    language: str,
) -> List[Tuple[str, LanguageAdapter, str]]:
    """Resolve file paths to (language, adapter, path) tuples."""

    ensure_default_adapters()

    resolved: List[Tuple[str, "LanguageAdapter", str]] = []
    if language != "auto":
        adapter = get_adapter(language)
        for path in paths:
            resolved.append((language, adapter, path))
        return resolved

    for path in paths:
        adapter = get_adapter_for_path(path)
        if not adapter:
            continue
        resolved.append((adapter.name, adapter, path))
    return resolved


def _parse_and_chunk(
    adapter_name: str,
    adapter,
    path: str,
    source: str,
    embedder: Embedder,
    chunk_config: ChunkingConfig,
) -> ParseResult:
    return adapter.process_file(path, source, embedder, chunk_config)


def _format_symbol_row(symbol: Symbol) -> Tuple[str, str, str, int, int, str, str, str, str, str, str, str]:
    return (
        symbol.path,
        symbol.name,
        symbol.kind,
        symbol.start_line,
        symbol.end_line,
        symbol.signature,
        symbol.docstring or "",
        ",".join(symbol.imports or []),
        ",".join(symbol.bases or []),
        symbol.language or "",
        symbol.namespace or "",
        symbol.symbol_type or "",
    )


def cmd_index(args: argparse.Namespace):
    repo = os.path.abspath(args.repo)
    index_dir = os.path.abspath(args.index_dir)
    os.makedirs(index_dir, exist_ok=True)

    db_path = os.path.join(index_dir, DB_NAME)
    index_path = os.path.join(index_dir, FAISS_INDEX)
    ensure_db(db_path)

    embedder = Embedder(model_name=args.model)

    chunk_config = ChunkingConfig(
        method=args.chunking,
        similarity_threshold=args.similarity_threshold,
    )

    ensure_default_adapters()

    language = getattr(args, "language", "auto")
    incremental = args.incremental

    if incremental and os.path.exists(db_path) and os.path.exists(index_path):
        changed_files = get_changed_files(repo, db_path, language=language)
        targets = _resolve_targets(changed_files, language)
        print(f"Found {len(targets)} changed/new files for incremental indexing")
    else:
        base_targets = collect_index_targets(repo, language)
        targets = [(adapter.name, adapter, path) for adapter, path in base_targets]
        reset_index(index_dir, dim=embedder.model.config.hidden_size)
        print(f"Performing fresh indexing of {len(targets)} files")

    all_symbol_rows: List[Tuple[str, str, str, int, int, str, str, str, str, str, str, str]] = []
    all_texts: List[str] = []
    file_hashes: List[Tuple[str, str, str]] = []

    for language_name, adapter, path in targets:
        try:
            source = read_text(path)
        except Exception as exc:
            if args.verbose:
                print(f"[WARN] Failed to read {path}: {exc}")
            continue

        current_hash = file_sha256_from_content(source.encode("utf-8"))

        parse_failed = False
        try:
            result = _parse_and_chunk(
                language_name,
                adapter,
                path,
                source,
                embedder,
                chunk_config,
            )
        except Exception as exc:
            parse_failed = True
            if args.verbose:
                print(f"[WARN] Failed to process {path}: {exc}")

        if parse_failed:
            symbols = [_make_module_symbol(path, source, language_name)]
            chunks = [Chunk(symbol=symbols[0], text=source)]
        else:
            symbols = list(result.symbols)
            chunks = list(result.chunks)

            if not symbols:
                symbols.append(_make_module_symbol(path, source, language_name))

        if not chunks:
            chunks = _default_chunks_for_symbols(source, symbols)

        for symbol in symbols:
            all_symbol_rows.append(_format_symbol_row(symbol))

        for chunk in chunks:
            all_texts.append(chunk.text)

        file_hashes.append((path, current_hash, language_name))

    with db_conn(db_path) as con:
        cur = con.cursor()

        if incremental and targets:
            for _language_name, _adapter, file_path in targets:
                cur.execute("DELETE FROM symbols WHERE path = ?", (file_path,))

        if all_symbol_rows:
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
                all_symbol_rows,
            )

        if file_hashes:
            cur.executemany(
                """
                INSERT OR REPLACE INTO files (path, hash, language)
                VALUES (?, ?, ?)
                """,
                file_hashes,
            )

        last_id = cur.execute("SELECT ifnull(MAX(id), 0) FROM symbols;").fetchone()[0]
        count = len(all_symbol_rows)
        first_id = last_id - count + 1 if count > 0 else 0
        symbol_ids = list(range(first_id, last_id + 1)) if count > 0 else []

        if all_texts:
            vecs = embedder.encode(all_texts, batch_size=args.batch)
            add_vectors(index_path, con, symbol_ids, vecs)

    try:
        from .keyword_search import KeywordSearcher
        from .store import get_all_symbols_for_keyword_index

        keyword_searcher = KeywordSearcher(index_dir)
        if not os.path.exists(os.path.join(index_dir, FAISS_INDEX)) or not incremental:
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
        print(f"Processed {len(targets)} changed/new files")

    # Optionally index external docs
    if getattr(args, 'include_docs', False):
        try:
            from .docs_indexer import index_docs
            index_docs(index_dir=index_dir, repo_root=repo, embedder=embedder, verbose=args.verbose)
        except Exception as e:
            print(f"[WARN] Docs indexing failed: {e}")


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

    # Optionally include docs search and merge
    doc_results = []
    if getattr(args, 'include_docs', False):
        try:
            from .doc_search import search_docs
            doc_results = search_docs(index_dir, qvec, top_k=args.top_k)
        except Exception as e:
            print(f"[WARN] Doc search failed: {e}")

    merged = []
    if doc_results:
        docs_weight = getattr(args, 'docs_weight', 0.4)
        # Normalize and merge
        code_scores = [r[0] for r in results] or [1.0]
        doc_scores = [r[0] for r in doc_results] or [1.0]
        max_code = max(code_scores) if code_scores else 1.0
        max_doc = max(doc_scores) if doc_scores else 1.0
        for r in results:
            merged.append(((1.0 - docs_weight) * (r[0] / (max_code or 1.0)), ('code', r)))
        for r in doc_results:
            merged.append((docs_weight * (r[0] / (max_doc or 1.0)), ('doc', r)))
        merged.sort(key=lambda x: x[0], reverse=True)
        merged = merged[:args.top_k]
    else:
        merged = [(r[0], ('code', r)) for r in results]
    
    if not merged:
        print("No results.")
        return
    for _score, (rtype, r) in merged:
        if rtype == 'code':
            score, sid, (path, name, kind, start, end, sig) = r
            print(f"code score={score:.4f} | {kind} {name} @ {path}:{start}-{end}")
            if sig:
                print(f"  sig: {sig}")
        else:
            score, pid, (package, version, url, title) = r
            print(f"doc  score={score:.4f} | {package} :: {title} -> {url}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="semindex", description="Local Python semantic indexer (CPU-only)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="Index a Python repository")
    p_idx.add_argument("repo", help="Path to repository root")
    p_idx.add_argument("--index-dir", default=".semindex", help="Path to store index files (FAISS + SQLite)")
    p_idx.add_argument("--model", default=os.environ.get("SEMINDEX_MODEL", "microsoft/codebert-base"))
    p_idx.add_argument("--batch", type=int, default=16)
    p_idx.add_argument("--verbose", action="store_true")
    p_idx.add_argument("--include-docs", action="store_true", help="Also index external library docs")
    p_idx.add_argument(
        "--chunking",
        choices=['symbol', 'semantic'],
        default='symbol',
        help="Chunking method to use: 'symbol' for original function/class-based, 'semantic' for semantic-aware chunking",
    )
    p_idx.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for semantic chunking (0.0-1.0)",
    )
    p_idx.add_argument(
        "--language",
        default="auto",
        help="Language adapter to use (default: auto-detect per file)",
    )
    p_idx.add_argument(
        "--incremental",
        action="store_true",
        help="Perform incremental indexing by only processing changed files",
    )
    p_idx.set_defaults(func=cmd_index)

    p_q = sub.add_parser("query", help="Query the semantic index")
    p_q.add_argument("query", help="Natural language or code query")
    p_q.add_argument("--index-dir", default=".semindex")
    p_q.add_argument("--model", default=os.environ.get("SEMINDEX_MODEL", "microsoft/codebert-base"))
    p_q.add_argument("--top-k", type=int, default=10)
    p_q.add_argument("--hybrid", action="store_true", help="Use hybrid search (combines vector and keyword search)")
    p_q.add_argument("--include-docs", action="store_true", help="Include external docs in retrieval")
    p_q.add_argument("--docs-weight", type=float, default=0.4, help="Relative weight for docs vs code results (0-1)")
    p_q.set_defaults(func=cmd_query)

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)
