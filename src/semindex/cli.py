import argparse
import json
import os
import sqlite3
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
    get_callers,
    get_callees,
)
from .store import DOCS_FAISS_INDEX
from .docs.graph_builder import (
    build_module_graph,
    build_adapter_graph,
    build_pipeline_graph,
    build_repo_statistics,
)
from .ai import (
    cmd_ai_chat,
    cmd_ai_explain,
    cmd_ai_suggest,
    cmd_ai_generate,
    cmd_ai_docs,
    cmd_ai_find_bugs,
    cmd_ai_refactor,
    cmd_ai_tests,
)


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
    """Query the semantic index."""
    from .search import search_similar
    
    index_dir = os.path.abspath(args.index_dir)
    
    embedder = Embedder(model_name=args.model)
    qvec = embedder.encode([args.query])
    
    # Perform search
    try:
        if args.hybrid:
            from .hybrid_search import hybrid_search as _hybrid_search
            results = _hybrid_search(index_dir, qvec, args.query, top_k=args.top_k)
        else:
            results = search_similar(index_dir, qvec, top_k=args.top_k)
    except Exception as e:
        print(f"Search failed: {e}")
        return
    
    # Handle docs retrieval if requested
    if args.include_docs:
        try:
            from .doc_search import search_docs
            doc_results = search_docs(index_dir, qvec, top_k=args.top_k)
        except Exception:
            doc_results = []
        
        if doc_results:
            # Merge code + docs with weighted normalization
            merged = []
            code_scores = [r[0] for r in results] or [1.0]
            doc_scores = [r[0] for r in doc_results] or [1.0]
            max_code = max(code_scores) if code_scores else 1.0
            max_doc = max(doc_scores) if doc_scores else 1.0
            for r in results:
                merged.append(((1.0 - args.docs_weight) * (r[0] / (max_code or 1.0)), ("code", r)))
            for r in doc_results:
                merged.append((args.docs_weight * (r[0] / (max_doc or 1.0)), ("doc", r)))
            merged.sort(key=lambda x: x[0], reverse=True)
            merged = merged[:args.top_k]
            results = [r for _score, (_rtype, r) in merged]
    
    if not results:
        print("No results found.")
        return
    
    print(f"Found {len(results)} results for query: '{args.query}'")
    print()
    
    for i, (score, symbol_id, symbol_info) in enumerate(results, 1):
        path, name, kind, start_line, end_line, signature = symbol_info[:6]
        print(f"{i}. {name} ({kind}) in {path}:{start_line}-{end_line}")
        if signature:
            print(f"   Signature: {signature}")
        print(f"   Score: {score:.4f}")
        print()


def cmd_graph(args: argparse.Namespace) -> int:
    ensure_db(os.path.join(args.index_dir, DB_NAME))
    graphs = []
    if args.module:
        graphs.append(build_module_graph(args.repo, args.index_dir))
    if args.adapter:
        graphs.append(build_adapter_graph())
    if args.pipeline:
        graphs.append(build_pipeline_graph())
    if graphs:
        for graph in graphs:
            print(graph.to_markdown())

    if args.stats:
        stats = build_repo_statistics(args.repo, args.index_dir)
        print(json.dumps(stats, indent=2))

    if args.callers or args.callees:
        db_path = os.path.join(args.index_dir, DB_NAME)
        con = sqlite3.connect(db_path)
        con.execute("PRAGMA foreign_keys = ON;")
        try:
            cur = con.cursor()
            target_name = args.callers or args.callees
            cur.execute(
                "SELECT id, path, name FROM symbols WHERE name = ?",
                (target_name,),
            )
            rows = cur.fetchall()
            if not rows:
                print(f"Symbol '{target_name}' not found in index.")
            elif len(rows) > 1:
                print(
                    f"Symbol '{target_name}' is ambiguous. Please qualify with module path or namespace."
                )
            else:
                symbol_id, symbol_path, symbol_name = rows[0]
                if args.callers:
                    callers = get_callers(con, symbol_id)
                    if not callers:
                        print(f"No callers found for {symbol_name} ({symbol_path}).")
                    else:
                        print(f"Callers of {symbol_name} ({symbol_path}):")
                        for caller_id, caller_name, _callee_id, _callee_path in callers:
                            caller_info = cur.execute(
                                "SELECT path FROM symbols WHERE id = ?",
                                (caller_id,),
                            ).fetchone()
                            caller_path = caller_info[0] if caller_info else "<unknown>"
                            print(f" - {caller_name} ({caller_path})")
                if args.callees:
                    callees = get_callees(con, symbol_id)
                    if not callees:
                        print(f"No callees found for {symbol_name} ({symbol_path}).")
                    else:
                        print(f"Callees of {symbol_name} ({symbol_path}):")
                        for _callee_id, callee_name, callee_symbol_id, callee_path in callees:
                            if callee_symbol_id:
                                path_row = cur.execute(
                                    "SELECT path FROM symbols WHERE id = ?",
                                    (callee_symbol_id,),
                                ).fetchone()
                                callee_path_resolved = path_row[0] if path_row else callee_path
                            else:
                                callee_path_resolved = callee_path
                            label = callee_path_resolved or "<unknown path>"
                            print(f" - {callee_name} ({label})")
        finally:
            con.close()

    if not (graphs or args.stats or args.callers or args.callees):
        print(
            "No graph options selected. Use --module, --adapter, --pipeline, --stats, --callers, or --callees."
        )
    return 0

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="semindex", description="Local Python semantic indexer (CPU-only)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="Index a repository")
    p_idx.add_argument("repo", help="Path to repository to index")
    p_idx.add_argument("--index-dir", default=".semindex", help="Index directory (default: .semindex)")
    p_idx.add_argument(
        "--model",
        default=os.environ.get("SEMINDEX_MODEL", "microsoft/codebert-base"),
        help="Embedding model to use",
    )
    p_idx.add_argument("--batch", type=int, default=16, help="Batch size for embedding")
    p_idx.add_argument("--chunking", default="symbol", help="Chunking strategy (symbol|semantic)")
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
    p_idx.add_argument(
        "--include-docs",
        action="store_true",
        help="Also index external documentation (requires docs indexer deps)",
    )
    p_idx.add_argument("--hybrid", action="store_true", help="Enable hybrid search artifacts (deprecated; for compatibility)")
    p_idx.add_argument("--verbose", action="store_true", help="Verbose output")
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

    p_graph = sub.add_parser("graph", help="Generate graphs (Mermaid, metadata)")
    p_graph.add_argument("repo", nargs="?", default=os.getcwd(), help="Repository root (default: cwd)")
    p_graph.add_argument("--index-dir", default=".semindex", help="Index directory (default: .semindex)")
    p_graph.add_argument("--module", action="store_true", help="Build module graph")
    p_graph.add_argument("--adapter", action="store_true", help="Build adapter graph")
    p_graph.add_argument("--pipeline", action="store_true", help="Build pipeline graph")
    p_graph.add_argument("--stats", action="store_true", help="Show index statistics")
    p_graph.add_argument("--callers", type=str, help="Show callers of fully qualified symbol name")
    p_graph.add_argument("--callees", type=str, help="Show callees of fully qualified symbol name")
    p_graph.set_defaults(func=cmd_graph)

    # AI command parser
    p_ai = sub.add_parser("ai", help="AI-powered commands for code understanding and generation")
    ai_sub = p_ai.add_subparsers(dest="ai_cmd", required=True)
    
    # AI chat command
    p_ai_chat = ai_sub.add_parser("chat", help="Start an interactive AI chat session about the codebase")
    p_ai_chat.add_argument("--index-dir", default=".semindex", help="Index directory (default: .semindex)")
    p_ai_chat.add_argument("--model", default=os.environ.get("SEMINDEX_MODEL", "microsoft/codebert-base"))
    p_ai_chat.add_argument("--llm-path", help="Path to local LLM model")
    p_ai_chat.add_argument("--top-k", type=int, default=5, help="Number of context snippets to retrieve")
    p_ai_chat.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens for LLM response")
    p_ai_chat.add_argument("--hybrid", action="store_true", help="Use hybrid search for context retrieval")
    p_ai_chat.set_defaults(func=cmd_ai_chat)
    
    # AI explain command
    p_ai_explain = ai_sub.add_parser("explain", help="Explain code functionality")
    p_ai_explain.add_argument("target", help="Code element to explain (function, class, module, etc.)")
    p_ai_explain.add_argument("--index-dir", default=".semindex", help="Index directory (default: .semindex)")
    p_ai_explain.add_argument("--model", default=os.environ.get("SEMINDEX_MODEL", "microsoft/codebert-base"))
    p_ai_explain.add_argument("--llm-path", help="Path to local LLM model")
    p_ai_explain.add_argument("--top-k", type=int, default=5, help="Number of context snippets to retrieve")
    p_ai_explain.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens for LLM response")
    p_ai_explain.add_argument("--hybrid", action="store_true", help="Use hybrid search for context retrieval")
    p_ai_explain.set_defaults(func=cmd_ai_explain)
    
    # AI suggest command
    p_ai_suggest = ai_sub.add_parser("suggest", help="Suggest improvements to the codebase")
    p_ai_suggest.add_argument("--index-dir", default=".semindex", help="Index directory (default: .semindex)")
    p_ai_suggest.add_argument("--model", default=os.environ.get("SEMINDEX_MODEL", "microsoft/codebert-base"))
    p_ai_suggest.add_argument("--llm-path", help="Path to local LLM model")
    p_ai_suggest.add_argument("--top-k", type=int, default=5, help="Number of context snippets to retrieve")
    p_ai_suggest.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens for LLM response")
    p_ai_suggest.add_argument("--hybrid", action="store_true", help="Use hybrid search for context retrieval")
    p_ai_suggest.set_defaults(func=cmd_ai_suggest)
    
    # AI generate command
    p_ai_generate = ai_sub.add_parser("generate", help="Generate code based on description")
    p_ai_generate.add_argument("description", help="Description of the code to generate")
    p_ai_generate.add_argument("--index-dir", default=".semindex", help="Index directory (default: .semindex)")
    p_ai_generate.add_argument("--model", default=os.environ.get("SEMINDEX_MODEL", "microsoft/codebert-base"))
    p_ai_generate.add_argument("--llm-path", help="Path to local LLM model")
    p_ai_generate.add_argument("--top-k", type=int, default=5, help="Number of context snippets to retrieve")
    p_ai_generate.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens for LLM response")
    p_ai_generate.add_argument("--hybrid", action="store_true", help="Use hybrid search for context retrieval")
    p_ai_generate.add_argument("--include-context", action="store_true", help="Include relevant code context in generation")
    p_ai_generate.set_defaults(func=cmd_ai_generate)
    
    # AI docs command
    p_ai_docs = ai_sub.add_parser("docs", help="Generate documentation for code elements")
    p_ai_docs.add_argument("target", help="Code element to document (function, class, module, etc.)")
    p_ai_docs.add_argument("--index-dir", default=".semindex", help="Index directory (default: .semindex)")
    p_ai_docs.add_argument("--model", default=os.environ.get("SEMINDEX_MODEL", "microsoft/codebert-base"))
    p_ai_docs.add_argument("--llm-path", help="Path to local LLM model")
    p_ai_docs.add_argument("--top-k", type=int, default=5, help="Number of context snippets to retrieve")
    p_ai_docs.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens for LLM response")
    p_ai_docs.add_argument("--hybrid", action="store_true", help="Use hybrid search for context retrieval")
    p_ai_docs.set_defaults(func=cmd_ai_docs)
    
    # AI find bugs command
    p_ai_bugs = ai_sub.add_parser("bugs", help="Find potential bugs in code")
    p_ai_bugs.add_argument("target", nargs="?", help="Specific code element to analyze for bugs (optional)")
    p_ai_bugs.add_argument("--index-dir", default=".semindex", help="Index directory (default: .semindex)")
    p_ai_bugs.add_argument("--model", default=os.environ.get("SEMINDEX_MODEL", "microsoft/codebert-base"))
    p_ai_bugs.add_argument("--llm-path", help="Path to local LLM model")
    p_ai_bugs.add_argument("--top-k", type=int, default=5, help="Number of context snippets to retrieve")
    p_ai_bugs.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens for LLM response")
    p_ai_bugs.add_argument("--hybrid", action="store_true", help="Use hybrid search for context retrieval")
    p_ai_bugs.set_defaults(func=cmd_ai_find_bugs)
    
    # AI refactoring command
    p_ai_refactor = ai_sub.add_parser("refactor", help="Suggest refactoring opportunities")
    p_ai_refactor.add_argument("target", nargs="?", help="Specific code element to analyze for refactoring (optional)")
    p_ai_refactor.add_argument("--index-dir", default=".semindex", help="Index directory (default: .semindex)")
    p_ai_refactor.add_argument("--model", default=os.environ.get("SEMINDEX_MODEL", "microsoft/codebert-base"))
    p_ai_refactor.add_argument("--llm-path", help="Path to local LLM model")
    p_ai_refactor.add_argument("--top-k", type=int, default=5, help="Number of context snippets to retrieve")
    p_ai_refactor.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens for LLM response")
    p_ai_refactor.add_argument("--hybrid", action="store_true", help="Use hybrid search for context retrieval")
    p_ai_refactor.set_defaults(func=cmd_ai_refactor)
    
    # AI tests command
    p_ai_tests = ai_sub.add_parser("tests", help="Generate unit tests for code")
    p_ai_tests.add_argument("target", help="Code element to generate tests for")
    p_ai_tests.add_argument("--framework", default="pytest", help="Testing framework to use (default: pytest)")
    p_ai_tests.add_argument("--index-dir", default=".semindex", help="Index directory (default: .semindex)")
    p_ai_tests.add_argument("--model", default=os.environ.get("SEMINDEX_MODEL", "microsoft/codebert-base"))
    p_ai_tests.add_argument("--llm-path", help="Path to local LLM model")
    p_ai_tests.add_argument("--top-k", type=int, default=5, help="Number of context snippets to retrieve")
    p_ai_tests.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens for LLM response")
    p_ai_tests.add_argument("--hybrid", action="store_true", help="Use hybrid search for context retrieval")
    p_ai_tests.set_defaults(func=cmd_ai_tests)

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)
