import os
from typing import List, Sequence, Tuple

from .embed import Embedder
from .languages import (
    collect_index_targets,
    ensure_default_adapters,
)
from .languages.base import ParseResult
from .model import Chunk, ChunkingConfig, Symbol
from .crawler import read_text
from .store import (
    DB_NAME,
    FAISS_INDEX,
    add_vectors,
    add_calls,
    ensure_db,
    db_conn,
    reset_index,
    get_changed_files,
    file_sha256_from_content,
)


class Indexer:
    def __init__(
        self,
        index_dir: str = ".semindex",
        model: str | None = None,
    ) -> None:
        self.index_dir = os.path.abspath(index_dir)
        os.makedirs(self.index_dir, exist_ok=True)
        self.model = model or os.environ.get("SEMINDEX_MODEL", "microsoft/codebert-base")
        self.embedder = Embedder(model_name=self.model)

    def _make_module_symbol(self, path: str, source: str, language_name: str) -> Symbol:
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

    def index_path(
        self,
        repo: str,
        incremental: bool = True,
        *,
        language: str = "auto",
        include_docs: bool = False,
        chunking: str = "symbol",
        similarity_threshold: float = 0.7,
        batch: int = 16,
        verbose: bool = False,
    ) -> None:
        """
        Index a repository path into the index directory.
        Mirrors the CLI behavior of `semindex index`.
        """
        repo = os.path.abspath(repo)
        os.makedirs(self.index_dir, exist_ok=True)

        db_path = os.path.join(self.index_dir, DB_NAME)
        index_path = os.path.join(self.index_dir, FAISS_INDEX)
        ensure_db(db_path)

        chunk_config = ChunkingConfig(
            method=chunking,
            similarity_threshold=similarity_threshold,
        )

        ensure_default_adapters()

        if incremental and os.path.exists(db_path) and os.path.exists(index_path):
            changed_files = get_changed_files(repo, db_path, language=language)
            # resolved targets already come as (adapter, path) pairs from collect_index_targets in CLI, but here we use languages API directly
            targets = [(adapter.name, adapter, path) for adapter, path in collect_index_targets(repo, language) if path in set(changed_files)]
            if verbose:
                print(f"Found {len(targets)} changed/new files for incremental indexing")
        else:
            base_targets = collect_index_targets(repo, language)
            targets = [(adapter.name, adapter, path) for adapter, path in base_targets]
            # Fresh index
            reset_index(self.index_dir, dim=self.embedder.model.config.hidden_size)
            if verbose:
                print(f"Performing fresh indexing of {len(targets)} files")

        all_symbol_rows: List[Tuple[str, str, str, int, int, str, str, str, str, str, str, str]] = []
        all_texts: List[str] = []
        file_hashes: List[Tuple[str, str, str]] = []
        call_records: List[Tuple[str, str, str]] = []

        for language_name, adapter, path in targets:
            try:
                source = read_text(path)
            except Exception as exc:
                if verbose:
                    print(f"[WARN] Failed to read {path}: {exc}")
                continue

            current_hash = file_sha256_from_content(source.encode("utf-8"))

            parse_failed = False
            calls_local: List[Tuple[str, str]] = []
            try:
                result: ParseResult = adapter.process_file(
                    path,
                    source,
                    self.embedder,
                    chunk_config,
                )
                calls_local = result.calls or []
            except Exception as exc:
                parse_failed = True
                if verbose:
                    print(f"[WARN] Failed to process {path}: {exc}")

            if parse_failed:
                symbols = [self._make_module_symbol(path, source, language_name)]
                chunks = [Chunk(symbol=symbols[0], text=source)]
                calls_local = []
            else:
                symbols = list(result.symbols)
                chunks = list(result.chunks)
                if not symbols:
                    symbols.append(self._make_module_symbol(path, source, language_name))

            if not chunks:
                # Fallback minimal chunk
                chunks = [Chunk(symbol=symbols[0], text=source)] if symbols else []

            # format rows like CLI
            def _format_symbol_row(symbol: Symbol):
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

            for symbol in symbols:
                all_symbol_rows.append(_format_symbol_row(symbol))

            for chunk in chunks:
                all_texts.append(chunk.text)

            file_hashes.append((path, current_hash, language_name))

            for caller, callee in calls_local:
                if caller and callee:
                    call_records.append((path, caller, callee))

        if not (all_symbol_rows or file_hashes or all_texts):
            if verbose:
                print("No targets to index.")
            return

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
                vecs = self.embedder.encode(all_texts, batch_size=batch)
                add_vectors(index_path, con, symbol_ids, vecs)

            if call_records:
                cur.execute("SELECT id, path, name, signature, kind FROM symbols")
                symbol_rows = cur.fetchall()
                symbol_id_map: dict[Tuple[str, str], int] = {}
                symbols_by_name: dict[str, List[Tuple[int, str]]] = {}
                for sid, spath, sname, ssignature, skind in symbol_rows:
                    symbol_id_map[(spath, sname)] = sid
                    symbols_by_name.setdefault(sname, []).append((sid, spath))

                call_rows: List[Tuple[int, str, Optional[int], Optional[str]]] = []
                seen_edges: set[Tuple[int, str, Optional[int]]] = set()
                for path, caller_name, callee_name in sorted(call_records):
                    caller_id = symbol_id_map.get((path, caller_name))
                    if not caller_id:
                        continue
                    callee_id: Optional[int] = None
                    callee_path: Optional[str] = None

                    candidate = symbol_id_map.get((path, callee_name))
                    if candidate:
                        callee_id = candidate
                        callee_path = path
                    else:
                        matches = symbols_by_name.get(callee_name) or []
                        if len(matches) == 1:
                            callee_id = matches[0][0]
                            callee_path = matches[0][1]

                    edge_key = (caller_id, callee_name, callee_id)
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
                    call_rows.append((caller_id, callee_name, callee_id, callee_path))

                if call_rows:
                    add_calls(con, call_rows)

        # Optional keyword index (best-effort)
        try:
            from .keyword_search import KeywordSearcher
            from .store import get_all_symbols_for_keyword_index

            keyword_searcher = KeywordSearcher(self.index_dir)
            if not os.path.exists(os.path.join(self.index_dir, FAISS_INDEX)) or not incremental:
                keyword_searcher.create_index()

            with db_conn(db_path) as con:
                all_symbols = get_all_symbols_for_keyword_index(con)
                for symbol in all_symbols:
                    try:
                        with open(symbol["path"], 'r', encoding='utf-8') as f:
                            source = f.read()
                            start_line = max(symbol["start_line"] - 1, 0)
                            end_line = min(symbol["end_line"], source.count("\n") + 1)
                            content = "\n".join(source.splitlines()[start_line:end_line])
                            symbol["content"] = content
                    except Exception as e:
                        if verbose:
                            print(f"[WARN] Could not read content for {symbol['path']}: {e}")
                        symbol["content"] = ""
                keyword_searcher.bulk_index_symbols(all_symbols)
            if verbose:
                print(f"Also indexed in keyword search: {len(all_symbol_rows)} symbols")
        except ImportError:
            if verbose:
                print("[WARN] Elasticsearch not available, skipping keyword indexing")
        except Exception as e:
            if verbose:
                print(f"[WARN] Could not connect to Elasticsearch: {e}, skipping keyword indexing")

        if verbose:
            print(f"Indexed {len(all_texts)} chunks from repository {repo}")
            if incremental:
                print(f"Processed {len(targets)} changed/new files")

        if include_docs:
            try:
                from .docs_indexer import index_docs
                index_docs(index_dir=self.index_dir, repo_root=repo, embedder=self.embedder, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"[WARN] Docs indexing failed: {e}")
