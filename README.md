# semindex (Python-first, CPU-only)

Advanced local semantic codebase indexer for Python (with optional Tree-sitter
languages) using AST + embeddings.

- Language-aware adapters with per-file metadata (language, namespace, symbol type)

- Pluggable language adapter registry with automatic file-type discovery
- Optional Tree-sitter powered JavaScript adapter
- AST extraction via Python `ast`
- Embeddings via HuggingFace Transformers (CPU)
- Vector search via FAISS (CPU)
- Keyword search via Elasticsearch
- Hybrid search with Reciprocal Rank Fusion
- Advanced semantic-aware chunking with CAST algorithm
- Incremental indexing by file hash
- Metadata/XRef via SQLite (basic)
- External library documentation indexing (PyPI/local site-packages) stored in a separate FAISS + SQLite space and merged at query time

## Install

On Windows (Python >= 3.9):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -e .  # uses pyproject.toml
```

If `-e .` fails, use:

```powershell
pip install -r requirements.txt
pip install -e .
```

The first run will download the model locally (one-time). Afterwards it runs fully offline.

## Using uv (recommended)

`uv` is a fast Python package manager. If you have `uv` installed:

```powershell
# Create and activate a virtualenv managed by uv
uv venv
.\.venv\Scripts\activate

# Install project (prod deps)
uv pip install -e .

# Install dev/test extras
uv pip install -e .[dev]

# Run tests
uv run pytest
```

## Usage

```powershell
# Index a repo with automatic language detection
semindex index <path-to-repo> --index-dir .semindex

# Force a specific adapter by name (e.g. javascript)
semindex index <path-to-repo> --language javascript

# Index with incremental updates (only changed files)
semindex index <path-to-repo> --index-dir .semindex --incremental

# Index with semantic-aware chunking
semindex index <path-to-repo> --index-dir .semindex --chunking semantic --similarity-threshold 0.7

# Index external library docs alongside code
semindex index <path-to-repo> --index-dir .semindex --include-docs --verbose

# Query with vector search (default)
semindex query "how to open a file" --index-dir .semindex

# Query with hybrid search (vector + keyword)
semindex query "how to open a file" --index-dir .semindex --hybrid

# Query including external docs merged with code
semindex query "fastapi router" --index-dir .semindex --include-docs --docs-weight 0.4
```

Indexing options:
- `--chunking` choose chunking method: `symbol` (function/class-based, default) or `semantic` (CAST algorithm)
- `--similarity-threshold` similarity threshold for semantic chunking (0.0-1.0, default 0.7)
- `--incremental` perform incremental indexing, only processing changed files
- `--language` select a registered adapter (`python`, `javascript`, etc.) or leave as `auto` (default) to detect per file based on extension. Auto-detection will skip files without a matching adapter.
- `--model` override default model (`microsoft/codebert-base`). For alternative models, consider:
  - `Salesforce/codet5-base` - CodeT5 model for code understanding
  - `BAAI/bge-large-en-v1.5` - Better general-purpose model
  - `sentence-transformers/all-MiniLM-L6-v2` - Lightweight general model
  Set env `SEMINDEX_MODEL` to persist.
- `--batch` controls embed batch size.

Query options:
- `--hybrid` enable hybrid search combining vector and keyword search
- `--model` specify the model to use for encoding the query
- `--top-k` number of results to return (default 10)
- `--include-docs` include external library docs results
- `--docs-weight` weight (0-1) applied when merging docs vs code results (default 0.4)

## Programmatic API

Use `Indexer` and `Searcher` directly from Python:

```python
from semindex.indexer import Indexer
from semindex.search import Searcher

# Build or update index
indexer = Indexer(index_dir=".semindex")  # optional: model="microsoft/codebert-base"
indexer.index_path(
    "src/",
    incremental=True,
    language="auto",
    include_docs=False,
    chunking="symbol",           # or "semantic"
    similarity_threshold=0.7,
    batch=16,
    verbose=True,
)

# Query the index
searcher = Searcher(index_dir=".semindex")  # optional: model="..."
results = searcher.query(
    "how is user auth implemented?",
    hybrid=True,          # vector + keyword; falls back to vector-only if keyword backend unavailable
    include_docs=False,
    top_k=10,
    docs_weight=0.4,
)

for score, symbol_id, (path, name, kind, start, end, sig) in results:
    print(f"{score:.4f} | {kind} {name} @ {path}:{start}-{end}")
```

## External Library Documentation

When `--include-docs` is used during indexing and/or querying, semindex:

- Downloads documentation entry-points discovered from the PyPI JSON API (Documentation/Homepage/ReadTheDocs links) for packages listed in `requirements.txt`.
- Discovers local docs under `site-packages/*/(docs|doc|documentation)` and indexes `.md` and `.html`.
- Parses HTML via `beautifulsoup4` to strip boilerplate and extract readable text.
- Stores docs in separate tables (`doc_packages`, `doc_pages`, `doc_vectors`) and a separate FAISS index `docs.faiss`.
- Merges results with code search using score normalization and a configurable `--docs-weight`.

Notes:
- Network is required for PyPI doc discovery; failures are logged and skipped.
- A checksum is stored per page to support incremental doc indexing.

## Notes

- Index can be rebuilt fresh or updated incrementally using file hash comparison.
- Chunking can be done per function/method/class or using semantic-aware chunking.
- Search supports both vector-only and hybrid (vector + keyword) modes.
- Search returns top-k symbols with scores from similarity or RRF ranking.
- SQLite `symbols` table now tracks `language`, `namespace`, `symbol_type`, and `bases` (serialized) per entry for richer metadata.
- Documentation tables: `doc_packages`, `doc_pages`, with vectors in `doc_vectors` and FAISS file `docs.faiss`.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the project roadmap.

