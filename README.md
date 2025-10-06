# semindex (Python-only, CPU-only)

Advanced local semantic codebase indexer for Python using AST + embeddings.

- AST extraction via Python `ast`
- Embeddings via HuggingFace Transformers (CPU)
- Vector search via FAISS (CPU)
- Keyword search via Elasticsearch
- Hybrid search with Reciprocal Rank Fusion
- Advanced semantic-aware chunking with CAST algorithm
- Incremental indexing by file hash
- Metadata/XRef via SQLite (basic)

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
# Index a repo
semindex index <path-to-repo> --index-dir .semindex

# Index with incremental updates (only changed files)
semindex index <path-to-repo> --index-dir .semindex --incremental

# Index with semantic-aware chunking
semindex index <path-to-repo> --index-dir .semindex --chunking semantic --similarity-threshold 0.7

# Query with vector search (default)
semindex query "how to open a file" --index-dir .semindex

# Query with hybrid search (vector + keyword)
semindex query "how to open a file" --index-dir .semindex --hybrid
```

Indexing options:
- `--chunking` choose chunking method: `symbol` (function/class-based, default) or `semantic` (CAST algorithm)
- `--similarity-threshold` similarity threshold for semantic chunking (0.0-1.0, default 0.7)
- `--incremental` perform incremental indexing, only processing changed files
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

## Notes

- Index can be rebuilt fresh or updated incrementally using file hash comparison.
- Chunking can be done per function/method/class or using semantic-aware chunking.
- Search supports both vector-only and hybrid (vector + keyword) modes.
- Search returns top-k symbols with scores from similarity or RRF ranking.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the project roadmap.

