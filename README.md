# semindex (Python-only, CPU-only)

Local semantic codebase indexer for Python using AST + embeddings.

- AST extraction via Python `ast`
- Embeddings via HuggingFace Transformers (CPU)
- Vector search via FAISS (CPU)
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

# Query
semindex query "how to open a file" --index-dir .semindex
```

Options:
- `--model` override default model (`BAAI/bge-small-en-v1.5`). Set env `SEMINDEX_MODEL` to persist.
- `--batch` controls embed batch size.

## Notes

- Current index is rebuilt fresh each run for simplicity.
- Chunking is per function/method/class; falls back to module chunk.
- Search returns top-k symbols with cosine similarity scores.

## Roadmap

- Incremental indexing by file hash
- Better code embeddings (code-specific model)
- Who-calls/used-by graph exploration
- Reranking with keyword + structure signals
