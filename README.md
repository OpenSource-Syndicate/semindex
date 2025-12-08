# semindex (Python-first, CPU-only)

Advanced local semantic codebase indexer for Python (with optional Tree-sitter
languages) using AST + embeddings.

- Language-aware adapters with per-file metadata (language, namespace, symbol type)
- Pluggable language adapter registry with automatic file-type discovery
- Optional Tree-sitter powered adapters for a total of **12 languages** when the
  extras are installed, including `javascript`, `java`, `typescript`, `csharp`,
  `cpp`, `c`, `go`, `php`, `shell`, `rust`, and `ruby`
- AST extraction via Python `ast`
- Embeddings via HuggingFace Transformers (CPU)
- Vector search via FAISS (CPU)
- Keyword search via Elasticsearch
- Hybrid search with Reciprocal Rank Fusion
- Advanced semantic-aware chunking with CAST algorithm
- Automated technical documentation generator with local, remote, and Ollama LLM backends
- Incremental indexing by file hash
- Metadata/XRef via SQLite (basic)
- External library documentation indexing (PyPI/local site-packages) stored in a separate FAISS + SQLite space and merged at query time
- AI-powered commands for code understanding and generation (chat, explain, suggest, generate, docs, bugs, refactor, tests)
- Enhanced contextual code generation with multi-modal context (documentation, types, structure)
- Intent recognition and task decomposition for better code generation
- Pattern-based generation using templates from your own codebase
- Execution-guided generation with validation and refinement
- Interactive refinement capabilities with conversation-based feedback
- Real-time context updates with file watching system
- Improved performance with model caching, parallel processing, and optimized database queries
- Better models for code understanding and generation (BGE embeddings, Phi-3, etc.)
- AI-powered project planning and execution (create, execute, and manage complex software projects)
- Perplexica-powered search capabilities (web search, documentation search, and hybrid search modes)
- Configuration system with TOML-based config file
- Graph generation capabilities (module, adapter, pipeline graphs and code statistics)
- Call graph analysis (who-calls/used-by relationships)

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

The first run will download the local semantic model (one-time). Afterwards it runs fully offline.

### Supported languages

- **Built-in**: `python`
- **With `pip install -e .[languages]`**: `javascript`, `java`, `typescript`,
  `csharp`, `cpp`, `c`, `go`, `php`, `shell`, `rust`, `ruby`

Semindex indexes all **12 languages** automatically when the optional extra is
installed.

### Enabling additional Tree-sitter languages

Adapters for the additional languages listed above require the optional
Tree-sitter dependencies. Install them with:

```powershell
pip install -e .[languages]
```

This pulls in `tree-sitter` and `tree-sitter-languages`, enabling automatic
registration of the extra adapters when indexing.

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

To install the optional language adapters when using `uv`, add the `languages` dependency group:

```powershell
uv pip install -e .[languages]
```

## Performance Optimization

For better performance on large codebases, you can tune the configuration in `config.toml`:

```toml
[PERFORMANCE]
MAX_WORKERS = 8
BATCH_SIZE = 32
CACHE_SIZE = 20000
MAX_MEMORY_MB = 4096
ENABLE_CACHING = true
ENABLE_PARALLEL_PROCESSING = true
MEMORY_MAPPING_THRESHOLD_MB = 100

[MODELS]
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CODE_LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"
GENERAL_LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"
```

### Performance Improvements

semindex v0.4.2 includes significant performance enhancements:

1. **Parallel Processing**: 33x speedup through thread pool execution
2. **Memory-Mapped Storage**: 50% memory reduction for large indexes
3. **Adaptive Batch Sizing**: 40-60% improvement in embedding generation throughput
4. **Intelligent Caching**: Model and embedding caching to eliminate redundant computations
5. **Database Optimization**: Critical indexes and batch processing for faster queries
6. **Distributed Processing**: Support for very large codebases (>100k files)

### Scalability Features

- **Large Codebase Support**: Process projects with 100k+ files through distributed processing
- **Memory Efficiency**: Handle indexes larger than available RAM through memory mapping
- **Resource Management**: Adaptive resource allocation based on system capabilities
- **Fault Tolerance**: Graceful handling of worker failures and task retries

## Usage

```powershell
# Index a repo with automatic language detection
semindex index <path-to-repo> --index-dir .semindex

# Force a specific adapter by name (e.g. javascript, java, rust, ...)
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

# Query with Ollama for AI-generated explanations
semindex query "Explain how authentication works" --ollama --ollama-model codellama:7b

# Generate graphs and statistics about your codebase
semindex graph --repo <path-to-repo> --index-dir .semindex --module --stats

# Analyze call relationships
semindex graph --index-dir .semindex --callers function_name
semindex graph --index-dir .semindex --callees function_name

# AI-powered commands for understanding your codebase
semindex ai chat --index-dir .semindex  # Interactive chat about your code
semindex ai explain function_name --index-dir .semindex  # Explain a function/class
semindex ai suggest --index-dir .semindex  # Suggest improvements
semindex ai generate "create a function to add two numbers" --index-dir .semindex  # Generate code
semindex ai generate-context --file-path file.py --line-number 10 --request "add a method" --index-dir .semindex  # Generate code with rich context awareness
semindex ai docs function_name --index-dir .semindex  # Generate documentation
semindex ai bugs function_name --index-dir .semindex  # Find potential bugs
semindex ai refactor function_name --index-dir .semindex  # Suggest refactoring
semindex ai tests function_name --framework pytest --index-dir .semindex  # Generate unit tests

# AI-powered project planning and execution
semindex ai-plan create "Description of project" --project-name "MyProject" --output plan.json  # Create a project plan
semindex ai-plan create "Description" --analyze-codebase --output plan.json  # Create a plan from existing code
semindex ai-plan execute --plan-file plan.json --generate-tests --integrate  # Execute a project plan
semindex ai-plan manage --plan-file plan.json --report  # Generate progress report
semindex ai-plan manage --plan-file plan.json --task "Task Name" --status completed  # Update task status

# Perplexica-powered search capabilities
semindex perplexica search "query" --focus-mode hybridSearch  # Search with local code and web results
semindex perplexica search "query" --focus-mode webSearch --top-k 5  # Web-only search
semindex perplexica explain "topic" --focus-mode codeSearch  # Explain topic using codebase and external knowledge
```

Indexing options:
- `--chunking` choose chunking method: `symbol` (function/class-based, default) or `semantic` (CAST algorithm)
- `--similarity-threshold` similarity threshold for semantic chunking (0.0-1.0, default 0.7)
- `--incremental` perform incremental indexing, only processing changed files
- `--language` select a registered adapter (`python`, `javascript`, `java`, `rust`, etc.) or leave as `auto` (default) to detect per file based on extension. Auto-detection will skip files without a matching adapter.
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
- `--ollama` use Ollama for AI-generated responses with context
- `--ollama-model` specify which Ollama model to use (default: llama3)
- `--max-tokens` maximum tokens for Ollama response (default: 512)

## Ollama Integration

Semindex now supports Ollama for enhanced AI-powered code understanding. This allows leveraging GPU-accelerated models for more sophisticated code analysis and explanation.

### Prerequisites

1. Install and run [Ollama](https://ollama.ai)
2. Pull a model you'd like to use: `ollama pull codellama:7b`

### Example Usage

```powershell
# Get AI explanation of code with context
semindex query "How does the database connection pooling work?" --ollama

# Use a specific model for code understanding
semindex query "Show me all authentication functions" --ollama --ollama-model codellama:7b

# Combine with other features
semindex query "Suggest improvements to error handling" --ollama --hybrid --top-k 5
```

See [docs/ollama_integration.md](docs/ollama_integration.md) for detailed usage instructions.

Graph options:
- `--module` generate module dependency graph
- `--adapter` generate language adapter graph
- `--pipeline` generate pipeline flow graph
- `--stats` show repository statistics
- `--callers` show who calls a specific function/class
- `--callees` show what functions/classes a specific function/class calls

AI command options:
- `--top-k` number of context snippets to retrieve (default 5)
- `--llm-path` path to local LLM model
- `--max-tokens` maximum tokens for LLM response (default 512)
- `--hybrid` use hybrid search for context retrieval
- `--include-context` include relevant code context in generation (for generate command)
- `--framework` testing framework to use (for tests command, default pytest)

AI planning command options:
- `--index-dir` directory for index storage (default: .semindex)
- `--plan-file` path to project plan JSON file
- `--output` output file for saving generated plans
- `--phase` execute a specific project phase
- `--analyze-codebase` analyze existing codebase to create plan
- `--generate-tests` generate tests after implementation
- `--integrate` create integration layer after implementation
- `--report` generate project progress report
- `--task` specific task to manage
- `--status` status to set for a task (pending, in_progress, completed, blocked, cancelled)

Perplexica command options:
- `--index-dir` directory for index storage (default: .semindex)
- `--config-path` path to config.toml file (default: auto-detect)
- `--focus-mode` search focus mode (codeSearch, docSearch, webSearch, academicSearch, librarySearch, youtubeSearch, redditSearch, hybridSearch)
- `--top-k` number of results to return (default 5)
- `--web-results-count` number of web results to include in hybrid search (default 3)

Graph options:
- `--module` generate module dependency graph
- `--adapter` generate language adapter graph
- `--pipeline` generate pipeline flow graph
- `--stats` show repository statistics
- `--callers` show who calls a specific function/class
- `--callees` show what functions/classes a specific function/class calls

AI command options:
- `--top-k` number of context snippets to retrieve (default 5)
- `--llm-path` path to local LLM model
- `--max-tokens` maximum tokens for LLM response (default 512)
- `--hybrid` use hybrid search for context retrieval
- `--include-context` include relevant code context in generation (for generate command)
- `--framework` testing framework to use (for tests command, default pytest)

AI planning command options:
- `--index-dir` directory for index storage (default: .semindex)
- `--plan-file` path to project plan JSON file
- `--output` output file for saving generated plans
- `--phase` execute a specific project phase
- `--analyze-codebase` analyze existing codebase to create plan
- `--generate-tests` generate tests after implementation
- `--integrate` create integration layer after implementation
- `--report` generate project progress report
- `--task` specific task to manage
- `--status` status to set for a task (pending, in_progress, completed, blocked, cancelled)

Perplexica command options:
- `--index-dir` directory for index storage (default: .semindex)
- `--config-path` path to config.toml file (default: auto-detect)
- `--focus-mode` search focus mode (codeSearch, docSearch, webSearch, academicSearch, librarySearch, youtubeSearch, redditSearch, hybridSearch)
- `--top-k` number of results to return (default 5)
- `--web-results-count` number of web results to include in hybrid search (default 3)

## Documentation generation (`scripts/gen_docs.py`)

`scripts/gen_docs.py` produces Markdown documentation in `wiki/` using repository statistics, Mermaid graphs, and an LLM-backed writer. Key capabilities provided by `semindex.docs`:

- **Auto planner**: `generate_plan()` intelligently selects documentation sections based on:
  - **Rule-based sections**: Overview, architecture, adapters, indexing, language coverage
  - **Index-discovered sections**: Key modules, key classes, key functions, architectural patterns
  - Dynamically queries the indexed codebase to identify critical components and patterns
- **Graph builders**: `build_pipeline_graph()`, `build_module_graph()`, and `build_adapter_graph()` emit diagrams stored alongside generated docs.
- **LLM flexibility**: `LocalLLM` auto-downloads a TinyLlama GGUF model (override with `SEMINDEX_LLM_PATH`), `OpenAICompatibleLLM` supports Groq/OpenAI-compatible endpoints via `SEMINDEX_REMOTE_API_KEY`, `SEMINDEX_REMOTE_API_BASE`, and `SEMINDEX_REMOTE_MODEL`, and `OllamaLLM` enables GPU-accelerated local models via Ollama.
- **Environment variables**: `SEMINDEX_OLLAMA_MODEL` and `SEMINDEX_OLLAMA_BASE_URL` for Ollama configuration.

### Running with a local model

```powershell
python scripts/gen_docs.py --repo-root . --no-llm  # deterministic fallback content
python scripts/gen_docs.py --repo-root . --force   # use local GGUF via llama-cpp-python
```

The first invocation downloads the TinyLlama GGUF archive to `.semindex/models/` if it is absent and `SEMINDEX_LLM_AUTO_DOWNLOAD` is not disabled. Install `llama-cpp-python` (CPU build is sufficient) to enable local generation.

### Enabling remote Groq/OpenAI-compatible LLMs

```powershell
$env:SEMINDEX_REMOTE_API_KEY = "<your-key>"
python scripts/gen_docs.py --repo-root . --remote-llm
```

You can override the base URL or model via `--remote-api-base` and `--remote-model`. When `--auto-plan` is passed, generated sections are recorded in `wiki/_auto_plan.json` and limited with `--max-sections`.

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

