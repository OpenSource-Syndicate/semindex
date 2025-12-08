# Changelog

## [0.5.0] - 2025-12-08

### Added
- Implemented Ollama integration for AI-powered code explanations with GPU acceleration
  - New `ollama_llm.py` module with `OllamaLLM` class for communication with local Ollama service
  - Added `--ollama` flag to enable Ollama-enhanced query responses
  - Added `--ollama-model` parameter to specify which Ollama model to use (default: llama3)
  - Added `--max-tokens` parameter to control response length
  - New `generate_answer_ollama()` function in `rag.py` for Ollama-based RAG responses
  - Comprehensive documentation in `docs/ollama_integration.md`
  - Full test suite for Ollama integration in `tests/test_ollama.py`, `tests/test_rag_ollama.py`, and `tests/test_cli_ollama.py`
- Implemented `cmd_query()` CLI function for semantic search with support for hybrid search and documentation merging
- Enhanced autoplan with index-driven section discovery:
  - `_discover_key_modules()`: Identifies core modules by symbol count
  - `_discover_key_classes()`: Finds important classes by structural complexity
  - `_discover_key_functions()`: Locates critical functions via call graph analysis
  - `_discover_patterns()`: Detects architectural patterns (testing, configuration, API, data layer)
- Added `PlanningRule` dataclass for declarative section planning rules
- Refactored `generate_plan()` to support both rule-based and index-discovered sections

### Changed
- Autoplan now generates documentation sections dynamically from indexed codebase rather than hardcoded templates
- `generate_plan()` accepts optional `index_dir` parameter to enable index-based discovery
- Improved code organization with separation of concerns in planning logic
- README.md updated to reflect index-driven documentation planning and Ollama integration
- CLI argument parser enhanced with new Ollama-specific options

### Fixed
- Fixed missing `cmd_query` function that was causing NameError in CLI
- Added missing imports (`json`, `sqlite3`) to cli.py
- Fixed unpacking error in query result formatting (6 fields instead of 7)

## [0.4.0] - 2025-11-15

### Added
- External library documentation indexing (PyPI + local site-packages)
- New public Python wrappers: `Indexer` and `Searcher` for programmatic indexing and querying
- New `semindex.docs` package with automated documentation generation
- Added `scripts/gen_docs.py` for generating wiki documentation
- Implemented `LocalLLM` with automatic model download for offline documentation generation
- Added `OpenAICompatibleLLM` for remote LLM integration

### Changed
- Extended `store.py` schema to manage docs-specific tables and FAISS index
- `cli.py` updated to support docs indexing and retrieval
- Improved error handling for external service failures

## [0.3.0] - 2025-10-20

### Added
- Added a pluggable language adapter registry in `semindex.languages` that
  powers automatic discovery of supported file types and supports runtime
  registration of custom adapters.
- Introduced an optional Tree-sitter powered JavaScript adapter that is
  registered when `tree_sitter_languages` is available, expanding
  multi-language indexing support.
- Expanded Tree-sitter backed adapters to cover `javascript`, `java`,
  `typescript`, `csharp`, `cpp`, `c`, `go`, `php`, `shell`, `rust`, and `ruby`
  (12 languages total with extras) and upgraded the JavaScript adapter to emit
  class/function symbols via the Tree-sitter AST.
- External library documentation indexing (PyPI + local site-packages). Docs are parsed (HTML/Markdown), normalized, embedded, and stored in dedicated tables (`doc_packages`, `doc_pages`, `doc_vectors`) and a separate FAISS index `docs.faiss`. CLI: `--include-docs` for `index` and `query`, with `--docs-weight` to control ranking merge.
- New public Python wrappers: `Indexer` (`semindex.indexer.Indexer`) and `Searcher` (`semindex.search.Searcher`) for programmatic indexing and querying, including hybrid search and optional docs merging.
- New `semindex.docs` package exposing `generate_plan()`, graph builders, and Mermaid utilities to power automated documentation.
- Added `scripts/gen_docs.py` CLI for generating wiki documentation from graphs, repo statistics, and LLM-authored narratives.
- Implemented `LocalLLM` with automatic TinyLlama GGUF download (configurable via `SEMINDEX_LLM_*` env vars) for offline documentation generation.
- Added `remote_llm.py` with `OpenAICompatibleLLM` + `resolve_groq_config()` to integrate Groq/OpenAI-compatible endpoints, and surfaced `--remote-llm` CLI options in `gen_docs.py`.
- Added automated planner tests in `tests/test_autoplan.py` and CLI coverage in `tests/test_gen_docs_cli.py`.

### Changed
- Incremental indexing now reuses the adapter registry so mixed-language
  repositories are handled consistently in both fresh and incremental runs.
- Extended `store.py` schema and index reset logic to manage docs-specific tables and FAISS index.
- `cli.py` updated to optionally index docs after code indexing and to merge doc results at query time.
- `README.md` and `ROADMAP.md` updated with the documentation generator workflow, LLM configuration, and dependency-group guidance.
- `pyproject.toml` now credits OpenSource Syndicate as the author and introduces a `languages` dependency group for uv-based installs.


## [0.2.0] - 2025-10-06

### Added
- Hybrid search functionality combining dense vector search and keyword search using Elasticsearch
- Reciprocal Rank Fusion (RRF) to combine search results from multiple sources
- Semantic-aware chunking using CAST-like algorithm for better context preservation
- Incremental indexing to only re-process changed files using file hashing
- `--hybrid` flag to enable hybrid search in query command
- `--chunking` option to select between symbol-based and semantic chunking
- `--similarity-threshold` parameter for controlling semantic chunking sensitivity
- `--incremental` flag for incremental indexing mode
- Comprehensive test suite for all new features

### Changed
- Enhanced CLI to support new hybrid search and chunking options
- Updated indexing process to support both fresh and incremental indexing
- Improved error handling for Elasticsearch connection failures
- Refactored chunking module to support both traditional and semantic-aware chunking
- Added proper hash-based comparison for Symbol class

### Fixed
- Symbol hashability issue that was causing errors in dictionary lookups
- Various mocking issues in tests to properly isolate functionality

### Dependencies
- Added elasticsearch dependency for keyword search functionality
- Added sentence-transformers for semantic similarity calculations

## [0.1.0] - 2025-10-05

### Added
- Initial release of semindex
- AST-based parsing for Python code
- Vector search using FAISS
- CLI interface for indexing and querying
- Basic chunking by function/class boundaries
- SQLite for metadata storage

[Unreleased]: https://github.com/OpenSource-Syndicate/semindex/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/OpenSource-Syndicate/semindex/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/OpenSource-Syndicate/semindex/releases/tag/v0.1.0