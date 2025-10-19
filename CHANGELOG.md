# Changelog
## [0.3.0] - 2025-10-20

### Added
- Added a pluggable language adapter registry in `semindex.languages` that
  powers automatic discovery of supported file types and supports runtime
  registration of custom adapters.
- Introduced an optional Tree-sitter powered JavaScript adapter that is
  registered when `tree_sitter_languages` is available, expanding
  multi-language indexing support.
- External library documentation indexing (PyPI + local site-packages). Docs are parsed (HTML/Markdown), normalized, embedded, and stored in dedicated tables (`doc_packages`, `doc_pages`, `doc_vectors`) and a separate FAISS index `docs.faiss`. CLI: `--include-docs` for `index` and `query`, with `--docs-weight` to control ranking merge.
- New public Python wrappers: `Indexer` (`semindex.indexer.Indexer`) and `Searcher` (`semindex.search.Searcher`) for programmatic indexing and querying, including hybrid search and optional docs merging.

### Changed
- Incremental indexing now reuses the adapter registry so mixed-language
  repositories are handled consistently in both fresh and incremental runs.
- Extended `store.py` schema and index reset logic to manage docs-specific tables and FAISS index.
- `cli.py` updated to optionally index docs after code indexing and to merge doc results at query time.
- `README.md` and `ROADMAP.md` updated to document docs indexing/retrieval and the new programmatic API.


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