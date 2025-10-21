# Changelog

## [0.4.0] - 2025-10-21

### Added
- Implemented `cmd_query()` CLI function for semantic search with support for hybrid search and documentation merging
- Enhanced autoplan with index-driven section discovery:
- Added enhanced contextual code generation system with multi-modal context awareness
  - New module `context_enhancer.py` with `DocumentationExtractor`, `TypeExtractor`, and `StructureAnalyzer`
  - Enhanced `ContextAggregator` with multi-modal context gathering (documentation, comments, types, structure)
  - Added call graph integration for cross-file dependency awareness
  - Implemented AST-based fine-grained context extraction instead of simple line-based context
  - Added `GeneratedCode` class to represent generated code with metadata
  - New `CodeValidator` for comprehensive code validation
  - New `ExecutionGuidedGenerator` for refinement based on validation feedback
- Added intent recognition and task decomposition capabilities
  - New module `intent_analyzer.py` with `IntentClassifier` that categorizes user requests
  - Added `IntentType` enum with categories: IMPLEMENTATION, REFACTORING, DEBUGGING, DOCUMENTATION, TESTING, ANALYSIS
  - Implemented `TaskDecomposer` to break complex requests into subtasks
  - Enhanced generation prompts based on detected intent
- Added advanced pattern recognition and template-based generation
  - New module `pattern_analyzer.py` with `PatternExtractor`, `PatternMatcher`, and `TemplateGenerator`
  - Implemented `TemplateRegistry` to store and manage code templates from the user's codebase
  - Added pattern-based suggestions to enhance code generation
- Added interactive refinement capabilities
  - New `InteractiveCodeRefiner` class for iterative improvement through user feedback
  - Added conversation memory to maintain context across interactions
  - Implemented feedback incorporation mechanism
- Added real-time context update system
  - New module `context_watcher.py` with `ContextFileWatcher` using watchdog library
  - Enhanced `ContextCache` with file dependency tracking and invalidation
  - Added debouncing mechanism to avoid excessive updates during rapid typing
- Added new CLI command `ai generate-context` that leverages enhanced contextual generation
  - Accepts file path, line number, and natural language request
  - Uses multi-modal context and intent recognition for better generation
  - Provides enhanced contextual awareness compared to basic generation
- Enhanced existing `ContextAwareCodeGenerator` with intent recognition, pattern matching, and execution feedback
- Added new dependencies: `watchdog` for file watching capabilities
  - `_discover_key_modules()`: Identifies core modules by symbol count
  - `_discover_key_classes()`: Finds important classes by structural complexity
  - `_discover_key_functions()`: Locates critical functions via call graph analysis
  - `_discover_patterns()`: Detects architectural patterns (testing, configuration, API, data layer)
- Added `PlanningRule` dataclass for declarative section planning rules
- Refactored `generate_plan()` to support both rule-based and index-discovered sections
- Added comprehensive test suite for language adapters, hybrid search, and CLI functionality
- Implemented graph generation capabilities (module, adapter, pipeline graphs and code statistics)
- Added call graph analysis with `--callers` and `--callees` options to the `graph` command
- Introduced AI-powered commands for code understanding and generation (ai subcommand with chat, explain, suggest, generate, docs, bugs, refactor, and tests)
- Added new test file `test_ai_commands.py` for testing AI command functionality
- Implemented AI-powered project planning system with `ai-plan` command featuring create, execute, and manage subcommands
  - Added `ProjectPlanner` class for generating project plans from description or existing codebase
  - Added `ProjectPhase`, `ProjectTask`, and `ProjectComponent` dataclasses to represent project structure
  - Added `TaskManager` class for tracking task status and progress
  - Added `DevelopmentWorkflow` class for executing project plans
  - Added `TestingFramework` class for generating tests for components
  - Added `IntegrationManager` class for creating integration layers after implementation
  - Added `AIImplementationAssistant` with explanation capabilities combining internal and external knowledge
- Implemented Perplexica-powered search capabilities with `perplexica` command featuring search and explain subcommands
  - Added `PerplexicaSearchAdapter` class for integrating with Perplexica's search API
  - Added support for various focus modes (webSearch, docSearch, academicSearch, librarySearch, youtubeSearch, redditSearch, hybridSearch)
  - Added configuration system with TOML-based config file for API endpoints and model settings
  - Implemented hybrid search combining local codebase and web results
  - Added fallback mechanisms when external APIs are unavailable
- Introduced configuration system with `config.py` module
  - Added `Config` class for loading and accessing settings from config.toml
  - Added support for default configuration values and config file discovery
  - Added methods for getting and setting configuration values via dot-separated paths
  - Added specific getters for different services (Perplexica, OpenAI, Groq, Ollama)
- Added new dependencies: `toml` for configuration file parsing
- Added new modules: `project_planner.py`, `task_manager.py`, `development_workflow.py`, `testing_framework.py`, `integration_manager.py`, `perplexica_adapter.py`, `config.py`, `ai_implementation_assistant.py`, `focus_modes.py`, `component_generator.py`
- Added new test modules: `test_config.py`, `test_perplexica_integration.py`, and test files in the `tests` directory (`test_ai_planning.py` and `e2e_test.py`) for end-to-end testing
- Added new command-line interface options for the new AI planning and Perplexica commands

### Changed
- Autoplan now generates documentation sections dynamically from indexed codebase rather than hardcoded templates
- `generate_plan()` accepts optional `index_dir` parameter to enable index-based discovery
- Improved code organization with separation of concerns in planning logic
- README.md updated to reflect index-driven documentation planning and new AI commands
- Enhanced test coverage with additional test cases for RRF edge cases and CLI functionality
- CLI interface expanded to support new AI planning and Perplexica commands
- `retrieve_context()` function in `rag.py` enhanced to support different focus modes and configuration-based settings
- Updated tree-sitter adapter registration to conditionally register adapters only when optional dependencies are available

### Fixed
- Fixed missing `cmd_query` function that was causing NameError in CLI
- Added missing imports (`json`, `sqlite3`) to cli.py
- Fixed unpacking error in query result formatting (6 fields instead of 7)
- Fixed issue where tree-sitter adapters would cause errors when optional dependencies weren't available
- Enhanced error handling in API calls to prevent crashes when external services are unavailable

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

[0.4.0]: https://github.com/OpenSource-Syndicate/semindex/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/OpenSource-Syndicate/semindex/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/OpenSource-Syndicate/semindex/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/OpenSource-Syndicate/semindex/releases/tag/v0.1.0