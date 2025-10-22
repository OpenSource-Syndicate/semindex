# semindex Project Context

## Project Overview

semindex is a local semantic codebase indexer specifically designed for Python code, using Abstract Syntax Tree (AST) parsing combined with embeddings for semantic search capabilities. The project uses Python's built-in `ast` module for parsing, HuggingFace Transformers for generating embeddings, FAISS for vector search, and SQLite for metadata storage.

## Architecture & Components

The project is organized into several key modules:

- **cli.py**: Main command-line interface that coordinates indexing and querying operations
- **crawler.py**: Discovers Python files in a repository while respecting ignore patterns
- **ast_py.py**: Parses Python code using the AST module, extracting symbols (functions, classes, methods) and their properties
- **chunker.py**: Splits code into meaningful chunks based on function/method/class boundaries
- **embed.py**: Handles text embedding using HuggingFace Transformers models
- **store.py**: Manages storage using both FAISS for vector embeddings and SQLite for metadata
- **search.py**: Provides search capabilities using the stored embeddings

## Functionality

The system provides two main commands:
- `semindex index <repo_path>`: Indexes a Python repository by parsing all Python files and creating embeddings
- `semindex query "<query>"`: Performs semantic search against the indexed codebase

The indexing process:
1. Discovers Python files recursively while excluding common directories (.git, __pycache__, etc.)
2. Parses each file using Python's AST module
3. Extracts symbols (functions, methods, classes) with metadata (signature, docstring, imports, etc.)
4. Creates chunks of code based on symbol boundaries
5. Generates embeddings for each chunk
6. Stores both embeddings (in FAISS) and metadata (in SQLite)

## Dependencies & Environment

- Python 3.12.11 (specific version requirement)
- Core dependencies: faiss-cpu, transformers, torch, numpy
- Uses setuptools for packaging
- Optional dev dependencies include pytest and ruff
- Has a console script entry point for the `semindex` command

## Development & Testing

- Tests are located in the `tests/` directory and use pytest
- Testing includes unit tests for AST parsing, CLI functionality, and crawler functionality
- The project supports both pip and uv for dependency management

## Current State & Roadmap

Based on the README, the project has completed Phase 1 of its roadmap (Weeks 1-8), which includes:
- Core architecture design
- Language parsing integration
- Basic chunking and indexing
- Initial retrieval functionality

Remaining work includes keyword/graph indexing, advanced chunking, hybrid search, incremental updates, documentation generation, and UI development.

## Key Files

- `pyproject.toml`: Project configuration and dependencies
- `README.md`: Installation, usage, and roadmap documentation
- `src/semindex/cli.py`: Main application logic
- `src/semindex/ast_py.py`: Core AST parsing and symbol extraction
- `src/semindex/store.py`: Data persistence layer
- `tests/`: Test suite