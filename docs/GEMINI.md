# semindex

## Project Overview

`semindex` is a command-line tool for semantic code indexing and searching. It's designed to help developers understand and navigate large codebases using natural language queries. The tool is written in Python and leverages several open-source libraries for its core functionality.

The project is structured as a Python package with a command-line interface. The main entry point is `src/semindex/cli.py`, which uses the `argparse` library to define the available commands and their arguments.

### Core Technologies

*   **Python:** The primary programming language.
*   **HuggingFace Transformers:** Used for generating code embeddings.
*   **FAISS:** Used for efficient similarity search on the code embeddings.
*   **Elasticsearch:** Used for keyword search.
*   **Tree-sitter:** Used for parsing multiple programming languages.
*   **SQLite:** Used for storing metadata about the indexed code.

### Architecture

The project follows a modular architecture, with different components responsible for specific tasks:

*   **`indexer.py`:** Handles the indexing process, including parsing files, generating embeddings, and storing them in the index.
*   **`search.py`:** Implements the search functionality, including vector search, keyword search, and hybrid search.
*   **`ai.py`:** Provides AI-powered features like code explanation, generation, and bug finding.
*   **`project_planner.py`:** Implements the AI-powered project planning and execution features.
*   **`config.py`:** Manages the configuration of the tool.
*   **`languages` directory:** Contains the language adapters for parsing different programming languages.

## Building and Running

### Installation

The project can be installed using `pip` and a virtual environment. The following commands will install the tool and its dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -e .
```

### Running

The tool is run from the command line using the `semindex` command. The following are some of the key commands:

*   **`semindex index <path-to-repo>`:** Indexes a repository.
*   **`semindex query "<your-query>"`:** Searches the indexed repository.
*   **`semindex ai chat`:** Starts an interactive AI chat session about the codebase.
*   **`semindex ai-plan create "<project-description>"`:** Creates a project plan from a description.

### Testing

The project uses `pytest` for testing. The tests are located in the `tests` directory and can be run with the following command:

```powershell
pytest
```

## Development Conventions

The project follows standard Python development conventions. The code is formatted using an autoformatter (likely Black or a similar tool) and linted with `ruff`. The project uses `pyproject.toml` to manage dependencies and project settings.

The project has a clear and consistent coding style, with a focus on readability and maintainability. The code is well-documented with comments and docstrings.
