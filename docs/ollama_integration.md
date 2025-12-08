# Ollama Integration for semindex

This document explains how to use Ollama with semindex to leverage GPU-accelerated language models for enhanced code understanding and generation.

## Prerequisites

1. Install and run Ollama on your system:
   - Visit [https://ollama.ai](https://ollama.ai) to download and install Ollama
   - Make sure Ollama is running (typically accessible at `http://localhost:11434`)

2. Pull a model you'd like to use:
   ```bash
   ollama pull llama3
   # or for code-specific models:
   ollama pull codellama:7b
   ollama pull deepseek-coder:6.7b
   ```

## Using Ollama with semindex

### Basic Query with Ollama

To perform a query and get an AI-generated response using Ollama:

```bash
semindex query "Explain how authentication works in this codebase" --ollama
```

### Specify a Model

To use a specific Ollama model:

```bash
semindex query "How do I add a new user?" --ollama --ollama-model codellama:7b
```

### Advanced Query with Context

You can combine Ollama with other features:

```bash
semindex query "Generate a test for the user authentication function" \
    --ollama \
    --ollama-model codellama:7b \
    --top-k 5 \
    --hybrid \
    --include-docs
```

### Environment Variables

You can set these environment variables to customize Ollama behavior:

- `SEMINDEX_OLLAMA_MODEL`: Default model to use (default: llama3)
- `SEMINDEX_OLLAMA_BASE_URL`: Base URL for Ollama API (default: http://localhost:11434)

## Example Use Cases

### 1. Code Explanation
```bash
semindex query "Explain the user management system" --ollama
```

### 2. Finding Implementation
```bash
semindex query "Show me how password hashing is implemented" --ollama
```

### 3. Generating Documentation
```bash
semindex query "Generate documentation for the API endpoints" --ollama
```

### 4. Code Suggestions
```bash
semindex query "How can I optimize the database queries in this module?" --ollama
```

## Comparison with Standard Query

Without Ollama (just retrieves relevant code snippets):
```bash
semindex query "How does user authentication work?"
```

With Ollama (retrieves code + generates AI explanation):
```bash
semindex query "How does user authentication work?" --ollama
```

The Ollama-enhanced query will not only show relevant code snippets but also provide an AI-generated explanation of how the authentication system works, making it more similar to tools like Claude Code or GitHub Copilot.

## Performance Tips

1. Use code-specific models like `codellama:7b` or `deepseek-coder:6.7b` for better code understanding
2. Adjust `--top-k` to control how many code snippets are provided as context (default is 10)
3. Use `--max-tokens` to control the length of the Ollama response
4. Consider using `--hybrid` and `--include-docs` for more comprehensive context