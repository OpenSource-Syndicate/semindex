# Testing Dynamic Documentation Generation

This guide walks through testing the new unified template system and index-driven planning.

## Prerequisites

- `uv` installed and configured
- Repository cloned and dependencies installed

## Step-by-Step Workflow

### 1. Clean Up Old Index and Wiki

```powershell
Remove-Item -Path ".semindex" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "wiki" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "Cleaned up old index and wiki"
```

### 2. Fresh Index

Index the source code to populate the database with symbols:

```powershell
uv run semindex index src --index-dir .semindex --verbose
```

Expected output:
```
Performing fresh indexing of 27 files
[WARN] Could not connect to Elasticsearch: ... (this is OK)
Indexed 218 chunks from repository ...
```

### 3. Test Discovery Functions (Optional)

Verify that discovery functions find content:

```powershell
uv run python tests/test_discovery.py
```

Expected output:
```
✅ Key Modules (5 found):
   - src/semindex/...
   - src/semindex/...
   ...

✅ Key Classes (10+ found):
   - ClassName (path/to/file.py)
   ...

✅ Key Functions (10+ found):
   - function_name (path/to/file.py)
   ...

✅ Detected Patterns (2+ found):
   - testing
   - configuration
   ...
```

### 4. Generate Wiki with Auto-Planning

Generate documentation with automatic section discovery:

```powershell
# Using local LLM (requires llama-cpp-python)
uv run python scripts/gen_docs.py --repo-root . --auto-plan --force

# Or using remote Groq/OpenAI-compatible LLM
$env:SEMINDEX_REMOTE_API_KEY = "your-api-key"
uv run python scripts/gen_docs.py --repo-root . --remote-llm --auto-plan --force

# Or without LLM (fallback content)
uv run python scripts/gen_docs.py --repo-root . --no-llm --auto-plan --force
```

### 5. Expected Output

With the fixed discovery functions, you should see:

```
[info] Auto planner selected 9 section(s)
[info] Using remote Groq-compatible LLM
[OK] Generated Overview.md
[OK] Generated Architecture.md
[OK] Generated Adapters.md
[OK] Generated Indexing.md
[OK] Generated Languages.md
[OK] Generated KeyModules.md
[OK] Generated KeyClasses.md
[OK] Generated KeyFunctions.md
[OK] Generated Patterns.md
```

### 6. Verify Generated Wiki

Check the generated documentation:

```powershell
# List generated files
Get-ChildItem wiki/*.md

# View a specific section
Get-Content wiki/KeyModules.md
Get-Content wiki/KeyClasses.md
Get-Content wiki/KeyFunctions.md
Get-Content wiki/Patterns.md
```

## What's Dynamic?

The following sections are now **dynamically discovered** from the indexed codebase:

1. **Key Modules** (`KeyModules.md`)
   - Distinct file paths from indexed symbols
   - Shows which files contain the most code

2. **Key Classes** (`KeyClasses.md`)
   - All class symbols found during indexing
   - Ordered by name for easy reference

3. **Key Functions** (`KeyFunctions.md`)
   - All function symbols found during indexing
   - Ordered by name for easy reference

4. **Patterns** (`Patterns.md`)
   - Detected architectural patterns:
     - Testing (presence of test files)
     - Configuration (config files)
     - API (handlers, routes, endpoints)
     - Data layer (queries, models, schemas)

## Troubleshooting

### No index-discovered sections generated

**Problem**: Only 5 sections generated (Overview, Architecture, Adapters, Indexing, Languages)

**Solution**: 
1. Verify index was created: `Test-Path .semindex/semindex.db`
2. Run tests/test_discovery.py to check if content is found
3. Check that gen_docs.py is passing index_dir to generate_plan()

### Empty discovery results

**Problem**: Discovery functions return empty lists

**Solution**:
1. Verify indexing completed: `uv run semindex index src --index-dir .semindex --verbose`
2. Check database has symbols: `sqlite3 .semindex/semindex.db "SELECT COUNT(*) FROM symbols;"`
3. Run tests/test_discovery.py with verbose output

### Template errors

**Problem**: FileNotFoundError for template files

**Solution**: This is now fixed! The system uses a unified template as fallback.
- No template files required
- Custom templates still supported if placed in `wiki/templates/`

## Key Features

✅ **No Template Files Required**: Works out of the box
✅ **Dynamic Content**: Discovers actual codebase structure
✅ **Flexible**: Supports custom templates while providing defaults
✅ **Backward Compatible**: Old template system still works
✅ **Intelligent Planning**: Combines rule-based and index-driven sections
✅ **Extensible**: Easy to add new discovery functions

## Next Steps

- Customize the unified template in `_get_unified_template()` in `scripts/gen_docs.py`
- Add custom discovery functions to `src/semindex/docs/autoplan.py`
- Create custom templates in `wiki/templates/` for specific sections
- Integrate with CI/CD to auto-generate docs on commits
