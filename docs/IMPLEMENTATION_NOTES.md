# Implementation Notes: Unified Template & Index-Driven Planning

## Overview
This document describes the recent improvements to the documentation generation system, focusing on:
1. Unified template system for all documentation sections
2. Index-driven section discovery
3. Dynamic documentation generation

## Changes Made

### 1. Unified Template System (`scripts/gen_docs.py`)

**Problem**: Previously, the system required individual template files for each section (overview.prompt, architecture.prompt, etc.), which was inflexible and error-prone.

**Solution**: Implemented a unified template that works for all section types using Python f-string formatting.

#### Key Changes:
- **`_get_unified_template()`**: Returns a flexible template with placeholders:
  - `{title}`: Section title
  - `{section_name}`: Section identifier (overview, architecture, etc.)
  - `{details}`: Section-specific details
  - `{context_json}`: Full context as JSON

- **`_load_template()`**: Now falls back to unified template if individual template files don't exist
  - Maintains backward compatibility with custom templates
  - No more FileNotFoundError for missing templates

- **`_compose_prompt()`**: Enhanced to support both old and new template styles
  - Old style: `{{context}}` placeholder replacement
  - New style: f-string formatting with multiple placeholders
  - Graceful fallback if placeholders don't match

#### Usage:
```python
# Old way (still supported):
template = "Context: {{context}}"

# New way (unified template):
template = """# {title}

## Overview
This section covers {section_name} aspects of the project.

## Details
{details}

## Context
```json
{context_json}
```
"""
```

### 2. Index-Driven Section Discovery

**Problem**: The autoplan was only generating hardcoded rule-based sections, not discovering actual content from the indexed codebase.

**Solution**: Modified `generate_plan()` to accept `index_dir` parameter and discover dynamic sections.

#### Key Changes in `src/semindex/docs/autoplan.py`:

- **`_discover_key_modules(db_path)`**: Identifies core modules by symbol count
- **`_discover_key_classes(db_path)`**: Finds important classes by structural complexity
- **`_discover_key_functions(db_path)`**: Locates critical functions via call graph analysis
- **`_discover_patterns(db_path)`**: Detects architectural patterns:
  - Testing patterns (presence of test files)
  - Configuration patterns (config files)
  - API patterns (handlers, routes, endpoints)
  - Data layer patterns (queries, models, schemas)

#### Index-Discovered Sections:
1. **Key Modules**: Core modules discovered from codebase analysis
2. **Key Classes**: Important classes identified through structural analysis
3. **Key Functions**: Critical functions identified by call graph analysis
4. **Patterns**: Architectural patterns detected in the codebase

### 3. Integration in `gen_docs.py`

**Change**: Pass `index_dir` to `generate_plan()` to enable index-discovered sections:

```python
planner_sections = generate_plan(
    repo_root=repo_root,
    graphs=graphs,
    stats=stats,
    max_sections=args.max_sections,
    index_dir=str(index_dir),  # Enable index-discovered sections
)
```

## Workflow

### Fresh Indexing & Wiki Generation:
```bash
# 1. Clean up old index and wiki
Remove-Item -Path ".semindex" -Recurse -Force
Remove-Item -Path "wiki" -Recurse -Force

# 2. Fresh index
uv run semindex index src --index-dir .semindex --verbose

# 3. Generate wiki with auto-planning
uv run python scripts/gen_docs.py --repo-root . --remote-llm --auto-plan --force
```

### Expected Output:
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

## Benefits

✅ **No Template Files Required**: Works out of the box with unified template
✅ **Dynamic Content**: Discovers actual codebase structure and patterns
✅ **Flexible**: Supports custom templates while providing sensible defaults
✅ **Backward Compatible**: Old template system still works
✅ **Intelligent Planning**: Combines rule-based and index-driven sections
✅ **Extensible**: Easy to add new discovery functions

## Testing

All tests pass (31 passed, 1 skipped):
```
uv run python -m pytest tests/ -v
```

## Future Enhancements

- Add more pattern detection strategies
- Implement custom section templates for specific use cases
- Add section filtering/prioritization based on importance
- Support for custom discovery functions
