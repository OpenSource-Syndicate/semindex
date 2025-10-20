#!/usr/bin/env python3
"""Quick test to verify index discovery functions work."""

import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from semindex.docs.autoplan import (
    _discover_key_modules,
    _discover_key_classes,
    _discover_key_functions,
    _discover_patterns,
)

def test_discovery():
    """Test discovery functions against .semindex database."""
    db_path = Path(".semindex/semindex.db")
    
    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        print("Please run: uv run semindex index src --index-dir .semindex")
        return False
    
    print("Testing discovery functions...\n")
    
    # Test modules
    modules = _discover_key_modules(str(db_path), limit=5)
    print(f"✅ Key Modules ({len(modules)} found):")
    for mod in modules:
        print(f"   - {mod}")
    
    # Test classes
    classes = _discover_key_classes(str(db_path), limit=10)
    print(f"\n✅ Key Classes ({len(classes)} found):")
    for name, path in classes[:5]:
        print(f"   - {name} ({path})")
    if len(classes) > 5:
        print(f"   ... and {len(classes) - 5} more")
    
    # Test functions
    functions = _discover_key_functions(str(db_path), limit=10)
    print(f"\n✅ Key Functions ({len(functions)} found):")
    for name, path in functions[:5]:
        print(f"   - {name} ({path})")
    if len(functions) > 5:
        print(f"   ... and {len(functions) - 5} more")
    
    # Test patterns
    patterns = _discover_patterns(str(db_path))
    print(f"\n✅ Detected Patterns ({len(patterns)} found):")
    for pattern in patterns:
        print(f"   - {pattern}")
    
    success = bool(modules or classes or functions or patterns)
    print(f"\n{'✅ Discovery working!' if success else '❌ No content discovered'}")
    return success

if __name__ == "__main__":
    success = test_discovery()
    sys.exit(0 if success else 1)
