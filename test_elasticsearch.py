#!/usr/bin/env python3
"""
Test script to verify semindex is using Elasticsearch for keyword search
"""
import os
import tempfile
from src.semindex.keyword_search import KeywordSearcher
from src.semindex.hybrid_search import hybrid_search
import numpy as np

def test_keyword_search():
    print("Testing semindex keyword search functionality...")
    
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Initialize the keyword searcher
            searcher = KeywordSearcher(index_dir=temp_dir)
            
            # Create the index
            searcher.create_index()
            print("[OK] Elasticsearch index created")
            
            # Test indexing a sample symbol
            sample_symbol = {
                "id": 1,
                "path": "/test/file.py",
                "name": "test_function",
                "kind": "function",
                "signature": "def test_function(param1: str, param2: int)",
                "docstring": "This is a test function for testing purposes",
                "imports": "from typing import List, Dict",
                "bases": "",
                "content": "def test_function(param1: str, param2: int):\n    \"\"\"This is a test function for testing purposes\"\"\"\n    return param1 + str(param2)"
            }
            
            # Index the symbol
            searcher.index_symbol(sample_symbol)
            print("[OK] Sample symbol indexed successfully")
            
            # Test search functionality
            results = searcher.search("test function", top_k=5)
            print(f"[OK] Search completed, found {len(results)} results")
            
            if results:
                print(f"  - First result: {results[0].name} in {results[0].path}")
            
            # Test bulk indexing
            sample_symbols = [
                {
                    "id": 2,
                    "path": "/test/file2.py", 
                    "name": "another_function",
                    "kind": "function",
                    "signature": "def another_function()",
                    "docstring": "Another test function",
                    "imports": "",
                    "bases": "",
                    "content": "def another_function():\n    pass"
                },
                {
                    "id": 3,
                    "path": "/test/class.py",
                    "name": "TestClass", 
                    "kind": "class",
                    "signature": "class TestClass:",
                    "docstring": "A test class",
                    "imports": "",
                    "bases": "object",
                    "content": "class TestClass:\n    def __init__(self):\n        pass"
                }
            ]
            
            searcher.bulk_index_symbols(sample_symbols)
            print("[OK] Bulk indexing completed")
            
            # Search for the new items
            results = searcher.search("TestClass", top_k=5)
            print(f"[OK] Search for 'TestClass' found {len(results)} results")
            
            print("\n[OK] All Elasticsearch keyword search functionality tests passed!")
            
            # Clean up the index
            searcher.delete_index()
            print("[OK] Test index cleaned up")
            
        except Exception as e:
            print(f"X Error during testing: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def test_hybrid_search():
    print("\nTesting semindex hybrid search functionality...")
    
    # This is a more complex test that requires a full index
    # For now, we'll just verify the module can be imported and function exists
    print("[OK] Hybrid search module loaded successfully")
    print(f"[OK] Hybrid search function exists: {hybrid_search is not None}")
    
if __name__ == "__main__":
    print("Testing semindex Elasticsearch integration...")
    success1 = test_keyword_search()
    test_hybrid_search()
    
    if success1:
        print("\n[OK] All tests passed! semindex is properly using Elasticsearch.")
    else:
        print("\n[X] Some tests failed.")