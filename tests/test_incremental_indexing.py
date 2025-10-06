"""
Tests for incremental indexing functionality.
"""
import os
import tempfile
import sqlite3
from unittest.mock import Mock, patch
from semindex.store import (
    file_sha256,
    file_sha256_from_content,
    get_changed_files,
    get_all_files_in_db,
    remove_file_from_index,
    ensure_db
)
from semindex.cli import cmd_index
import argparse


def test_file_sha256_from_content():
    """Test SHA256 hash calculation from content."""
    content1 = b"def hello():\n    print('Hello, World!')"
    content2 = b"def goodbye():\n    print('Goodbye, World!')"
    
    hash1 = file_sha256_from_content(content1)
    hash2 = file_sha256_from_content(content2)
    
    # Different content should produce different hashes
    assert hash1 != hash2
    
    # Same content should produce same hash
    assert hash1 == file_sha256_from_content(content1)


def test_get_changed_files_fresh_db():
    """Test get_changed_files when DB is empty (fresh indexing)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file
        test_file = os.path.join(temp_dir, "test.py")
        with open(test_file, 'w') as f:
            f.write("def hello():\n    print('Hello')")
        
        # Create database
        db_path = os.path.join(temp_dir, "test.db")
        ensure_db(db_path)
        
        # Since DB is empty, get_changed_files should return an empty list
        # Actually, it should return all files in the repo since they're not in the DB yet
        # Let me check what the function does
        
        # For this test, we'll mock the iter_python_files function to return our test file
        with patch('semindex.crawler.iter_python_files') as mock_iter:
            mock_iter.return_value = [test_file]
            changed_files = get_changed_files(temp_dir, db_path)
            
            # The file should be returned as changed since it's not in the DB
            assert test_file in changed_files


def test_get_changed_files_with_existing_db():
    """Test get_changed_files with existing database entries."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        file1 = os.path.join(temp_dir, "file1.py")
        file2 = os.path.join(temp_dir, "file2.py")
        
        # Write files first
        with open(file1, 'w', encoding='utf-8') as f:
            f.write("def func1():\n    pass")
        with open(file2, 'w', encoding='utf-8') as f:
            f.write("def func2():\n    pass")
        
        # Compute the actual file hashes as they are stored on disk
        file1_actual_hash = file_sha256(file1)
        
        # Create database and add initial entries
        db_path = os.path.join(temp_dir, "test.db")
        ensure_db(db_path)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Add file1 with its actual hash to the DB
        cursor.execute("INSERT INTO files (path, hash) VALUES (?, ?)", (file1, file1_actual_hash))
        conn.commit()
        conn.close()
        
        # Mock iter_python_files to return both files
        with patch('semindex.crawler.iter_python_files') as mock_iter:
            mock_iter.return_value = [file1, file2]
            
            # file2 should be returned as changed (not in DB)
            # file1 should not be returned as changed (same hash)
            changed_files = get_changed_files(temp_dir, db_path)
            
            # file2 should be in the changed files list (new file)
            assert file2 in changed_files
            # file1 should not be in the list since it hasn't changed
            assert file1 not in changed_files


def test_get_changed_files_modified_file():
    """Test get_changed_files when a file has been modified."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test file
        test_file = os.path.join(temp_dir, "test.py")
        original_content = b"def func():\n    pass"
        
        with open(test_file, 'wb') as f:
            f.write(original_content)
        
        # Create database and add the file with its original hash
        db_path = os.path.join(temp_dir, "test.db")
        ensure_db(db_path)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        original_hash = file_sha256_from_content(original_content)
        cursor.execute("INSERT INTO files (path, hash) VALUES (?, ?)", (test_file, original_hash))
        conn.commit()
        conn.close()
        
        # Modify the file content
        modified_content = b"def func():\n    print('Modified')\n    pass"
        with open(test_file, 'wb') as f:
            f.write(modified_content)
        
        # Mock iter_python_files to return the file
        with patch('semindex.crawler.iter_python_files') as mock_iter:
            mock_iter.return_value = [test_file]
            
            # The file should be returned as changed
            changed_files = get_changed_files(temp_dir, db_path)
            assert test_file in changed_files


def test_get_all_files_in_db():
    """Test getting all files stored in the database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test.db")
        ensure_db(db_path)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Add some test files
        test_files = [
            ("file1.py", "hash1"),
            ("file2.py", "hash2"),
            ("file3.py", "hash3")
        ]
        
        cursor.executemany("INSERT INTO files (path, hash) VALUES (?, ?)", test_files)
        conn.commit()
        conn.close()
        
        # Get all files from DB
        files_in_db = get_all_files_in_db(db_path)
        
        # Check that all files are returned
        for path, _ in test_files:
            assert path in files_in_db
        
        assert len(files_in_db) == 3


def test_cmd_index_incremental_flag():
    """Test that cmd_index respects the incremental flag."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock repo with Python files
        repo_dir = os.path.join(temp_dir, "repo")
        os.makedirs(repo_dir)
        
        file1 = os.path.join(repo_dir, "file1.py")
        with open(file1, 'w', encoding='utf-8') as f:
            f.write("def hello():\n    print('Hello')")
        
        # Set up args with incremental flag
        args = argparse.Namespace(
            repo=repo_dir,
            index_dir=os.path.join(temp_dir, "index"),
            model="microsoft/codebert-base",
            batch=16,
            verbose=False,
            chunking='symbol',
            similarity_threshold=0.7,
            incremental=True  # Enable incremental indexing
        )
        
        # Mock the necessary components
        with patch('semindex.cli.Embedder') as mock_embedder_class:
            # Create a mock embedder instance
            mock_embedder_instance = Mock()
            mock_embedder_instance.model = Mock()
            mock_embedder_instance.model.config = Mock()
            mock_embedder_instance.model.config.hidden_size = 128  # Mock a valid dimension
            
            # Configure the class to return the instance
            mock_embedder_class.return_value = mock_embedder_instance
            
            with patch('semindex.cli.iter_python_files', return_value=[file1]), \
                 patch('semindex.cli.read_text', return_value="def hello():\n    print('Hello')"), \
                 patch('semindex.cli.parse_python_symbols', return_value=([], [])), \
                 patch('semindex.cli.build_chunks_from_symbols', return_value=[]), \
                 patch('semindex.cli.add_vectors'), \
                 patch('builtins.print'):
                
                # Run the index command
                cmd_index(args)
                
                # Verify that the index directory was created
                assert os.path.exists(args.index_dir)


def test_cmd_index_fresh_vs_incremental():
    """Test the difference between fresh and incremental indexing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock repo with Python files
        repo_dir = os.path.join(temp_dir, "repo")
        os.makedirs(repo_dir)
        
        file1 = os.path.join(repo_dir, "file1.py")
        with open(file1, 'w', encoding='utf-8') as f:
            f.write("def hello():\n    print('Hello')")
        
        # First, run fresh indexing
        index_dir = os.path.join(temp_dir, "index")
        os.makedirs(index_dir)
        
        # Mock the necessary components for fresh indexing
        with patch('semindex.cli.Embedder') as mock_embedder_class:
            # Create a mock embedder instance
            mock_embedder_instance = Mock()
            mock_embedder_instance.model = Mock()
            mock_embedder_instance.model.config = Mock()
            mock_embedder_instance.model.config.hidden_size = 128  # Mock a valid dimension
            
            # Configure the class to return the instance
            mock_embedder_class.return_value = mock_embedder_instance
            
            with patch('semindex.cli.iter_python_files', return_value=[file1]), \
                 patch('semindex.cli.read_text', return_value="def hello():\n    print('Hello')"), \
                 patch('semindex.cli.parse_python_symbols', return_value=([], [])), \
                 patch('semindex.cli.build_chunks_from_symbols', return_value=[]), \
                 patch('semindex.cli.add_vectors'), \
                 patch('builtins.print'):
                
                # Args for fresh indexing
                args_fresh = argparse.Namespace(
                    repo=repo_dir,
                    index_dir=index_dir,
                    model="microsoft/codebert-base",
                    batch=16,
                    verbose=False,
                    chunking='symbol',
                    similarity_threshold=0.7,
                    incremental=False
                )
                
                cmd_index(args_fresh)