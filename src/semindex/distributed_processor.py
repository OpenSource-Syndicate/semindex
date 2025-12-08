"""
Distributed Processing Module for semindex
Enables processing of very large codebases across multiple machines or processes
"""
import os
import json
import hashlib
import multiprocessing as mp
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import concurrent.futures
from dataclasses import dataclass
import queue
import sqlite3

from .indexer import Indexer
from .store import DB_NAME, FAISS_INDEX, ensure_db, reset_index, add_vectors, add_calls, db_conn
from .embed import Embedder
from .model import ChunkingConfig, Symbol, Chunk
from .languages import get_adapter, ensure_default_adapters


@dataclass
class DistributedTask:
    """Represents a task to be processed in a distributed system"""
    task_id: str
    file_path: str
    language: str
    content_hash: str
    priority: int = 1


@dataclass
class DistributedResult:
    """Represents the result of a distributed task"""
    task_id: str
    symbols: List[Symbol]
    chunks: List[Chunk]
    calls: List[Tuple[str, str]]
    success: bool
    error_message: Optional[str] = None


class DistributedProcessor:
    """Manages distributed processing of large codebases"""
    
    def __init__(self, index_dir: str, num_workers: int = None, max_queue_size: int = 1000):
        self.index_dir = os.path.abspath(index_dir)
        self.num_workers = num_workers or min(32, mp.cpu_count() + 4)
        self.max_queue_size = max_queue_size
        self.embedder = Embedder()  # Use default embedder
        self.task_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue()
        self.workers = []
        self.is_running = False
        
        # Ensure index directory exists
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Ensure database exists
        db_path = os.path.join(self.index_dir, DB_NAME)
        ensure_db(db_path)
    
    def start_workers(self):
        """Start worker processes"""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Start worker processes
        for i in range(self.num_workers):
            worker = mp.Process(target=self._worker_process, args=(i,))
            worker.start()
            self.workers.append(worker)
    
    def stop_workers(self):
        """Stop worker processes"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Send shutdown signals to all workers
        for _ in range(self.num_workers):
            try:
                self.task_queue.put(None, block=False)  # None signals shutdown
            except queue.Full:
                pass  # Queue is full, that's fine
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)  # Wait up to 5 seconds
            if worker.is_alive():
                worker.terminate()  # Force terminate if still alive
        
        self.workers.clear()
    
    def _worker_process(self, worker_id: int):
        """Worker process function that processes tasks"""
        try:
            while self.is_running:
                try:
                    # Get task from queue (with timeout to allow checking is_running)
                    task = self.task_queue.get(timeout=1)
                    
                    # Check for shutdown signal
                    if task is None:
                        break
                    
                    # Process the task
                    result = self._process_task(task)
                    
                    # Put result in result queue
                    self.result_queue.put(result)
                    
                except queue.Empty:
                    # Timeout, continue loop to check is_running
                    continue
                except Exception as e:
                    # Handle any other exceptions
                    error_result = DistributedResult(
                        task_id=task.task_id if task else "unknown",
                        symbols=[],
                        chunks=[],
                        calls=[],
                        success=False,
                        error_message=str(e)
                    )
                    self.result_queue.put(error_result)
        except Exception as e:
            # Handle catastrophic worker errors
            print(f"Worker {worker_id} failed: {e}")
    
    def _process_task(self, task: DistributedTask) -> DistributedResult:
        """Process a single distributed task"""
        try:
            # Read file content
            with open(task.file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Get adapter for the language
            ensure_default_adapters()
            adapter = get_adapter(task.language)
            
            # Process file with adapter
            from .cli import _parse_and_chunk
            chunk_config = ChunkingConfig(method="symbol", similarity_threshold=0.7)
            result = _parse_and_chunk(
                task.language, 
                adapter, 
                task.file_path, 
                source, 
                self.embedder, 
                chunk_config
            )
            
            # Return successful result
            return DistributedResult(
                task_id=task.task_id,
                symbols=list(result.symbols),
                chunks=list(result.chunks),
                calls=result.calls or [],
                success=True
            )
        except Exception as e:
            # Return error result
            return DistributedResult(
                task_id=task.task_id,
                symbols=[],
                chunks=[],
                calls=[],
                success=False,
                error_message=str(e)
            )
    
    def submit_tasks(self, tasks: List[DistributedTask]) -> List[str]:
        """Submit tasks for distributed processing"""
        submitted_task_ids = []
        
        for task in tasks:
            try:
                self.task_queue.put(task, block=True, timeout=5)
                submitted_task_ids.append(task.task_id)
            except queue.Full:
                print(f"Task queue full, could not submit task {task.task_id}")
                # Could implement retry logic here
        
        return submitted_task_ids
    
    def collect_results(self, timeout: float = 1.0) -> List[DistributedResult]:
        """Collect available results from the result queue"""
        results = []
        
        try:
            while True:
                # Try to get a result (non-blocking after first)
                result = self.result_queue.get(block=len(results) == 0, timeout=timeout)
                results.append(result)
        except queue.Empty:
            # No more results available right now
            pass
        
        return results
    
    def process_large_codebase(self, repo_path: str, incremental: bool = True) -> None:
        """Process a large codebase using distributed processing"""
        from .crawler import iter_python_files
        from .languages import collect_index_targets, get_adapter_for_path
        
        print(f"Processing large codebase at {repo_path} with {self.num_workers} workers")
        
        # Start worker processes
        self.start_workers()
        
        # Collect files to process
        ensure_default_adapters()
        targets = collect_index_targets(repo_path, "auto")
        
        # Create tasks for all files
        tasks = []
        for adapter, file_path in targets:
            # Calculate file hash for change detection in incremental mode
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                    content_hash = hashlib.sha256(content).hexdigest()
            except Exception as e:
                print(f"Could not read {file_path}: {e}")
                continue
            
            task = DistributedTask(
                task_id=f"{adapter.name}_{file_path}",
                file_path=file_path,
                language=adapter.name,
                content_hash=content_hash,
                priority=1
            )
            tasks.append(task)
        
        print(f"Created {len(tasks)} tasks for distributed processing")
        
        # Submit tasks in batches to avoid overwhelming the queue
        batch_size = min(100, self.max_queue_size // 2)
        submitted_count = 0
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            submitted_ids = self.submit_tasks(batch)
            submitted_count += len(submitted_ids)
            print(f"Submitted batch {i//batch_size + 1}/{(len(tasks)-1)//batch_size + 1} ({len(submitted_ids)} tasks)")
        
        print(f"Submitted {submitted_count} tasks for processing")
        
        # Collect results and process them
        processed_count = 0
        all_symbol_rows = []
        all_texts = []
        all_calls = []
        file_hashes = []
        
        from .cli import _format_symbol_row
        
        # Process results as they become available
        while processed_count < submitted_count:
            results = self.collect_results(timeout=5.0)
            
            for result in results:
                processed_count += 1
                
                if result.success:
                    # Format symbols for database insertion
                    for symbol in result.symbols:
                        all_symbol_rows.append(_format_symbol_row(symbol))
                    
                    # Collect text for embedding
                    for chunk in result.chunks:
                        all_texts.append(chunk.text)
                    
                    # Collect calls for database insertion
                    for caller, callee in result.calls:
                        if caller and callee:
                            all_calls.append((result.task_id, caller, callee, None))  # Last None is for path
                    
                    # Collect file hash information
                    # Note: We would need to extract the file path from task_id or store it separately
                    print(f"Processed task {result.task_id}")
                else:
                    print(f"Task {result.task_id} failed: {result.error_message}")
        
        print(f"Processed {processed_count} tasks")
        
        # Insert collected data into database
        if all_symbol_rows or all_texts:
            db_path = os.path.join(self.index_dir, DB_NAME)
            index_path = os.path.join(self.index_dir, FAISS_INDEX)
            
            with db_conn(db_path) as con:
                cur = con.cursor()
                
                # Insert symbols
                if all_symbol_rows:
                    cur.executemany(
                        """
                        INSERT INTO symbols (
                            path,
                            name,
                            kind,
                            start_line,
                            end_line,
                            signature,
                            docstring,
                            imports,
                            bases,
                            language,
                            namespace,
                            symbol_type
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        all_symbol_rows,
                    )
                
                # Insert file hashes for incremental processing
                if file_hashes:
                    cur.executemany(
                        """
                        INSERT OR REPLACE INTO files (path, hash, language)
                        VALUES (?, ?, ?)
                        """,
                        file_hashes,
                    )
                
                # Generate embeddings for collected texts
                if all_texts:
                    print(f"Generating embeddings for {len(all_texts)} text chunks...")
                    vecs = self.embedder.encode(all_texts, batch_size=None)  # Use adaptive batch sizing
                    
                    # Get symbol IDs for vector insertion
                    last_id = cur.execute("SELECT ifnull(MAX(id), 0) FROM symbols;").fetchone()[0]
                    count = len(all_symbol_rows)
                    first_id = last_id - count + 1 if count > 0 else 0
                    symbol_ids = list(range(first_id, last_id + 1)) if count > 0 else []
                    
                    # Add vectors to FAISS and database
                    if symbol_ids:
                        add_vectors(index_path, con, symbol_ids, vecs, batch_size=1000)
                
                # Insert calls
                if all_calls:
                    cur.executemany(
                        """
                        INSERT OR IGNORE INTO calls (caller_id, callee_name, callee_symbol_id, callee_path)
                        VALUES (?, ?, ?, ?);
                        """,
                        [(caller_id, callee_name, callee_symbol_id, callee_path) 
                         for caller_id, callee_name, callee_symbol_id, callee_path in all_calls]
                    )
        
        # Stop worker processes
        self.stop_workers()
        
        print(f"Distributed processing completed. Processed {len(all_texts)} text chunks.")


def process_codebase_distributed(
    repo_path: str, 
    index_dir: str = ".semindex", 
    num_workers: int = None,
    incremental: bool = True
) -> None:
    """Process a codebase using distributed processing for very large codebases.
    
    Args:
        repo_path: Path to the repository to index
        index_dir: Directory to store the index
        num_workers: Number of worker processes (defaults to CPU count + 4)
        incremental: Whether to do incremental processing or full reindex
    """
    # Create distributed processor
    processor = DistributedProcessor(index_dir, num_workers)
    
    # Process the codebase
    processor.process_large_codebase(repo_path, incremental=incremental)