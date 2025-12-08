"""
Performance benchmarking module for semindex.
This module provides tools to measure the performance of various semindex operations.
"""
import time
import psutil
import os
import sys
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import gc


@dataclass
class BenchmarkResult:
    """Data class to store benchmark results."""
    name: str
    duration: float  # in seconds
    memory_before: int  # in bytes
    memory_after: int   # in bytes
    peak_memory: Optional[int] = None  # peak memory usage during operation
    throughput: Optional[float] = None  # ops/second or items/second
    extra_metrics: Optional[Dict[str, Any]] = None


class MemoryTracker:
    """Advanced memory usage tracker."""
    @staticmethod
    def get_memory_usage() -> int:
        """Get current process memory usage in bytes."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss  # Resident Set Size
    
    @staticmethod
    def get_peak_memory() -> int:
        """Get peak memory usage since the process started."""
        # For more detailed peak memory tracking, we can use resource module on Unix systems
        # But for cross-platform, we'll just return current memory
        # In a real implementation, this could use platform-specific tools
        return MemoryTracker.get_memory_usage()


class BenchmarkRunner:
    """Main benchmark runner class."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    @contextmanager
    def benchmark_context(self, name: str):
        """Context manager to measure execution time and memory usage."""
        # Force garbage collection before benchmarking to get more accurate memory measurements
        gc.collect()
        
        start_time = time.perf_counter()
        mem_before = MemoryTracker.get_memory_usage()
        peak_memory = mem_before  # Initialize peak memory
        
        try:
            yield
        finally:
            # Force garbage collection after benchmarking
            gc.collect()
            
            end_time = time.perf_counter()
            mem_after = MemoryTracker.get_memory_usage()
            
            # Track if memory increased during operation
            peak_memory = max(peak_memory, mem_after, mem_before)
            
            duration = end_time - start_time
            memory_delta = mem_after - mem_before
            
            result = BenchmarkResult(
                name=name,
                duration=duration,
                memory_before=mem_before,
                memory_after=mem_after,
                peak_memory=peak_memory,
                extra_metrics={
                    "memory_delta": memory_delta,
                    "peak_memory_increase": peak_memory - mem_before
                }
            )
            self.results.append(result)
    
    def run_benchmark(self, name: str, func: Callable, *args, **kwargs) -> BenchmarkResult:
        """Run a single benchmark function."""
        with self.benchmark_context(name) as _:
            result = func(*args, **kwargs)
        return result
    
    def run_multiple_benchmarks(self, iterations: int = 1) -> List[BenchmarkResult]:
        """Run all registered benchmarks."""
        all_results = []
        for _ in range(iterations):
            # We'll run benchmarks in the order they were added
            # For now we just return the results list
            all_results.extend(self.results)
        return all_results
    
    def print_results(self):
        """Print benchmark results in a formatted way."""
        print("\n" + "="*80)
        print("SEMINDEX PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        
        for result in self.results:
            print(f"\nBenchmark: {result.name}")
            print(f"  Duration: {result.duration:.4f}s")
            print(f"  Memory Before: {result.memory_before / 1024 / 1024:.2f} MB")
            print(f"  Memory After: {result.memory_after / 1024 / 1024:.2f} MB")
            print(f"  Memory Delta: {result.extra_metrics['memory_delta'] / 1024 / 1024:.2f} MB")
            if result.peak_memory:
                print(f"  Peak Memory: {result.peak_memory / 1024 / 1024:.2f} MB")
                print(f"  Peak Memory Increase: {result.extra_metrics['peak_memory_increase'] / 1024 / 1024:.2f} MB")
            if result.throughput:
                print(f"  Throughput: {result.throughput:.2f} ops/sec")
        
        print("\n" + "="*80)
    
    def plot_results(self, output_path: Optional[str] = None):
        """Plot benchmark results."""
        if not self.results:
            print("No results to plot.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        names = [r.name for r in self.results]
        durations = [r.duration for r in self.results]
        memory_deltas = [r.extra_metrics['memory_delta'] / 1024 / 1024 for r in self.results]  # MB
        peak_memory_increases = [r.extra_metrics['peak_memory_increase'] / 1024 / 1024 for r in self.results]  # MB
        throughputs = [r.throughput or 0 for r in self.results]
        
        # Plot execution time
        axes[0].bar(names, durations)
        axes[0].set_title('Execution Time (seconds)')
        axes[0].set_ylabel('Time (s)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot memory usage delta
        axes[1].bar(names, memory_deltas)
        axes[1].set_title('Memory Usage Change (MB)')
        axes[1].set_ylabel('Memory (MB)')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Plot peak memory increase
        axes[2].bar(names, peak_memory_increases)
        axes[2].set_title('Peak Memory Increase (MB)')
        axes[2].set_ylabel('Memory (MB)')
        axes[2].tick_params(axis='x', rotation=45)
        
        # Plot throughput
        axes[3].bar(names, throughputs)
        axes[3].set_title('Throughput (operations/second)')
        axes[3].set_ylabel('Ops/sec')
        axes[3].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()


def create_sample_code(size: str = "medium") -> str:
    """
    Create sample code of different sizes for benchmarking.
    
    Args:
        size: "small", "medium", or "large"
    """
    sizes = {
        "small": 50,      # ~50 lines of code
        "medium": 200,    # ~200 lines of code
        "large": 1000     # ~1000 lines of code
    }
    
    lines = sizes.get(size, 200)
    
    code = [
        "class SampleClass:",
        "    \"\"\"A sample class for benchmarking.\"\"\"",
        "",
        "    def __init__(self, value=0):",
        "        self.value = value",
        "",
        "    def method1(self):",
        "        result = 0",
        "        for i in range(100):",
        "            result += i",
        "        return result",
        "",
        "    def method2(self, x, y):",
        "        \"\"\"Another sample method.\"\"\"",
        "        if x > y:",
        "            return x",
        "        else:",
        "            return y",
    ]
    
    # Add more methods based on the desired size
    num_extra_methods = (lines - len(code)) // 5  # Rough estimate
    
    for i in range(num_extra_methods):
        code.extend([
            f"",
            f"    def extra_method_{i}(self, param):",
            f"        \"\"\"Extra method {i} for larger code sample.\"\"\"",
            f"        return param * {i}",
            f""
        ])
    
    # Add a function at the end
    code.extend([
        "",
        f"def util_function_{size}():",
        f"    \"\"\"Utility function for {size} sample.\"\"\"",
        f"    obj = SampleClass()",
        f"    return obj.method1() + obj.method2(10, 20)"
    ])
    
    return "\n".join(code)


def create_sample_repo(repo_path: str, size: str = "medium", num_files: int = 5) -> str:
    """
    Create a sample repository with multiple code files.
    
    Args:
        repo_path: Path to create the sample repo
        size: Size of each file ("small", "medium", "large")
        num_files: Number of files to create
    """
    repo_path = Path(repo_path)
    repo_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_files):
        code = create_sample_code(size=size)
        file_path = repo_path / f"sample_{i}.py"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
    
    return str(repo_path)


def benchmark_indexing(repo_path: str, index_dir: str = ".semindex_benchmark", model: str = "microsoft/codebert-base"):
    """
    Benchmark the indexing process.
    
    Args:
        repo_path: Path to the repository to index
        index_dir: Directory to store the index
        model: Model to use for embeddings
    """
    from .indexer import Indexer
    
    indexer = Indexer(index_dir=index_dir, model=model)
    
    # Perform indexing and measure performance
    start_time = time.perf_counter()
    mem_before = MemoryTracker.get_memory_usage()
    
    indexer.index_path(repo_path, verbose=False)
    
    end_time = time.perf_counter()
    mem_after = MemoryTracker.get_memory_usage()
    
    duration = end_time - start_time
    memory_delta = mem_after - mem_before
    
    # Calculate throughput
    import os
    files_count = sum(len(files) for _, _, files in os.walk(repo_path) if files)
    throughput = files_count / duration if duration > 0 else 0
    
    result = BenchmarkResult(
        name="Indexing Performance",
        duration=duration,
        memory_before=mem_before,
        memory_after=mem_after,
        throughput=throughput,
        extra_metrics={
            "memory_delta": memory_delta,
            "files_processed": files_count
        }
    )
    
    return result


def benchmark_embedding_generation(texts: List[str], model: str = "microsoft/codebert-base", batch_size: int = 16):
    """
    Benchmark embedding generation performance.
    
    Args:
        texts: List of text snippets to encode
        model: Model to use for embeddings
        batch_size: Batch size for encoding
    """
    from .embed import Embedder
    
    embedder = Embedder(model_name=model)
    
    start_time = time.perf_counter()
    mem_before = MemoryTracker.get_memory_usage()
    
    embeddings = embedder.encode(texts, batch_size=batch_size)
    
    end_time = time.perf_counter()
    mem_after = MemoryTracker.get_memory_usage()
    
    duration = end_time - start_time
    memory_delta = mem_after - mem_before
    
    throughput = len(texts) / duration if duration > 0 else 0
    
    result = BenchmarkResult(
        name="Embedding Generation Performance",
        duration=duration,
        memory_before=mem_before,
        memory_after=mem_after,
        throughput=throughput,
        extra_metrics={
            "memory_delta": memory_delta,
            "embeddings_generated": len(embeddings),
            "avg_embedding_time": duration / len(texts) if texts else 0
        }
    )
    
    return result


def benchmark_search_queries(index_dir: str, queries: List[str], top_k: int = 10, model: str = "microsoft/codebert-base"):
    """
    Benchmark search query performance.
    
    Args:
        index_dir: Directory containing the index
        queries: List of search queries to execute
        top_k: Number of top results to return
        model: Model to use for query embeddings
    """
    from .search import Searcher
    
    searcher = Searcher(index_dir=index_dir, model=model)
    
    start_time = time.perf_counter()
    mem_before = MemoryTracker.get_memory_usage()
    
    total_results = 0
    for query in queries:
        results = searcher.query(query, top_k=top_k)
        total_results += len(results)
    
    end_time = time.perf_counter()
    mem_after = MemoryTracker.get_memory_usage()
    
    duration = end_time - start_time
    memory_delta = mem_after - mem_before
    
    throughput = len(queries) / duration if duration > 0 else 0
    
    result = BenchmarkResult(
        name="Search Query Performance",
        duration=duration,
        memory_before=mem_before,
        memory_after=mem_after,
        throughput=throughput,
        extra_metrics={
            "memory_delta": memory_delta,
            "queries_executed": len(queries),
            "total_results": total_results,
            "avg_query_time": duration / len(queries) if queries else 0
        }
    )
    
    return result


if __name__ == "__main__":
    # Example usage of the benchmark runner
    print("This is the benchmarking module. Run the benchmark runner script to execute benchmarks.")