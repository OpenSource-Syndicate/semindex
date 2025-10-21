#!/usr/bin/env python3
"""
Benchmark runner script for semindex performance testing.
This script runs various benchmarks on semindex components.
"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

from semindex.benchmark import (
    BenchmarkRunner,
    create_sample_repo,
    benchmark_indexing,
    benchmark_embedding_generation,
    benchmark_search_queries,
    create_sample_code
)


def run_indexing_benchmarks():
    """Run indexing performance benchmarks."""
    print("Running indexing benchmarks...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample repositories of different sizes and complexities
        scenarios = [
            ("small_simple", "small", 5),
            ("medium_simple", "medium", 10),
            ("large_simple", "large", 15),
            ("small_complex", "large", 5),  # Complex but fewer files
            ("large_simple_many_files", "small", 50),  # Many small files
        ]
        
        results = []
        
        for name, size, num_files in scenarios:
            print(f"  Running {name} scenario...")
            repo_path = create_sample_repo(os.path.join(temp_dir, f"repo_{name}"), size=size, num_files=num_files)
            index_dir = os.path.join(temp_dir, f"index_{name}")
            
            result = benchmark_indexing(repo_path, index_dir)
            results.append(result)
            print(f"    {name}: {result.duration:.4f}s, {result.throughput:.2f} files/sec")
        
        return results


def run_embedding_benchmarks():
    """Run embedding generation benchmarks."""
    print("Running embedding benchmarks...")
    
    # Create sample texts with different complexities
    scenarios = [
        ("small_texts", "small", 50),
        ("medium_texts", "medium", 30),
        ("large_texts", "large", 20),
        ("mixed_complexity", "medium", 40),  # Will use varied complexity
    ]
    
    results = []
    
    for name, size, count in scenarios:
        print(f"  Running {name} scenario...")
        
        # Create sample text of specified size
        sample_text = create_sample_code(size)
        
        # For mixed complexity, create texts of different sizes
        if name == "mixed_complexity":
            texts = []
            sizes = ["small", "medium", "large"]
            for i in range(count):
                text_size = sizes[i % len(sizes)]
                texts.append(create_sample_code(text_size))
        else:
            texts = [sample_text] * count
        
        result = benchmark_embedding_generation(texts)
        results.append(result)
        print(f"    {name}: {result.duration:.4f}s, {result.throughput:.2f} texts/sec")
    
    # Test different batch sizes
    print("  Testing different batch sizes...")
    sample_texts = [create_sample_code("medium") for _ in range(100)]
    batch_sizes = [1, 4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        result = benchmark_embedding_generation(sample_texts, batch_size=batch_size)
        result.name = f"Embedding Generation Performance (batch_size={batch_size})"
        results.append(result)
        print(f"    Batch size {batch_size}: {result.duration:.4f}s, {result.throughput:.2f} texts/sec")
    
    return results


def run_search_benchmarks():
    """Run search query benchmarks."""
    print("Running search benchmarks...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample repos of different sizes to search
        scenarios = [
            ("small_search", "small", 5),
            ("medium_search", "medium", 10),
            ("large_search", "large", 20),
        ]
        
        results = []
        
        for name, size, num_files in scenarios:
            print(f"  Running {name} search scenario...")
            
            # Create a sample repo to search
            repo_path = create_sample_repo(os.path.join(temp_dir, f"search_test_repo_{name}"), size=size, num_files=num_files)
            
            # Create an index
            index_dir = os.path.join(temp_dir, f"search_index_{name}")
            benchmark_indexing(repo_path, index_dir)
            
            # Define test queries
            queries = [
                "how to implement a class",
                "function to calculate sum",
                "method for data processing",
                "error handling implementation",
                "initialize a new object"
            ]
            
            # Run search benchmark
            result = benchmark_search_queries(index_dir, queries)
            result.name = f"Search Query Performance ({name})"
            results.append(result)
            print(f"    {name}: {result.duration:.4f}s, {result.throughput:.2f} queries/sec")
        
        # Test search performance with different top_k values
        print("  Testing search with different top_k values...")
        
        # Use medium-sized repo for this test
        medium_repo = create_sample_repo(os.path.join(temp_dir, "search_topk_repo"), size="medium", num_files=15)
        medium_index = os.path.join(temp_dir, "search_topk_index")
        benchmark_indexing(medium_repo, medium_index)
        
        top_k_values = [5, 10, 20, 50]
        queries = ["implement class", "function method", "calculate value"]
        
        for k in top_k_values:
            result = benchmark_search_queries(medium_index, queries, top_k=k)
            result.name = f"Search Query Performance (top_k={k})"
            results.append(result)
            print(f"    top_k {k}: {result.duration:.4f}s, {result.throughput:.2f} queries/sec")
        
        return results


def run_chunking_method_comparison():
    """Compare different chunking methods performance."""
    print("Running chunking method comparison benchmarks...")
    
    from semindex.indexer import Indexer
    from semindex.benchmark import MemoryTracker
    import time
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample repo to test chunking
        repo_path = create_sample_repo(os.path.join(temp_dir, "chunking_repo"), size="medium", num_files=10)
        
        chunking_methods = ["symbol", "semantic"]
        results = []
        
        for method in chunking_methods:
            print(f"  Testing {method} chunking method...")
            
            index_dir = os.path.join(temp_dir, f"index_{method}")
            
            # Run indexing with specific chunking method
            indexer = Indexer(index_dir=index_dir, model="microsoft/codebert-base")
            
            start_time = time.perf_counter()
            mem_before = MemoryTracker.get_memory_usage()
            
            indexer.index_path(repo_path, chunking=method, verbose=False)
            
            end_time = time.perf_counter()
            mem_after = MemoryTracker.get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = mem_after - mem_before
            
            import os
            files_count = sum(len(files) for _, _, files in os.walk(repo_path) if files)
            throughput = files_count / duration if duration > 0 else 0
            
            result = {
                "name": f"Indexing with {method} chunking",
                "duration": duration,
                "memory_before": mem_before,
                "memory_after": mem_after,
                "throughput": throughput,
                "extra_metrics": {
                    "memory_delta": memory_delta,
                    "files_processed": files_count,
                    "chunking_method": method
                }
            }
            
            from semindex.benchmark import BenchmarkResult
            benchmark_result = BenchmarkResult(**result)
            results.append(benchmark_result)
            
            print(f"    {method} chunking: {duration:.4f}s, {throughput:.2f} files/sec")
        
        return results


def run_complete_benchmark_suite():
    """Run the complete benchmark suite."""
    print("Starting semindex performance benchmark suite...")
    print("="*60)
    
    all_results = []
    
    # Run each benchmark category
    indexing_results = run_indexing_benchmarks()
    all_results.extend(indexing_results)
    
    print()
    embedding_results = run_embedding_benchmarks()
    all_results.extend(embedding_results)
    
    print()
    search_results = run_search_benchmarks()
    all_results.extend(search_results)
    
    print()
    chunking_results = run_chunking_method_comparison()
    all_results.extend(chunking_results)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUITE COMPLETED")
    print("="*60)
    
    total_duration = sum(r.duration for r in all_results)
    print(f"Total benchmark execution time: {total_duration:.4f}s")
    
    # Print detailed results
    for result in all_results:
        print(f"\n{result.name}:")
        print(f"  Duration: {result.duration:.4f}s")
        print(f"  Memory Delta: {result.extra_metrics['memory_delta'] / 1024 / 1024:.2f} MB")
        if result.throughput:
            print(f"  Throughput: {result.throughput:.2f} ops/sec")
    
    return all_results


def run_incremental_indexing_benchmarks():
    """Run incremental indexing benchmarks."""
    print("Running incremental indexing benchmarks...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create initial repo
        repo_path = os.path.join(temp_dir, "incremental_repo")
        index_dir = os.path.join(temp_dir, "incremental_index")
        
        # Create initial set of files
        create_sample_repo(repo_path, size="medium", num_files=10)
        
        # Initial indexing
        initial_result = benchmark_indexing(repo_path, index_dir)
        print(f"  Initial indexing: {initial_result.duration:.4f}s")
        
        # Add some new files to simulate changes
        for i in range(5):
            code = create_sample_code("medium")
            file_path = Path(repo_path) / f"new_file_{i}.py"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
        
        # Run incremental indexing
        incremental_result = benchmark_indexing(repo_path, index_dir, model=initial_result.extra_metrics.get('model', 'microsoft/codebert-base'))
        print(f"  Incremental indexing: {incremental_result.duration:.4f}s")
        
        return [initial_result, incremental_result]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run semindex performance benchmarks")
    parser.add_argument("--benchmark", choices=["indexing", "embedding", "search", "complete", "incremental"], 
                       default="complete", help="Which benchmark to run")
    parser.add_argument("--output-plot", type=str, help="Path to save benchmark results plot")
    
    args = parser.parse_args()
    
    if args.benchmark == "indexing":
        results = run_indexing_benchmarks()
    elif args.benchmark == "embedding":
        results = run_embedding_benchmarks()
    elif args.benchmark == "search":
        results = run_search_benchmarks()
    elif args.benchmark == "incremental":
        results = run_incremental_indexing_benchmarks()
    else:  # complete
        results = run_complete_benchmark_suite()
    
    # Print results
    runner = BenchmarkRunner()
    runner.results = results  # Set the results directly
    runner.print_results()
    
    # Save plot if requested
    if args.output_plot:
        runner.plot_results(args.output_plot)
        print(f"Plot saved to {args.output_plot}")