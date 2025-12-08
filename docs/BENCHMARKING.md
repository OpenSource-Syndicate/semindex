# Performance Benchmarking for semindex

This document describes the performance benchmarking system created for semindex.

## Overview

The benchmarking system measures the performance of semindex operations including:
- Indexing performance (how fast code repositories can be indexed)
- Embedding generation performance (how fast text can be converted to embeddings)
- Search query performance (how fast queries can be executed against the index)
- Memory usage during operations

## Components

### 1. Benchmark Module (`src/semindex/benchmark.py`)

This module contains:
- `BenchmarkResult`: Data class to store benchmark results
- `MemoryTracker`: Class to track memory usage
- `BenchmarkRunner`: Main runner class to execute benchmarks
- Functions to create sample code and repositories for testing
- Dedicated benchmarking functions for each operation type

### 2. Benchmark Runner Script (`scripts/run_benchmarks.py`)

This script provides:
- Command-line interface to run different benchmark suites
- Multiple test scenarios with different code sizes and complexities
- Comparison of different chunking methods
- Memory usage and execution time measurements
- Throughput calculations (operations per second)

## Usage

### Running All Benchmarks

```bash
python scripts/run_benchmarks.py
```

### Running Specific Benchmark Categories

```bash
# Run only indexing benchmarks
python scripts/run_benchmarks.py --benchmark indexing

# Run only embedding benchmarks
python scripts/run_benchmarks.py --benchmark embedding

# Run only search benchmarks
python scripts/run_benchmarks.py --benchmark search

# Run incremental indexing benchmarks
python scripts/run_benchmarks.py --benchmark incremental

# Run complete suite (default)
python scripts/run_benchmarks.py --benchmark complete
```

### Saving Results to Plot

```bash
python scripts/run_benchmarks.py --output-plot results.png
```

## Test Scenarios

The benchmark system includes various test scenarios:

1. **Different Code Sizes**
   - Small (50 lines)
   - Medium (200 lines)
   - Large (1000 lines)

2. **Different File Counts**
   - Few large files
   - Many small files
   - Mixed file sizes

3. **Different Operations**
   - Indexing with different chunking methods (symbol vs semantic)
   - Embedding generation with different batch sizes
   - Search with different top-k values

## Metrics Tracked

- Execution time for each operation
- Memory usage before and after operations
- Memory delta (how much memory was consumed)
- Peak memory usage during operations
- Throughput (operations per second)
- Number of items processed

## Performance Insights

The benchmark system helps identify:

1. **Performance bottlenecks** in the indexing pipeline
2. **Memory efficiency** of different operations
3. **Optimal batch sizes** for embedding generation
4. **Scalability** with different repository sizes
5. **Effectiveness** of different chunking strategies

## Extending Benchmarks

To add new benchmarks:
1. Add a new function in `benchmark.py` that follows the same pattern
2. Update the runner script to call your new benchmark function
3. Ensure proper memory and time tracking using the provided utilities