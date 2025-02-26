"""
PNG Metadata Tools Benchmarking Suite

A simple framework for measuring performance of the PNG metadata tools library.

This module provides benchmarking capabilities while adhering to proper 
British standards for software engineering.

Usage:
    python -m benchmarks.run --help
"""

__version__ = "0.1.0"
__author__ = "Proper British Standards"

from benchmarks.utils import BenchmarkUtils

# Import available benchmarks
BENCHMARK_TYPES = {}

# Always import performance benchmark
try:
    from benchmarks.performance import PerformanceBenchmark
    BENCHMARK_TYPES["performance"] = PerformanceBenchmark
except ImportError:
    pass

# Optionally import other benchmarks if they exist
try:
    from benchmarks.memory import MemoryBenchmark
    BENCHMARK_TYPES["memory"] = MemoryBenchmark
except ImportError:
    pass

try:
    from benchmarks.multithreaded import MultithreadedBenchmark
    BENCHMARK_TYPES["multithreaded"] = MultithreadedBenchmark
except ImportError:
    pass

# Define standard benchmark configurations
DEFAULT_CONFIGS = {
    "performance": {
        "small": {"sizes": [(512, 512), (1024, 1024), (2048, 2048)], "iterations": 10},
        "medium": {"sizes": [(1024, 1024), (2048, 2048), (4096, 4096)], "iterations": 5},
        "large": {"sizes": [(2048, 2048), (4096, 4096), (6144, 6144)], "iterations": 3}
    }
}

# Add memory and multithreaded configs if available
if "memory" in BENCHMARK_TYPES:
    DEFAULT_CONFIGS["memory"] = {
        "small": {"sizes": [(512, 512), (1024, 1024), (2048, 2048)], "iterations": 5},
        "medium": {"sizes": [(1024, 1024), (2048, 2048), (4096, 4096)], "iterations": 3},
        "large": {"sizes": [(2048, 2048), (4096, 4096), (6144, 6144)], "iterations": 2}
    }

if "multithreaded" in BENCHMARK_TYPES:
    DEFAULT_CONFIGS["multithreaded"] = {
        "small": {"file_counts": [10, 20, 30], "worker_counts": [1, 2, 4], "operations_per_file": 10},
        "medium": {"file_counts": [30, 60, 90], "worker_counts": [1, 2, 4, 8], "operations_per_file": 10},
        "large": {"file_counts": [50, 100, 150], "worker_counts": [1, 2, 4, 8, 16], "operations_per_file": 5}
    }