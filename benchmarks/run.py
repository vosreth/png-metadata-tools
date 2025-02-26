"""
Command-line interface for running PNG metadata benchmarks.

This module provides a convenient entry point for running various
benchmark types with configurable parameters.
"""

import argparse
import sys
import traceback
from typing import List, Tuple

from benchmarks import (
    BENCHMARK_TYPES,
    DEFAULT_CONFIGS,
    BenchmarkUtils
)

def parse_size_argument(size_arg: str) -> List[Tuple[int, int]]:
    """
    Parse size argument into a list of (width, height) tuples.
    
    Args:
        size_arg: Comma-separated list of sizes (e.g., "512,1024,2048" or "512x512,1024x1024")
        
    Returns:
        List of (width, height) tuples
    """
    sizes = []
    for size_str in size_arg.split(','):
        if 'x' in size_str:
            # Format: "WIDTHxHEIGHT"
            width, height = map(int, size_str.split('x'))
            sizes.append((width, height))
        else:
            # Format: single number (square image)
            size = int(size_str)
            sizes.append((size, size))
    return sizes

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PNG Metadata Benchmarking Suite",
        epilog="Example: python -m benchmarks.run --type performance --sizes 512,1024,2048"
    )
    
    # Only show available benchmark types
    parser.add_argument("--type", choices=list(BENCHMARK_TYPES.keys()),
                       default="performance" if "performance" in BENCHMARK_TYPES else None,
                       help="Type of benchmark to run")
    
    parser.add_argument("--preset", choices=["small", "medium", "large"],
                       help="Use preset configuration (overrides other options)")
    
    # Performance and Memory benchmark options
    parser.add_argument("--sizes", default="512,1024,2048",
                       help="Comma-separated list of image sizes (size or widthxheight)")
    parser.add_argument("--iterations", type=int, default=10,
                       help="Number of iterations for each test")
    
    # Multithreaded benchmark options
    parser.add_argument("--file-counts", default="10,30,50",
                       help="Comma-separated list of file counts for multithreaded tests")
    parser.add_argument("--worker-counts", default="1,2,4,8",
                       help="Comma-separated list of worker counts for multithreaded tests")
    parser.add_argument("--operations-per-file", type=int, default=10,
                       help="Number of operations per file for multithreaded tests")
    
    # Output options
    parser.add_argument("--output", choices=["json", "csv", "markdown", "all"],
                       default="csv", help="Output format for results")
    
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    return parser.parse_args()

def run_performance_benchmark(args):
    """Run performance benchmark with given arguments."""
    # Import the performance benchmark class
    PerformanceBenchmark = BENCHMARK_TYPES["performance"]
    
    # Check if using preset configuration
    if args.preset and args.preset in DEFAULT_CONFIGS["performance"]:
        config = DEFAULT_CONFIGS["performance"][args.preset]
        sizes = config["sizes"]
        iterations = config["iterations"]
    else:
        # Parse sizes argument
        sizes = parse_size_argument(args.sizes)
        iterations = args.iterations
    
    # Run benchmark
    print(f"Running performance benchmark with {iterations} iterations for each size...")
    benchmark = PerformanceBenchmark(sizes, iterations)
    benchmark.run()
    
    # Save results
    saved_files = benchmark.save_results(args.output)
    
    print("\nPerformance benchmark complete!")
    for format_type, path in saved_files.items():
        print(f"Results saved as {format_type}: {path}")

def run_memory_benchmark(args):
    """Run memory benchmark with given arguments."""
    if "memory" not in BENCHMARK_TYPES:
        print("Memory benchmark not available. Please implement memory.py first.")
        return
        
    # Import the memory benchmark class
    MemoryBenchmark = BENCHMARK_TYPES["memory"]
    
    # Check if using preset configuration
    if args.preset and args.preset in DEFAULT_CONFIGS.get("memory", {}):
        config = DEFAULT_CONFIGS["memory"][args.preset]
        sizes = config["sizes"]
        iterations = config["iterations"]
    else:
        # Parse sizes argument
        sizes = parse_size_argument(args.sizes)
        iterations = args.iterations if args.iterations else 5  # Default to 5 for memory benchmark
    
    # Run benchmark
    print(f"Running memory benchmark with {iterations} iterations for each size...")
    benchmark = MemoryBenchmark(sizes, iterations)
    benchmark.run()
    
    # Save results
    saved_files = benchmark.save_results(args.output)
    
    print("\nMemory benchmark complete!")
    for format_type, path in saved_files.items():
        print(f"Results saved as {format_type}: {path}")

def run_multithreaded_benchmark(args):
    """Run multithreaded benchmark with given arguments."""
    if "multithreaded" not in BENCHMARK_TYPES:
        print("Multithreaded benchmark not available. Please implement multithreaded.py first.")
        return
        
    # Import the multithreaded benchmark class
    MultithreadedBenchmark = BENCHMARK_TYPES["multithreaded"]
    
    # Check if using preset configuration
    if args.preset and args.preset in DEFAULT_CONFIGS.get("multithreaded", {}):
        config = DEFAULT_CONFIGS["multithreaded"][args.preset]
        file_counts = config["file_counts"]
        worker_counts = config["worker_counts"]
        operations_per_file = config["operations_per_file"]
    else:
        # Parse arguments
        file_counts = [int(x) for x in args.file_counts.split(',')] if args.file_counts else [10, 30, 50]
        worker_counts = [int(x) for x in args.worker_counts.split(',')] if args.worker_counts else [1, 2, 4, 8]
        operations_per_file = args.operations_per_file
    
    # Run benchmark
    print(f"Running multithreaded benchmark with files={file_counts}, workers={worker_counts}...")
    benchmark = MultithreadedBenchmark(file_counts, worker_counts, operations_per_file)
    benchmark.run()
    
    # Save results
    saved_files = benchmark.save_results(args.output)
    
    print("\nMultithreaded benchmark complete!")
    for format_type, path in saved_files.items():
        print(f"Results saved as {format_type}: {path}")

def main():
    """Main entry point for the benchmark suite."""
    # Check if any benchmarks are available
    if not BENCHMARK_TYPES:
        print("Error: No benchmark implementations found.")
        print("Please ensure at least one benchmark module is properly installed.")
        return 1
        
    # Ensure benchmark directories exist
    BenchmarkUtils.ensure_dirs()
    
    # Parse command-line arguments
    args = parse_args()
    
    # Check if benchmark type was specified
    if args.type is None:
        print(f"Error: No benchmark type specified. Available types: {', '.join(BENCHMARK_TYPES.keys())}")
        return 1
        
    try:
        if args.verbose:
            print(f"Starting {args.type} benchmark with the following settings:")
            for arg, value in vars(args).items():
                print(f"  {arg}: {value}")
            print()
        
        # Run appropriate benchmark
        if args.type == "performance" and "performance" in BENCHMARK_TYPES:
            run_performance_benchmark(args)
        elif args.type == "memory" and "memory" in BENCHMARK_TYPES:
            run_memory_benchmark(args)
        elif args.type == "multithreaded" and "multithreaded" in BENCHMARK_TYPES:
            run_multithreaded_benchmark(args)
        else:
            print(f"Unknown or unavailable benchmark type: {args.type}")
            return 1
            
        return 0
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
        return 130
    except Exception as e:
        print(f"\nError running benchmark: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())