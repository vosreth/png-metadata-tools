"""
PNG Metadata Tools Benchmarking Script

A simple interactive utility for running benchmarks.
This script uses the native Python interpreter, making it compatible
with your virtual environment without additional configuration.
"""

import os
import sys
import subprocess
from pathlib import Path

def ensure_dirs():
    """Ensure benchmark directories exist."""
    Path("benchmarks/data").mkdir(parents=True, exist_ok=True)
    Path("benchmarks/results").mkdir(parents=True, exist_ok=True)

def get_available_benchmarks():
    """Determine which benchmarks are available."""
    # Import the benchmark types dictionary
    try:
        from benchmarks import BENCHMARK_TYPES
        return list(BENCHMARK_TYPES.keys())
    except ImportError:
        # If cannot import, default to performance only
        return ["performance"]

def run_benchmark():
    """Run the benchmark with user-selected options."""
    print("PNG Metadata Tools Benchmark")
    print("===========================")
    print()
    
    # Get available benchmarks
    benchmarks = get_available_benchmarks()
    
    if not benchmarks:
        print("Error: No benchmark modules found!")
        input("\nPress Enter to exit...")
        return
    
    # Select benchmark type
    if len(benchmarks) > 1:
        print("Select benchmark type:")
        for i, benchmark in enumerate(benchmarks, 1):
            print(f"{i}. {benchmark.capitalize()} benchmark")
        print()
        
        while True:
            try:
                selection = input(f"Enter selection (1-{len(benchmarks)}): ")
                benchmark_idx = int(selection) - 1
                if 0 <= benchmark_idx < len(benchmarks):
                    benchmark_type = benchmarks[benchmark_idx]
                    break
                else:
                    print(f"Invalid selection. Please enter a number between 1 and {len(benchmarks)}.")
            except ValueError:
                print("Please enter a valid number.")
            except KeyboardInterrupt:
                print("\nBenchmark cancelled.")
                return
    else:
        # Only one benchmark available
        benchmark_type = benchmarks[0]
        print(f"Using {benchmark_type.capitalize()} benchmark (only available type)")
        print()
    
    # Select preset or custom benchmark
    print("Select benchmark preset:")
    print("1. Small  (512, 1024, 2048 - 10 iterations)")
    print("2. Medium (1024, 2048, 4096 - 5 iterations)")
    print("3. Large  (2048, 4096, 6144 - 3 iterations)")
    print("4. Custom configuration")
    print()
    
    while True:
        try:
            preset = input("Enter selection (1-4): ")
            
            if preset == "1":
                cmd_args = ["--preset", "small"]
                break
            elif preset == "2":
                cmd_args = ["--preset", "medium"]
                break
            elif preset == "3":
                cmd_args = ["--preset", "large"]
                break
            elif preset == "4":
                cmd_args = []
                
                # Get custom sizes
                sizes = input("\nEnter image sizes (e.g., 512,1024,2048): ").strip()
                if sizes:
                    cmd_args.extend(["--sizes", sizes])
                    
                # Get custom iterations
                iterations = input("\nEnter iterations (default: 10 for performance, 5 for memory): ").strip()
                if iterations:
                    cmd_args.extend(["--iterations", iterations])
                
                break
            else:
                print("Invalid selection. Please enter a number between 1 and 4.")
        except KeyboardInterrupt:
            print("\nBenchmark cancelled.")
            return
    
    # Ensure directories exist
    ensure_dirs()
    
    # Construct the command
    cmd = [sys.executable, "-m", "benchmarks.run", "--type", benchmark_type, "--output", "all", "--verbose"]
    cmd.extend(cmd_args)
    
    print("\nRunning benchmark...")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        # Run the benchmark as a subprocess
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("\nBenchmark completed successfully!")
            print("Results are available in the benchmarks/results directory.")
        else:
            print(f"\nBenchmark failed with exit code {result.returncode}")
    except Exception as e:
        print(f"\nError running benchmark: {e}")
    
    # Wait for user input before exiting
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    run_benchmark()