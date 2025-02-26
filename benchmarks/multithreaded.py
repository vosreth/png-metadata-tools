"""
Multithreaded benchmarks for PNG metadata operations with proper British standards.

This module provides comprehensive benchmarking for concurrent metadata operations,
measuring performance across varying thread counts, file quantities, and operation types.
"""

import time
import json
import csv
import statistics
import threading
import concurrent.futures
import queue
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from datetime import datetime
import random

from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Import from parent package
from png_metadata_tools.metadata_queue import PNGMetadataQueue
from png_metadata_tools.batch_operations import BatchEditor, BatchProcessor
from png_metadata_tools.base_handler import PNGMetadataHandlerBase
from png_metadata_tools.chunk_handler import PNGMetadataHandler
from png_metadata_tools.streaming_chunk_handler import StreamingPNGMetadataHandler

# Import local modules
from benchmarks.utils import BenchmarkUtils, RESULTS_DIR


class MultithreadedBenchmark:
    """
    Benchmark for measuring performance of multithreaded PNG metadata operations.
    
    This benchmark evaluates:
    1. Queue throughput with varying worker counts
    2. Batch processing performance under different concurrency models
    3. Lock contention scenarios with shared files
    4. Scaling efficiency as file count increases
    """
    
    def __init__(self, file_counts: List[int], worker_counts: List[int], operations_per_file: int = 10,
                image_size: Tuple[int, int] = (512, 512)):
        """
        Initialize the multithreaded benchmark.
        
        Args:
            file_counts: List of file quantities to test
            worker_counts: List of worker thread counts to test
            operations_per_file: Number of operations to perform per file
            image_size: Size of test images (width, height)
        """
        self.file_counts = file_counts
        self.worker_counts = worker_counts
        self.operations_per_file = operations_per_file
        self.image_size = image_size
        self.results = {}
        
    def run(self):
        """
        Run the multithreaded benchmark and return the results with enhanced error handling.
        
        Returns:
            Dictionary containing benchmark results
        """
        # Ensure directories exist
        from benchmarks.utils import BenchmarkUtils, DATA_DIR
        BenchmarkUtils.ensure_dirs()
        
        # Create specific directory for multithreaded tests
        multithreaded_dir = DATA_DIR / "multithreaded"
        multithreaded_dir.mkdir(parents=True, exist_ok=True)
        print(f"Benchmark directory: {multithreaded_dir.absolute()}")
        
        # Generate test images (only need to do this once, regardless of file_counts)
        max_files = max(self.file_counts)
        print(f"Preparing {max_files} test files...")
        
        try:
            images = self._generate_test_images(max_files)
            print(f"Successfully generated {len(images)} test images")
        except Exception as e:
            print(f"ERROR: Failed to generate test images: {e}")
            raise
        
        results = {}
        
        print("\nRunning multithreaded benchmark...")
        try:
            from tqdm import tqdm
            progress_tracker = tqdm
        except ImportError:
            # Simple progress tracker if tqdm is not available
            def progress_tracker(iterable, **kwargs):
                print(f"Processing {len(iterable)} configurations...")
                return iterable
        
        # Store all configurations we need to test
        configs = []
        for file_count in self.file_counts:
            for worker_count in self.worker_counts:
                configs.append((file_count, worker_count))
        
        # Verify images before starting benchmarks
        valid_images = [path for path in images if path.exists()]
        if len(valid_images) < max_files:
            print(f"WARNING: Only {len(valid_images)} of {max_files} test images are valid")
            if not valid_images:
                print("ERROR: No valid test images available! Cannot continue.")
                raise RuntimeError("No valid test images for benchmark")
        
        # Run benchmarks for each configuration
        config_count = 0
        for file_count, worker_count in progress_tracker(configs, desc="Testing configurations"):
            config_count += 1
            print(f"\nConfiguration {config_count}/{len(configs)}: {file_count} files, {worker_count} workers")
            
            # Verify we have enough valid images for this configuration
            if file_count > len(valid_images):
                print(f"WARNING: Requested {file_count} files but only {len(valid_images)} are available")
                print(f"Using all {len(valid_images)} available files for this test")
                test_files = valid_images
            else:
                # Subset of images for this file count
                test_files = valid_images[:file_count]
            
            print(f"Using {len(test_files)} test files for this configuration")
            
            # Reset files between tests
            try:
                self._reset_test_files(test_files)
            except RuntimeError as e:
                print(f"ERROR: Unable to reset test files: {e}")
                print("Skipping this configuration")
                continue
            
            try:
                # Run the individual benchmarks
                print("Running queue benchmark...")
                queue_results = self._benchmark_queue(test_files, worker_count)
                
                print("Running batch editor benchmark...")
                batch_results = self._benchmark_batch_editor(test_files, worker_count)
                
                print("Running shared files benchmark...")
                shared_results = self._benchmark_shared_files(test_files, worker_count)
                
                # Store results
                config_key = f"files_{file_count}_workers_{worker_count}"
                results[config_key] = {
                    "file_count": file_count,
                    "worker_count": worker_count,
                    "queue_benchmark": queue_results,
                    "batch_benchmark": batch_results,
                    "shared_files_benchmark": shared_results
                }
                
                # Print mini summary for this configuration
                print(f"\nConfiguration summary: {file_count} files, {worker_count} workers")
                print(f"  Queue throughput: {queue_results['operations_per_second']:.1f} op/s")
                print(f"  Batch processing: {batch_results['operations_per_second']:.1f} op/s")
                print(f"  Shared files: {shared_results['operations_per_second']:.1f} op/s")
            except Exception as e:
                print(f"ERROR during benchmark execution: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing with next configuration...")
        
        if not results:
            print("WARNING: No benchmark configurations completed successfully!")
        else:
            print(f"Completed {len(results)}/{len(configs)} benchmark configurations")
        
        self.results = results
        return results
    
    def _generate_test_images(self, count: int) -> List[Path]:
        """
        Generate test images for benchmarking with enhanced error handling.
        
        Args:
            count: Number of images to generate
            
        Returns:
            List of paths to generated images
        """
        print(f"Generating {count} test images...")
        
        # Ensure benchmark data directory exists with explicit error handling
        from benchmarks.utils import DATA_DIR
        benchmark_dir = DATA_DIR / "multithreaded"
        
        try:
            benchmark_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created or verified directory: {benchmark_dir.absolute()}")
        except Exception as e:
            print(f"ERROR: Failed to create directory {benchmark_dir.absolute()}: {e}")
            raise
        
        if not benchmark_dir.exists():
            print(f"ERROR: Directory {benchmark_dir.absolute()} doesn't exist after creation attempt")
            raise RuntimeError(f"Directory creation failed: {benchmark_dir.absolute()}")
        
        # Generate images with consistent metadata
        paths = []
        for i in range(count):
            path = benchmark_dir / f"test_image_{i}.png"
            absolute_path = path.absolute()
            
            print(f"Processing image {i+1}/{count}: {absolute_path}")
            
            # Only create if it doesn't exist
            if not path.exists():
                try:
                    metadata = BenchmarkUtils.create_complex_metadata("medium")
                    # Properly extract width and height from self.image_size tuple
                    width, height = self.image_size
                    BenchmarkUtils.create_test_image(path, width, height, metadata)
                    
                    # Verify file was created
                    if not path.exists():
                        print(f"WARNING: File {absolute_path} was not created by BenchmarkUtils.create_test_image")
                    else:
                        print(f"Successfully created file: {absolute_path} ({path.stat().st_size} bytes)")
                except Exception as e:
                    print(f"ERROR creating test image {absolute_path}: {e}")
                    # Create a fallback simple image if the utility fails
                    try:
                        print(f"Attempting fallback image creation for {absolute_path}")
                        from PIL import Image
                        import numpy as np
                        img = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
                        img.save(path)
                        print(f"Fallback creation succeeded for {absolute_path}")
                    except Exception as fallback_error:
                        print(f"CRITICAL: Even fallback creation failed: {fallback_error}")
            else:
                print(f"File already exists: {absolute_path} ({path.stat().st_size} bytes)")
            
            # Always use absolute paths to avoid path resolution issues
            paths.append(absolute_path)
        
        # Final verification
        missing_files = [str(p) for p in paths if not p.exists()]
        if missing_files:
            print(f"WARNING: {len(missing_files)} files are missing after generation:")
            for file in missing_files[:5]:  # Show first 5 missing files
                print(f"  - {file}")
            if len(missing_files) > 5:
                print(f"  - ...and {len(missing_files) - 5} more")
        else:
            print(f"All {len(paths)} test files verified successfully")
        
        return paths
    
    def _reset_test_files(self, test_files: List[Path]) -> None:
        """
        Reset test files to a clean state between benchmark runs with enhanced error handling.
        
        Args:
            test_files: List of test file paths
        """
        print(f"Resetting {len(test_files)} test files...")
        
        # Verify all files exist first and report any missing files
        missing_files = [str(path) for path in test_files if not path.exists()]
        if missing_files:
            print(f"WARNING: {len(missing_files)} files don't exist:")
            for file in missing_files[:5]:  # Show first 5 missing files
                print(f"  - {file}")
            if len(missing_files) > 5:
                print(f"  - ...and {len(missing_files) - 5} more")
        
        # Process only existing files
        processed = 0
        errors = 0
        for path in test_files:
            if path.exists():
                try:
                    # Create handler and clear metadata
                    handler = PNGMetadataHandlerBase.create(path)
                    handler.clear_metadata()
                    
                    # Add a single metadata field for consistency
                    handler.update_metadata("benchmark_initialized", "true")
                    processed += 1
                except Exception as e:
                    print(f"ERROR resetting {path}: {e}")
                    errors += 1
        
        print(f"Reset complete: {processed} files processed, {errors} errors, {len(missing_files)} files missing")
        
        # If all files are missing, this is a critical error that should stop the benchmark
        if len(missing_files) == len(test_files):
            raise RuntimeError(f"All {len(test_files)} test files are missing. Cannot proceed with benchmark.")
    
    def _benchmark_queue(self, test_files: List[Path], worker_count: int) -> Dict[str, Any]:
        """
        Benchmark the PNGMetadataQueue with various workloads and enhanced error handling.
        
        Args:
            test_files: List of test files to use
            worker_count: Number of worker threads
            
        Returns:
            Dictionary with benchmark results
        """
        # Verify files before starting
        valid_files = [path for path in test_files if path.exists()]
        if len(valid_files) < len(test_files):
            print(f"WARNING: Only {len(valid_files)}/{len(test_files)} files exist for queue benchmark")
            if not valid_files:
                print("ERROR: No valid files for queue benchmark")
                return {
                    "total_operations": 0,
                    "total_time": 0.001,  # Avoid division by zero
                    "operations_per_second": 0,
                    "per_priority_times": {},
                    "error": "No valid files available"
                }
            # Continue with only valid files
            test_files = valid_files
            
        print(f"Running queue benchmark with {len(test_files)} files and {worker_count} workers...")
        
        results = {
            "total_operations": 0,
            "total_time": 0,
            "operations_per_second": 0,
            "per_priority_times": {}
        }
        
        # Initialize queue
        queue = PNGMetadataQueue(num_workers=worker_count)
        queue.start()
        
        try:
            # Track timings for different priority levels
            priority_times = {0: [], 5: [], 10: []}
            
            # Create a mix of priorities
            priorities = [0] * 7 + [5] * 2 + [10] * 1  # 70% low, 20% medium, 10% high
            
            # Start timing
            start_time = time.perf_counter()
            
            # Queue operations with different priorities
            operations_queued = 0
            for path in test_files:
                # Double-check file existence right before queuing
                if not path.exists():
                    print(f"WARNING: File {path} disappeared during queue benchmark")
                    continue
                    
                for i in range(self.operations_per_file):
                    priority = random.choice(priorities)
                    key = f"queue_benchmark_{i}_{priority}"
                    value = f"value_{time.time()}"
                    
                    # Track individual operation timing
                    op_start = time.perf_counter()
                    try:
                        queue.update_metadata(path, key, value, priority=priority)
                        op_time = time.perf_counter() - op_start
                        
                        # Store time by priority
                        priority_times[priority].append(op_time * 1000)  # Convert to ms
                        operations_queued += 1
                    except Exception as e:
                        print(f"ERROR queuing operation for {path}: {e}")
            
            if operations_queued == 0:
                print("ERROR: No operations were successfully queued")
                return {
                    "total_operations": 0,
                    "total_time": 0.001,
                    "operations_per_second": 0,
                    "per_priority_times": {},
                    "error": "No operations queued"
                }
                
            print(f"Queued {operations_queued} operations. Waiting for completion...")
            
            # Wait for all operations to complete
            queue.stop(wait=True)
            
            # Calculate total time
            total_time = time.perf_counter() - start_time
            
            # Store results
            results["total_operations"] = operations_queued
            results["total_time"] = total_time
            results["operations_per_second"] = operations_queued / total_time if total_time > 0 else 0
            
            # Calculate statistics for each priority
            for priority, times in priority_times.items():
                if times:
                    results["per_priority_times"][str(priority)] = {
                        "mean_ms": statistics.mean(times),
                        "median_ms": statistics.median(times),
                        "min_ms": min(times),
                        "max_ms": max(times),
                        "std_ms": statistics.stdev(times) if len(times) > 1 else 0
                    }
            
            print(f"Queue benchmark complete: {operations_queued} operations in {total_time:.2f}s "
                  f"({results['operations_per_second']:.1f} op/s)")
        
        except Exception as e:
            print(f"ERROR during queue benchmark: {e}")
            import traceback
            traceback.print_exc()
            results["error"] = str(e)
        
        finally:
            # Ensure queue is stopped
            if queue.running:
                queue.stop(wait=True)
        
        return results
    
    def _benchmark_batch_editor(self, test_files: List[Path], worker_count: int) -> Dict[str, Any]:
        """
        Benchmark the BatchEditor with various workloads.
        
        Args:
            test_files: List of test files to use
            worker_count: Number of worker threads
            
        Returns:
            Dictionary with benchmark results
        """
        results = {
            "total_operations": 0,
            "total_time": 0,
            "operations_per_second": 0,
            "queue_vs_direct": {}
        }
        
        # Benchmark with both queue and direct modes
        for use_queue in [True, False]:
            mode = "queue" if use_queue else "direct"
            
            # Start timing
            start_time = time.perf_counter()
            
            # Use BatchEditor
            with BatchEditor(workers=worker_count, use_queue=use_queue) as batch:
                # Create a mix of operations:
                # 70% individual updates, 20% batch updates, 10% removes
                for i, path in enumerate(test_files):
                    op_type = i % 10
                    
                    if op_type < 7:  # 70% individual updates
                        batch.update(path, {
                            f"batch_key_{i}": f"batch_value_{i}",
                            f"timestamp_{i}": str(time.time())
                        })
                    elif op_type < 9:  # 20% batch updates (multiple files)
                        # Update this file and the next one (if available)
                        batch_files = [path]
                        if i+1 < len(test_files):
                            batch_files.append(test_files[i+1])
                        
                        batch.update_many(batch_files, {
                            "batch_shared": f"shared_value_{i}",
                            "timestamp_shared": str(time.time())
                        })
                    else:  # 10% removes
                        batch.remove(path, [f"batch_key_{i-1}", f"non_existent_key"])
            
            # Calculate total time
            total_time = time.perf_counter() - start_time
            total_ops = len(test_files)  # One operation per file (approximately)
            
            # Store results for this mode
            results["queue_vs_direct"][mode] = {
                "total_time": total_time,
                "operations_per_second": total_ops / total_time if total_time > 0 else 0
            }
        
        # Calculate overall metrics
        queue_ops = results["queue_vs_direct"]["queue"]["operations_per_second"]
        direct_ops = results["queue_vs_direct"]["direct"]["operations_per_second"]
        
        results["total_operations"] = len(test_files)
        results["total_time"] = min(results["queue_vs_direct"]["queue"]["total_time"],
                                  results["queue_vs_direct"]["direct"]["total_time"])
        results["operations_per_second"] = max(queue_ops, direct_ops)
        results["queue_vs_direct"]["efficiency_ratio"] = queue_ops / direct_ops if direct_ops > 0 else 0
        
        return results

    def _prepare_scenario_files(self, test_files: List[Path], scenario: str) -> List[List[Path]]:
        """
        Prepare file groups for different contention scenarios.
        
        Args:
            test_files: List of test files
            scenario: Contention scenario ("low", "medium", "high")
            
        Returns:
            List of file groups, with distribution appropriate to the scenario
        """
        if not test_files:
            return []
            
        # Create copies to avoid modifying the original
        files = test_files.copy()
        
        # Ensure we have enough files for fair distribution
        while len(files) < 10:  # Ensure at least 10 files for good distribution
            files.extend(files[:10-len(files)])
        
        # Create file groups based on scenario
        file_groups = []
        
        if scenario == "low":
            # Low contention: Each group gets mostly distinct files
            # Create N groups with minimal overlap
            group_count = min(len(files) // 2, 5)  # Up to 5 groups
            for i in range(group_count):
                start = i * len(files) // group_count
                end = (i + 1) * len(files) // group_count
                file_groups.append(files[start:end])
                
        elif scenario == "medium":
            # Medium contention: Some overlap between groups
            # Create groups with ~50% overlap
            group_count = min(len(files) // 2, 5)  # Up to 5 groups
            stride = len(files) // (group_count * 2)  # 50% overlap
            for i in range(group_count):
                start = i * stride
                end = start + stride * 2  # Double size to create overlap
                file_groups.append(files[start:min(end, len(files))])
                
        elif scenario == "high":
            # High contention: All groups work on the same small set of files
            contended_file_count = min(3, len(files))  # Use at most 3 files for high contention
            contended_files = files[:contended_file_count]
            # Create multiple identical groups
            for _ in range(5):  # Fixed number of groups for high contention
                file_groups.append(contended_files)
        
        return file_groups

    def _benchmark_shared_files(self, test_files: List[Path], worker_count: int) -> Dict[str, Any]:
        """
        Benchmark concurrent access to shared files to measure lock contention.
        Uses PNGMetadataQueue for proper thread coordination.
        
        Args:
            test_files: List of test files to use
            worker_count: Number of worker threads
            
        Returns:
            Dictionary with benchmark results
        """
        results = {
            "total_operations": 0,
            "total_time": 0,
            "operations_per_second": 0,
            "contention_scenarios": {}
        }
        
        # Test different contention scenarios
        for scenario in ["low", "medium", "high"]:
            print(f"Running {scenario} contention scenario...")
            
            # Create a queue with the specified worker count
            queue = PNGMetadataQueue(num_workers=worker_count)
            queue.start()
            
            try:
                # Prepare work distribution based on scenario
                file_groups = self._prepare_scenario_files(test_files, scenario)
                
                # Record start time
                start_time = time.perf_counter()
                
                # Queue operations with appropriate distribution
                operations_queued = 0
                for file_group in file_groups:
                    for file_path in file_group:
                        if not file_path.exists():
                            print(f"WARNING: File {file_path} not found, skipping")
                            continue
                            
                        # Queue operations appropriate to the scenario
                        operations_queued += self._queue_scenario_operations(queue, file_path, scenario)
                
                # Wait for completion with proper progress reporting
                print(f"Queued {operations_queued} operations for {scenario} scenario. Waiting for completion...")
                queue.stop(wait=True)
                
                # Calculate timing
                total_time = time.perf_counter() - start_time
                ops_per_second = operations_queued / total_time if total_time > 0 else 0
                
                # Store results
                results["contention_scenarios"][scenario] = {
                    "total_operations": operations_queued,
                    "total_time": total_time,
                    "operations_per_second": ops_per_second
                }
                
                print(f"{scenario.capitalize()} contention: {operations_queued} operations in {total_time:.2f}s "
                      f"({ops_per_second:.1f} op/s)")
                      
            except Exception as e:
                print(f"ERROR during {scenario} contention benchmark: {e}")
                import traceback
                traceback.print_exc()
                
                # Store error result
                results["contention_scenarios"][scenario] = {
                    "total_operations": 0,
                    "total_time": 0.001,
                    "operations_per_second": 0,
                    "error": str(e)
                }
                
            finally:
                # Ensure queue is stopped
                if queue and queue.running:
                    queue.stop(wait=True)
        
        # Calculate overall metrics across all scenarios
        scenario_ops = [s["operations_per_second"] for s in results["contention_scenarios"].values()]
        if scenario_ops:
            results["operations_per_second"] = statistics.mean(scenario_ops)
            results["total_operations"] = sum(s["total_operations"] for s in results["contention_scenarios"].values())
            results["total_time"] = sum(s["total_time"] for s in results["contention_scenarios"].values())
        
        return results

    def _queue_scenario_operations(self, queue: PNGMetadataQueue, file_path: Path, scenario: str) -> int:
        """
        Queue operations appropriate to the given contention scenario.
        
        Args:
            queue: PNGMetadataQueue to use
            file_path: Path to the file to operate on
            scenario: Contention scenario ("low", "medium", "high")
            
        Returns:
            Number of operations queued
        """
        operations_queued = 0
        
        try:
            if scenario == "low":
                # Simple updates with minimal reads
                for i in range(3):
                    queue.update_metadata(file_path, f"low_contention_{i}", f"value_{time.time()}")
                    operations_queued += 1
                    
            elif scenario == "medium":
                # Initial read to simulate read-modify-write pattern
                metadata = queue.get_metadata(file_path)
                operations_queued += 1
                
                # Update based on read values
                for i in range(2):
                    key = f"medium_contention_{i}"
                    old_value = metadata.get(key, "0")
                    new_value = f"{float(old_value) + 1.0}" if old_value.replace('.', '', 1).isdigit() else "1.0"
                    queue.update_metadata(file_path, key, new_value)
                    operations_queued += 1
                    
            elif scenario == "high":
                # Multiple read-modify-write cycles on same keys
                # This creates highest contention as each thread repeatedly accesses the same keys
                metadata = queue.get_metadata(file_path)
                operations_queued += 1
                
                # Use dependent updates that rely on previous state
                key = "high_contention_counter"
                counter = metadata.get(key, "0")
                counter_val = int(counter) if counter.isdigit() else 0
                
                # Multiple updates to same key to create contention
                for i in range(4):
                    counter_val += 1
                    queue.update_metadata(file_path, key, str(counter_val))
                    operations_queued += 1
                    
        except Exception as e:
            print(f"Error queuing operations for {file_path} in {scenario} scenario: {e}")
        
        return operations_queued

    def _create_work_batches(self, test_files: List[Path], worker_count: int, 
                           scenario: str) -> List[List[Path]]:
        """
        Create work batches for threads based on contention scenario.
        
        Args:
            test_files: List of test files
            worker_count: Number of worker threads
            scenario: Contention scenario ("low", "medium", "high")
            
        Returns:
            List of file batches, one per worker
        """
        if not test_files:
            return []
            
        # Create copies to avoid modifying the original
        files = test_files.copy()
        
        # Ensure we have at least as many files as workers for fair distribution
        while len(files) < worker_count:
            files.extend(files[:worker_count-len(files)])
        
        # Create batches based on scenario
        batches = []
        
        if scenario == "low":
            # Each worker gets distinct files
            files_per_worker = len(files) // worker_count
            for i in range(worker_count):
                start = i * files_per_worker
                end = start + files_per_worker if i < worker_count - 1 else len(files)
                batches.append(files[start:end])
                
        elif scenario == "medium":
            # 50% overlap between workers
            files_per_worker = len(files) // (worker_count // 2 + 1)
            for i in range(worker_count):
                start = (i * files_per_worker) // 2  # 50% overlap
                end = min(start + files_per_worker, len(files))
                batches.append(files[start:end])
                
        elif scenario == "high":
            # All workers operate on the same small set of files
            contended_file_count = min(3, len(files))  # Use at most 3 files for high contention
            contended_files = files[:contended_file_count]
            for _ in range(worker_count):
                batches.append(contended_files)
                
        return batches
    
    def _process_file_batch(self, files: List[Path], scenario: str) -> int:
        """
        Process a batch of files, simulating real workloads with mixed operations.
        
        Args:
            files: List of files to process
            scenario: Contention scenario for specialized behavior
            
        Returns:
            Number of operations completed
        """
        operations_completed = 0
        
        # Create a handler for each file
        for file_path in files:
            try:
                handler = PNGMetadataHandlerBase.create(file_path)
                
                # Perform operations based on scenario
                if scenario == "low":
                    # Simple updates with minimal reads
                    for i in range(3):
                        handler.update_metadata(f"low_contention_{i}", f"value_{time.time()}")
                        operations_completed += 1
                        
                elif scenario == "medium":
                    # Mix of reads and writes
                    current = handler.get_metadata()
                    operations_completed += 1
                    
                    # Update based on current values
                    for i in range(2):
                        key = f"medium_contention_{i}"
                        old_value = current.get(key, "0")
                        new_value = f"{float(old_value) + 1.0}" if old_value.replace('.', '', 1).isdigit() else "1.0"
                        handler.update_metadata(key, new_value)
                        operations_completed += 1
                        
                elif scenario == "high":
                    # Lots of read-modify-write cycles on same keys
                    for i in range(5):  # More operations for high contention
                        # Read current value
                        current = handler.get_metadata()
                        operations_completed += 1
                        
                        # Update counter
                        key = "high_contention_counter"
                        counter = current.get(key, "0")
                        counter_val = int(counter) if counter.isdigit() else 0
                        handler.update_metadata(key, str(counter_val + 1))
                        operations_completed += 1
                        
                        # Simulate some work
                        time.sleep(0.001)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return operations_completed
    
    def save_results(self, format_type: str = "all") -> Dict[str, Path]:
        """
        Save benchmark results in various formats.
        
        Args:
            format_type: Output format, one of "json", "csv", "markdown", or "all"
            
        Returns:
            Dictionary mapping format types to output file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        if format_type in ("json", "all"):
            json_path = RESULTS_DIR / f"multithreaded_benchmark_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            saved_files["json"] = json_path
            
        if format_type in ("csv", "all"):
            csv_path = RESULTS_DIR / f"multithreaded_benchmark_{timestamp}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow([
                    'Configuration', 'Files', 'Workers',
                    'Queue Ops/sec', 'Batch Ops/sec', 'Shared Files Ops/sec',
                    'Low Contention Ops/sec', 'Medium Contention Ops/sec', 'High Contention Ops/sec',
                    'Queue/Direct Efficiency'
                ])
                
                # Write data rows
                for config, data in self.results.items():
                    writer.writerow([
                        config, 
                        data['file_count'], 
                        data['worker_count'],
                        data['queue_benchmark']['operations_per_second'],
                        data['batch_benchmark']['operations_per_second'],
                        data['shared_files_benchmark']['operations_per_second'],
                        data['shared_files_benchmark']['contention_scenarios']['low']['operations_per_second'],
                        data['shared_files_benchmark']['contention_scenarios']['medium']['operations_per_second'],
                        data['shared_files_benchmark']['contention_scenarios']['high']['operations_per_second'],
                        data['batch_benchmark']['queue_vs_direct'].get('efficiency_ratio', 0)
                    ])
            
            saved_files["csv"] = csv_path
            
        if format_type in ("markdown", "all"):
            # Create comprehensive markdown report
            md_path = RESULTS_DIR / f"multithreaded_benchmark_{timestamp}.md"
            with open(md_path, 'w') as f:
                f.write("# PNG Metadata Multithreaded Benchmark\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Queue performance section
                f.write("## Queue Performance\n\n")
                f.write("| Files | Workers | Operations/second | Low Priority (ms) | Medium Priority (ms) | High Priority (ms) |\n")
                f.write("|-------|---------|-------------------|-------------------|----------------------|--------------------|\n")
                
                for config, data in self.results.items():
                    queue_data = data['queue_benchmark']
                    priority_data = queue_data.get('per_priority_times', {})
                    
                    # Extract mean values for each priority
                    low = priority_data.get('0', {}).get('mean_ms', 0)
                    med = priority_data.get('5', {}).get('mean_ms', 0)
                    high = priority_data.get('10', {}).get('mean_ms', 0)
                    
                    f.write(f"| {data['file_count']} | {data['worker_count']} | "
                          f"{queue_data['operations_per_second']:.1f} | "
                          f"{low:.1f} | {med:.1f} | {high:.1f} |\n")
                
                # Batch editor section
                f.write("\n## Batch Editor Performance\n\n")
                f.write("| Files | Workers | Queue Mode (ops/s) | Direct Mode (ops/s) | Efficiency Ratio |\n")
                f.write("|-------|---------|-------------------|---------------------|------------------|\n")
                
                for config, data in self.results.items():
                    batch_data = data['batch_benchmark']
                    queue_vs_direct = batch_data.get('queue_vs_direct', {})
                    queue_ops = queue_vs_direct.get('queue', {}).get('operations_per_second', 0)
                    direct_ops = queue_vs_direct.get('direct', {}).get('operations_per_second', 0)
                    ratio = queue_vs_direct.get('efficiency_ratio', 0)
                    
                    f.write(f"| {data['file_count']} | {data['worker_count']} | "
                          f"{queue_ops:.1f} | {direct_ops:.1f} | {ratio:.2f} |\n")
                
                # Contention scenarios section
                f.write("\n## Contention Scenarios Performance\n\n")
                f.write("| Files | Workers | Low Contention (ops/s) | Medium Contention (ops/s) | High Contention (ops/s) |\n")
                f.write("|-------|---------|------------------------|---------------------------|-------------------------|\n")
                
                for config, data in self.results.items():
                    shared_data = data['shared_files_benchmark']
                    contention = shared_data.get('contention_scenarios', {})
                    
                    low_ops = contention.get('low', {}).get('operations_per_second', 0)
                    med_ops = contention.get('medium', {}).get('operations_per_second', 0)
                    high_ops = contention.get('high', {}).get('operations_per_second', 0)
                    
                    f.write(f"| {data['file_count']} | {data['worker_count']} | "
                          f"{low_ops:.1f} | {med_ops:.1f} | {high_ops:.1f} |\n")
                
                # Summary section
                f.write("\n## Performance Summary\n\n")
                f.write("| Files | Workers | Overall Ops/sec | Queue Ops/sec | Batch Ops/sec | Shared Files Ops/sec |\n")
                f.write("|-------|---------|-----------------|---------------|---------------|----------------------|\n")
                
                for config, data in self.results.items():
                    queue_ops = data['queue_benchmark']['operations_per_second']
                    batch_ops = data['batch_benchmark']['operations_per_second']
                    shared_ops = data['shared_files_benchmark']['operations_per_second']
                    overall = (queue_ops + batch_ops + shared_ops) / 3
                    
                    f.write(f"| {data['file_count']} | {data['worker_count']} | "
                          f"{overall:.1f} | {queue_ops:.1f} | {batch_ops:.1f} | {shared_ops:.1f} |\n")
            
            saved_files["markdown"] = md_path
        
        # Print summary to console
        print("\nMultithreaded Benchmark Summary:")
        print("-" * 87)
        print(f"{'Config':<15} {'Queue (op/s)':<15} {'Batch (op/s)':<15} {'Low Cont.':<12} {'Med Cont.':<12} {'High Cont.':<12}")
        print("-" * 87)
        
        for config, data in sorted(self.results.items()):
            queue_ops = data['queue_benchmark']['operations_per_second']
            batch_ops = data['batch_benchmark']['operations_per_second']
            
            contention = data['shared_files_benchmark']['contention_scenarios']
            low_ops = contention['low']['operations_per_second']
            med_ops = contention['medium']['operations_per_second']
            high_ops = contention['high']['operations_per_second']
            
            print(f"{config:<15} "
                  f"{queue_ops:>8.1f}      "
                  f"{batch_ops:>8.1f}      "
                  f"{low_ops:>8.1f}   "
                  f"{med_ops:>8.1f}   "
                  f"{high_ops:>8.1f}")
                  
        return saved_files