"""
Memory usage benchmarks for PNG metadata operations.

This module provides memory usage testing to compare different metadata 
handler implementations across various image sizes, with a focus on
measuring peak memory consumption.
"""

import time
import json
import csv
import gc
import tracemalloc
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Import from parent package
from png_metadata_tools.chunk_handler import PNGMetadataHandler
from png_metadata_tools.streaming_chunk_handler import StreamingPNGMetadataHandler

# Import local modules
from benchmarks.utils import BenchmarkUtils, RESULTS_DIR


class MemoryBenchmark:
    """
    Benchmark for measuring memory usage of different handler implementations.
    
    This benchmark tracks peak memory consumption when updating metadata using
    different implementations: PIL, standard chunk handler, and streaming chunk handler.
    """
    
    def __init__(self, sizes: List[Tuple[int, int]], iterations: int = 5):
        """
        Initialize the memory benchmark.
        
        Args:
            sizes: List of (width, height) tuples defining image sizes
            iterations: Number of iterations for each test
        """
        self.sizes = sizes
        self.iterations = iterations
        self.results = {}
        
    def run(self):
        """
        Run the memory benchmark and return the results.
        
        Returns:
            Dictionary containing benchmark results
        """
        # Generate test images
        images = BenchmarkUtils.generate_test_images(self.sizes)
        
        # Define handlers to benchmark
        handlers = {
            "PIL": self._run_pil_memory,
            "Standard": self._run_standard_memory,
            "Streaming": self._run_streaming_memory
        }
        
        results = {}
        
        print("\nRunning memory benchmark...")
        try:
            from tqdm import tqdm
            progress_tracker = tqdm
        except ImportError:
            # Simple progress tracker if tqdm is not available
            def progress_tracker(iterable, **kwargs):
                print(f"Processing {len(iterable)} image sizes...")
                return iterable
        
        for size_tuple, image_path in progress_tracker(images.items()):
            width, height = size_tuple
            megapixels = (width * height) / 1_000_000
            file_size = image_path.stat().st_size / (1024 * 1024)  # MB
            
            size_results = {
                "megapixels": megapixels,
                "file_size": file_size,
                "width": width,
                "height": height,
                "handlers": {}
            }
            
            # Run benchmark for each handler
            for name, handler_func in handlers.items():
                # Run memory profiling
                memory_stats = []
                for _ in range(self.iterations):
                    stats = self._profile_memory_usage(handler_func, image_path)
                    memory_stats.append(stats)
                
                # Calculate average memory usage
                peak_values = [stats["peak_mb"] for stats in memory_stats]
                current_values = [stats["current_mb"] for stats in memory_stats]
                
                # Calculate statistics
                peak_mean = statistics.mean(peak_values)
                current_mean = statistics.mean(current_values)
                
                try:
                    peak_stdev = statistics.stdev(peak_values)
                    current_stdev = statistics.stdev(current_values)
                except:
                    # Fallback if only one iteration
                    peak_stdev = 0.0
                    current_stdev = 0.0
                
                size_results["handlers"][name] = {
                    "peak_mb": peak_mean,
                    "peak_stdev_mb": peak_stdev,
                    "current_mb": current_mean,
                    "current_stdev_mb": current_stdev,
                    "raw_data": memory_stats
                }
            
            # Calculate memory ratios
            pil_peak = size_results["handlers"]["PIL"]["peak_mb"]
            for name in ["Standard", "Streaming"]:
                if name in size_results["handlers"]:
                    handler_peak = size_results["handlers"][name]["peak_mb"]
                    size_results["handlers"][name]["memory_ratio"] = handler_peak / pil_peak
            
            results[f"{width}x{height}"] = size_results
        
        self.results = results
        return results
    
    def _profile_memory_usage(self, func: callable, *args, **kwargs) -> Dict[str, float]:
        """
        Profile memory usage of a function using tracemalloc.
        
        Args:
            func: Function to profile
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Dictionary with memory usage statistics
        """
        # Force garbage collection
        gc.collect()
        
        # Start tracking memory allocations
        tracemalloc.start()
        
        try:
            # Call the function
            func(*args, **kwargs)
            
            # Get memory statistics
            current, peak = tracemalloc.get_traced_memory()
        finally:
            # Stop tracking even if an exception occurs
            tracemalloc.stop()
        
        # Convert to MB
        current_mb = current / 1024 / 1024
        peak_mb = peak / 1024 / 1024
        
        return {
            "current_mb": current_mb,
            "peak_mb": peak_mb
        }
    
    def _run_pil_memory(self, image_path: Path) -> None:
        """Run a memory test using PIL."""
        with Image.open(image_path) as img:
            metadata = PngInfo()
            for key, value in img.info.items():
                if isinstance(value, str):
                    metadata.add_text(key, value)
            metadata.add_text('benchmark', str(time.time()))
            img.save(image_path, pnginfo=metadata)
    
    def _run_standard_memory(self, image_path: Path) -> None:
        """Run a memory test using the standard handler."""
        handler = PNGMetadataHandler(image_path)
        handler.update_metadata('benchmark', str(time.time()))
    
    def _run_streaming_memory(self, image_path: Path) -> None:
        """Run a memory test using the streaming handler."""
        handler = StreamingPNGMetadataHandler(image_path)
        handler.update_metadata('benchmark', str(time.time()))
    
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
            json_path = RESULTS_DIR / f"memory_benchmark_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            saved_files["json"] = json_path
            
        if format_type in ("csv", "all"):
            csv_path = RESULTS_DIR / f"memory_benchmark_{timestamp}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow([
                    'Size', 'Width', 'Height', 'Megapixels', 'File Size (MB)',
                    'PIL Peak (MB)', 'PIL Std (MB)', 
                    'Standard Peak (MB)', 'Standard Std (MB)', 'Standard Memory Ratio',
                    'Streaming Peak (MB)', 'Streaming Std (MB)', 'Streaming Memory Ratio'
                ])
                
                # Write data
                for size_name, data in self.results.items():
                    pil_data = data["handlers"]["PIL"]
                    standard_data = data["handlers"]["Standard"]
                    streaming_data = data["handlers"]["Streaming"]
                    
                    writer.writerow([
                        size_name, data["width"], data["height"], 
                        data["megapixels"], data["file_size"],
                        pil_data["peak_mb"], pil_data["peak_stdev_mb"],
                        standard_data["peak_mb"], standard_data["peak_stdev_mb"], standard_data["memory_ratio"],
                        streaming_data["peak_mb"], streaming_data["peak_stdev_mb"], streaming_data["memory_ratio"]
                    ])
            
            saved_files["csv"] = csv_path
            
        if format_type in ("markdown", "all"):
            # Simple text-based markdown without fancy formatting
            md_path = RESULTS_DIR / f"memory_benchmark_{timestamp}.md"
            with open(md_path, 'w') as f:
                f.write("# PNG Metadata Memory Usage Benchmark\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Summary table in simple markdown format
                f.write("## Summary\n\n")
                f.write("| Size | Megapixels | File Size (MB) | PIL Peak (MB) | Standard Peak (MB) | Streaming Peak (MB) | Standard Ratio | Streaming Ratio |\n")
                f.write("|------|------------|---------------|---------------|-------------------|-------------------|---------------|----------------|\n")
                
                for size_name, data in self.results.items():
                    pil_data = data["handlers"]["PIL"]
                    standard_data = data["handlers"]["Standard"]
                    streaming_data = data["handlers"]["Streaming"]
                    
                    f.write(f"| {size_name} | {data['megapixels']:.1f} | {data['file_size']:.1f} | "
                          f"{pil_data['peak_mb']:.2f} ± {pil_data['peak_stdev_mb']:.2f} | "
                          f"{standard_data['peak_mb']:.2f} ± {standard_data['peak_stdev_mb']:.2f} | "
                          f"{streaming_data['peak_mb']:.2f} ± {streaming_data['peak_stdev_mb']:.2f} | "
                          f"{standard_data['memory_ratio']:.2f}x | {streaming_data['memory_ratio']:.2f}x |\n")
                
                f.write("\n")
            
            saved_files["markdown"] = md_path
        
        # Print summary to console
        print("\nMemory Usage Benchmark Summary:")
        print("-" * 85)
        print(f"{'Size':<10} {'PIL (MB)':<15} {'Standard (MB)':<17} {'Streaming (MB)':<17} {'Std Ratio':<10} {'Str Ratio':<10}")
        print("-" * 85)
        
        # Sort by size for display
        for size_name in sorted(self.results.keys(), key=lambda x: int(x.split('x')[0])):
            data = self.results[size_name]
            pil_data = data["handlers"]["PIL"]
            standard_data = data["handlers"]["Standard"]
            streaming_data = data["handlers"]["Streaming"]
            
            print(f"{size_name:<10} "
                  f"{pil_data['peak_mb']:>6.2f} ± {pil_data['peak_stdev_mb']:>5.2f} "
                  f"{standard_data['peak_mb']:>6.2f} ± {standard_data['peak_stdev_mb']:>5.2f} "
                  f"{streaming_data['peak_mb']:>6.2f} ± {streaming_data['peak_stdev_mb']:>5.2f} "
                  f"{standard_data['memory_ratio']:>8.2f}x {streaming_data['memory_ratio']:>8.2f}x")
                  
        return saved_files