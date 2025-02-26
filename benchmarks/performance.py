"""
Performance benchmarks for PNG metadata operations.

This module provides straightforward performance testing to compare
different metadata handler implementations across various image sizes.
"""

import time
import json
import csv
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


class PerformanceBenchmark:
    """
    Benchmark for comparing performance of different handler implementations.
    
    This benchmark measures the time taken to update metadata using different
    implementations: PIL, standard chunk handler, and streaming chunk handler.
    """
    
    def __init__(self, sizes: List[Tuple[int, int]], iterations: int = 10):
        """
        Initialize the performance benchmark.
        
        Args:
            sizes: List of (width, height) tuples defining image sizes
            iterations: Number of iterations for each test
        """
        self.sizes = sizes
        self.iterations = iterations
        self.results = {}
        
    def run(self):
        """
        Run the performance benchmark and return the results.
        
        Returns:
            Dictionary containing benchmark results
        """
        # Generate test images
        images = BenchmarkUtils.generate_test_images(self.sizes)
        
        # Define handlers to benchmark
        handlers = {
            "PIL": self._run_pil_update,
            "Standard": self._run_standard_update,
            "Streaming": self._run_streaming_update
        }
        
        results = {}
        
        print("\nRunning performance benchmark...")
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, **kwargs):
                return iterable
        
        for size_tuple, image_path in tqdm(images.items()):
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
                # Warm-up run
                for _ in range(3):
                    handler_func(image_path)
                
                # Timed runs
                times = []
                for _ in range(self.iterations):
                    start_time = time.perf_counter()
                    handler_func(image_path)
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)  # Convert to ms
                
                # Calculate basic statistics using standard library
                times.sort()
                mean = sum(times) / len(times)
                try:
                    std = statistics.stdev(times)
                    median = statistics.median(times)
                except:
                    # Fallback if statistics module is not available
                    squared_diffs = [(t - mean) ** 2 for t in times]
                    std = (sum(squared_diffs) / len(squared_diffs)) ** 0.5
                    median = times[len(times) // 2]
                
                # Calculate simple percentiles
                p25_idx = len(times) // 4
                p75_idx = len(times) * 3 // 4
                percentiles = [times[p25_idx], times[p75_idx]]
                
                size_results["handlers"][name] = {
                    "mean_ms": mean,
                    "std_ms": std,
                    "median_ms": median,
                    "p25_ms": percentiles[0],
                    "p75_ms": percentiles[1],
                    "raw_times_ms": times
                }
            
            # Calculate speedups
            pil_mean = size_results["handlers"]["PIL"]["mean_ms"]
            for name in ["Standard", "Streaming"]:
                if name in size_results["handlers"]:
                    handler_mean = size_results["handlers"][name]["mean_ms"]
                    size_results["handlers"][name]["speedup"] = pil_mean / handler_mean
            
            results[f"{width}x{height}"] = size_results
        
        self.results = results
        return results
    
    def _run_pil_update(self, image_path: Path) -> None:
        """Run a metadata update using PIL."""
        with Image.open(image_path) as img:
            metadata = PngInfo()
            for key, value in img.info.items():
                if isinstance(value, str):
                    metadata.add_text(key, value)
            metadata.add_text('benchmark', str(time.time()))
            img.save(image_path, pnginfo=metadata)
    
    def _run_standard_update(self, image_path: Path) -> None:
        """Run a metadata update using the standard handler."""
        handler = PNGMetadataHandler(image_path)
        handler.update_metadata('benchmark', str(time.time()))
    
    def _run_streaming_update(self, image_path: Path) -> None:
        """Run a metadata update using the streaming handler."""
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
            json_path = RESULTS_DIR / f"performance_benchmark_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            saved_files["json"] = json_path
            
        if format_type in ("csv", "all"):
            csv_path = RESULTS_DIR / f"performance_benchmark_{timestamp}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow([
                    'Size', 'Width', 'Height', 'Megapixels', 'File Size (MB)',
                    'PIL Mean (ms)', 'PIL Std (ms)', 
                    'Standard Mean (ms)', 'Standard Std (ms)', 'Standard Speedup',
                    'Streaming Mean (ms)', 'Streaming Std (ms)', 'Streaming Speedup'
                ])
                
                # Write data
                for size_name, data in self.results.items():
                    pil_data = data["handlers"]["PIL"]
                    standard_data = data["handlers"]["Standard"]
                    streaming_data = data["handlers"]["Streaming"]
                    
                    writer.writerow([
                        size_name, data["width"], data["height"], 
                        data["megapixels"], data["file_size"],
                        pil_data["mean_ms"], pil_data["std_ms"],
                        standard_data["mean_ms"], standard_data["std_ms"], standard_data["speedup"],
                        streaming_data["mean_ms"], streaming_data["std_ms"], streaming_data["speedup"]
                    ])
            
            saved_files["csv"] = csv_path
            
        if format_type in ("markdown", "all"):
            # Simple text-based markdown without fancy formatting
            md_path = RESULTS_DIR / f"performance_benchmark_{timestamp}.md"
            with open(md_path, 'w') as f:
                f.write("# PNG Metadata Performance Benchmark\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Summary table in simple markdown format
                f.write("## Summary\n\n")
                f.write("| Size | Megapixels | File Size (MB) | PIL (ms) | Standard (ms) | Streaming (ms) | Standard Speedup | Streaming Speedup |\n")
                f.write("|------|------------|---------------|----------|---------------|----------------|-----------------|------------------|\n")
                
                for size_name, data in self.results.items():
                    pil_data = data["handlers"]["PIL"]
                    standard_data = data["handlers"]["Standard"]
                    streaming_data = data["handlers"]["Streaming"]
                    
                    f.write(f"| {size_name} | {data['megapixels']:.1f} | {data['file_size']:.1f} | "
                          f"{pil_data['mean_ms']:.2f} ± {pil_data['std_ms']:.2f} | "
                          f"{standard_data['mean_ms']:.2f} ± {standard_data['std_ms']:.2f} | "
                          f"{streaming_data['mean_ms']:.2f} ± {streaming_data['std_ms']:.2f} | "
                          f"{standard_data['speedup']:.1f}x | {streaming_data['speedup']:.1f}x |\n")
                
                f.write("\n")
            
            saved_files["markdown"] = md_path
        
        # Print summary to console
        print("\nPerformance Benchmark Summary:")
        print("-" * 80)
        print(f"{'Size':<10} {'PIL (ms)':<15} {'Standard (ms)':<15} {'Streaming (ms)':<15} {'Std Speedup':<12} {'Str Speedup':<12}")
        print("-" * 80)
        
        # Sort by size for display
        for size_name in sorted(self.results.keys(), key=lambda x: int(x.split('x')[0])):
            data = self.results[size_name]
            pil_data = data["handlers"]["PIL"]
            standard_data = data["handlers"]["Standard"]
            streaming_data = data["handlers"]["Streaming"]
            
            print(f"{size_name:<10} "
                  f"{pil_data['mean_ms']:>6.2f} ± {pil_data['std_ms']:>5.2f} "
                  f"{standard_data['mean_ms']:>6.2f} ± {standard_data['std_ms']:>5.2f} "
                  f"{streaming_data['mean_ms']:>6.2f} ± {streaming_data['std_ms']:>5.2f} "
                  f"{standard_data['speedup']:>10.1f}x {streaming_data['speedup']:>10.1f}x")
                  
        return saved_files