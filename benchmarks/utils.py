"""
Utilities for benchmark preparation and execution.

This module provides shared functionality for generating test images,
managing benchmark directories, and creating complex metadata.
"""

import os
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

# Constants
BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
RESULTS_DIR = BENCHMARK_DIR / "results"


class BenchmarkUtils:
    """Utilities for benchmark preparation and execution."""
    
    @staticmethod
    def ensure_dirs():
        """Ensure all necessary directories exist with enhanced error handling."""
        dirs_to_create = [
            DATA_DIR,
            RESULTS_DIR,
            DATA_DIR / "multithreaded"
        ]
        
        for directory in dirs_to_create:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"Directory verified: {directory.absolute()}")
                
                # Test write permissions by creating a test file
                test_file = directory / ".dir_test"
                test_file.write_text("Directory write test")
                test_file.unlink()  # Clean up test file
                
            except Exception as e:
                print(f"ERROR: Failed to create or verify directory {directory}: {e}")
                raise RuntimeError(f"Directory creation/verification failed for {directory}: {e}")
        
    @staticmethod
    def create_test_image(path: Path, width: int, height: int, metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Create a test image with optional metadata and enhanced error handling.
        
        Args:
            path: Path to save the image
            width: Image width in pixels
            height: Image height in pixels
            metadata: Optional metadata to add to the image
            
        Returns:
            Path to the created image
        """
        # Ensure parent directory exists
        parent_dir = path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Import libraries for image creation
            from PIL import Image, PngImagePlugin
            import numpy as np
            
            # Create simple test image data
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Create a gradient pattern for more realistic file sizes
            y, x = np.mgrid[0:height, 0:width]
            img_array[..., 0] = np.sin(x/width * 3.14) * 127 + 128  # Red channel
            img_array[..., 1] = np.sin(y/height * 3.14) * 127 + 128  # Green channel
            img_array[..., 2] = np.sin((x+y)/(width+height) * 3.14) * 127 + 128  # Blue channel
            
            # Convert to PIL Image
            img = Image.fromarray(img_array)
            
            # Add metadata if provided
            if metadata:
                png_info = PngImagePlugin.PngInfo()
                for key, value in metadata.items():
                    if isinstance(value, (dict, list)):
                        png_info.add_text(key, json.dumps(value))
                    else:
                        png_info.add_text(key, str(value))
                img.save(path, "PNG", pnginfo=png_info)
            else:
                img.save(path, "PNG")
            
            # Verify file was created successfully
            if not path.exists():
                raise RuntimeError(f"Failed to create image at {path} - file does not exist after save")
            
            # Log file size for debugging
            file_size = path.stat().st_size
            if file_size < 100:  # Suspiciously small file
                print(f"WARNING: Created image at {path} is very small ({file_size} bytes)")
                
            return path
                
        except ImportError as e:
            error_msg = f"Error: Required libraries missing: {e}"
            print(error_msg)
            raise
        except Exception as e:
            error_msg = f"Error creating test image at {path}: {e}"
            print(error_msg)
            
            # Create an absolute minimal fallback image as last resort
            try:
                print(f"Attempting absolute minimal image creation for {path}")
                img = Image.new('RGB', (width, height), color=(128, 128, 128))
                img.save(path, 'PNG')
                
                if not path.exists():
                    raise RuntimeError("File still does not exist after fallback creation")
                    
                return path
            except Exception as fallback_e:
                print(f"CRITICAL: Fallback image creation also failed: {fallback_e}")
                raise RuntimeError(f"Failed to create test image at {path}: {e}. Fallback also failed.") from e
    
    @staticmethod
    def create_complex_metadata(size: str = "small") -> Dict[str, Any]:
        """
        Create complex metadata of different sizes for testing.
        
        Args:
            size: Size of metadata to create ("small", "medium", or "large")
            
        Returns:
            Dictionary containing metadata
        """
        if size == "small":
            return {
                "title": "Test Image",
                "author": "British Standards",
                "rating": "1500.0",
                "tags": "test,benchmark,performance",
                "created": datetime.now().isoformat()
            }
        elif size == "medium":
            # Medium-sized metadata similar to ComfyUI parameters
            workflow = {
                "nodes": {
                    "1": {
                        "inputs": {
                            "seed": random.randint(100000, 999999),
                            "steps": 20,
                            "cfg": 7.5,
                            "sampler_name": "euler_a",
                            "scheduler": "normal",
                            "denoise": 0.75,
                        },
                        "class_type": "KSampler",
                    },
                    "2": {
                        "inputs": {
                            "ckpt_name": "v1-5-pruned.ckpt",
                            "vae_name": "vae-ft-mse-840000-ema-pruned.ckpt",
                        },
                        "class_type": "CheckpointLoaderSimple",
                    },
                },
                "extra": {
                    "version": "v1.3.0",
                    "timestamp": datetime.now().isoformat(),
                },
            }
            
            base_metadata = BenchmarkUtils.create_complex_metadata("small")
            base_metadata["workflow"] = json.dumps(workflow)
            base_metadata["positive_prompt"] = "masterpiece, best quality, ultra realistic"
            base_metadata["negative_prompt"] = "worst quality, low quality, bad anatomy"
            
            return base_metadata
            
        elif size == "large":
            # Large metadata similar to a complex configuration
            base_metadata = BenchmarkUtils.create_complex_metadata("medium")
            
            # Add a lot more fields
            for i in range(50):
                base_metadata[f"extra_field_{i}"] = f"value_{i}" * 10
                
            # Add a complex JSON structure
            config = {
                "model_settings": {
                    "weights": [random.random() for _ in range(100)],
                    "biases": [random.random() for _ in range(100)],
                    "layers": [
                        {"id": i, "type": "linear", "size": 512, "activation": "relu"}
                        for i in range(10)
                    ],
                    "optimizers": {
                        "adam": {"lr": 0.001, "beta1": 0.9, "beta2": 0.999},
                        "sgd": {"lr": 0.01, "momentum": 0.9}
                    }
                },
                "training_history": [
                    {"epoch": i, "loss": random.random(), "accuracy": random.random()}
                    for i in range(100)
                ]
            }
            
            base_metadata["config"] = json.dumps(config)
            
            return base_metadata
            
        return {}

    @staticmethod
    def generate_test_images(sizes: List[Tuple[int, int]], 
                           metadata_size: str = "medium") -> Dict[Tuple[int, int], Path]:
        """
        Generate test images of different sizes with appropriate metadata.
        
        Args:
            sizes: List of (width, height) tuples
            metadata_size: Size of metadata to add ("small", "medium", or "large")
            
        Returns:
            Dictionary mapping size tuples to image paths
        """
        try:
            from tqdm import tqdm
        except ImportError:
            # Fallback if tqdm is not available
            def tqdm(iterable, **kwargs):
                return iterable
        
        BenchmarkUtils.ensure_dirs()
        images = {}
        
        metadata = BenchmarkUtils.create_complex_metadata(metadata_size)
        
        print("Generating test images...")
        for width, height in tqdm(sizes):
            filename = f"test_{width}x{height}.png"
            path = DATA_DIR / filename
            
            # Skip if already exists
            if path.exists():
                images[(width, height)] = path
                continue
                
            # Create new test image
            BenchmarkUtils.create_test_image(path, width, height, metadata)
            images[(width, height)] = path
            
        return images