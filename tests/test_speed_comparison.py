"""
Extended speed comparison with proper test image management and larger sizes
"""
import pytest
import time
import shutil
from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json
import numpy as np
from png_metadata_tools.chunk_handler import PNGMetadataHandler

TEST_ROOT = Path(__file__).parent
TEST_DATA = TEST_ROOT / "data"
TEST_TEMP = TEST_DATA / "temp"

def create_realistic_test_image(width: int, height: int) -> np.ndarray:
    """
    Creates a more realistic test image with proper texture and complexity.
    """
    # Create base noise layers
    fine_noise = np.random.normal(0, 1, (height, width, 3))
    coarse_noise = np.random.normal(0, 1, (height//8, width//8, 3))
    
    # Upscale coarse noise
    coarse_upscaled = np.kron(coarse_noise, np.ones((8, 8, 1)))
    if coarse_upscaled.shape != (height, width, 3):
        coarse_upscaled = coarse_upscaled[:height, :width, :]
    
    # Create gradients
    y, x = np.mgrid[0:height, 0:width]
    gradient_x = x / width
    gradient_y = y / height
    
    # Combine layers with varying weights
    combined = (
        gradient_x[..., np.newaxis] * 0.3 +
        gradient_y[..., np.newaxis] * 0.3 +
        fine_noise * 0.2 +
        coarse_upscaled * 0.2
    )
    
    # Normalize and convert to uint8
    combined = (combined - combined.min()) / (combined.max() - combined.min())
    image_data = (combined * 255).astype(np.uint8)
    
    # Add some color variation
    image_data[..., 0] = np.clip(image_data[..., 0] * 1.2, 0, 255)  # Boost red
    image_data[..., 2] = np.clip(image_data[..., 2] * 0.8, 0, 255)  # Reduce blue
    
    return image_data

def export_speed_results(results):
    """Export speed comparison results to CSV for later visualization."""
    import csv
    from datetime import datetime
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"speed_comparison_{timestamp}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Image Size (MP)', 'File Size (MB)', 
                        'Width', 'Height',
                        'PIL Mean (ms)', 'PIL Std (ms)', 
                        'Chunk Mean (ms)', 'Chunk Std (ms)',
                        'Speedup Factor'])
        
        # Write data rows
        for result in results:
            writer.writerow([
                result['megapixels'],
                result['file_size'],
                result['width'],
                result['height'],
                result['pil_mean'] * 1000,  # Convert to ms
                result['pil_std'] * 1000,
                result['chunk_mean'] * 1000,
                result['chunk_std'] * 1000,
                result['pil_mean'] / result['chunk_mean']
            ])
    
    print(f"Results exported to {filename}")
    return filename

class TestExtendedSpeedComparison:
    # Test sizes including some massive images
    SIZES = [
        (512, 512),       # 0.25 MP
        (1024, 1024),     # 1 MP
        (2048, 2048),     # 4 MP
        (3072, 3072),     # 9 MP
        (4096, 4096),     # 16 MP
        (5120, 5120),     # 25 MP
        (6144, 6144),     # 36 MP
    ]

    # Initialize results list here
    results = [] 

    @pytest.fixture(scope="class")
    def setup_test_env(self):
        """Setup test environment once for all test cases"""
        TEST_TEMP.mkdir(parents=True, exist_ok=True)

        self.__class__.results = []
    
        print("\nCreating test images:")
        for width, height in self.SIZES:
            # Create realistic image data
            image_data = create_realistic_test_image(width, height)
            img = Image.fromarray(image_data)
            
            # Add substantial ComfyUI-like metadata
            metadata = PngInfo()
            workflow = {
                "nodes": {
                    "1": {
                        "inputs": {
                            "seed": np.random.randint(1000000),
                            "steps": 20,
                            "cfg": 7.5,
                            "sampler_name": "euler_a",
                            "scheduler": "normal",
                            "denoise": 0.75,
                            "noise_offset": 0.1,
                            "clip_skip": 2,
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
                    "timestamp": "2024-02-24-07-36-00",
                    "generation_mode": "txt2img",
                },
            }
            metadata.add_text('workflow', json.dumps(workflow))
            
            # Add a complex prompt
            prompt = {
                "positive": ("masterpiece, best quality, highly detailed, " * 5 +
                           "ultra realistic, photographic, 8k, raw photo, " +
                           "unedited, professional photography, hyperrealistic " +
                           "documentary style, award winning photography"),
                "negative": ("worst quality, low quality, normal quality, " * 3 +
                           "lowres, bad anatomy, bad hands, text, error, " +
                           "missing fingers, cropped, jpeg artifacts")
            }
            metadata.add_text('prompt', json.dumps(prompt))
            
            filename = f"test_{width}x{height}.png"
            img.save(TEST_TEMP / filename, "PNG", pnginfo=metadata, optimize=True)
            
            file_size = (TEST_TEMP / filename).stat().st_size
            print(f"{width}x{height}: {file_size / (1024*1024):.1f} MB")
        
        yield

        # Export results after all tests complete
        export_speed_results(self.results)

        # Cleanup at end of all tests
        shutil.rmtree(TEST_TEMP)

    @pytest.fixture(scope="function")
    def test_image(self, request):
        """Provide fresh test image for each test case"""
        size = request.param
        width, height = size
        original = TEST_TEMP / f"test_{width}x{height}.png"
        backup = TEST_TEMP / f"test_{width}x{height}_backup.png"
        
        # Create backup if needed
        if not backup.exists():
            shutil.copy(original, backup)
        
        # Reset from backup
        shutil.copy(backup, original)
        yield original

    def _time_operation(self, func, iterations: int = 20) -> tuple[float, float, list[float]]:
        """Time an operation with detailed statistics"""
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            func()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return np.mean(times), np.std(times), times

    @pytest.mark.parametrize("test_image", SIZES, indirect=True)
    def test_extended_comparison(self, test_image, setup_test_env):
        """Extended speed comparison with detailed analysis"""
        width, height = test_image.stem.split('_')[1].split('x')
        width, height = int(width), int(height)
        megapixels = (width * height) / 1_000_000
        file_size = test_image.stat().st_size / (1024 * 1024)  # MB
        
        # Warm-up runs
        for _ in range(3):
            self._run_pil_update(test_image)
            self._run_chunk_update(test_image)
            shutil.copy(test_image.parent / f"{test_image.stem}_backup.png", test_image)
        
        # Time both methods
        pil_mean, pil_std, pil_times = self._time_operation(
            lambda: self._run_pil_update(test_image),
            iterations=10 if megapixels > 16 else 20  # Fewer iterations for very large images
        )
        shutil.copy(test_image.parent / f"{test_image.stem}_backup.png", test_image)
        
        chunk_mean, chunk_std, chunk_times = self._time_operation(
            lambda: self._run_chunk_update(test_image)
        )
        
        # Calculate percentiles
        pil_percentiles = np.percentile(pil_times, [25, 50, 75])
        chunk_percentiles = np.percentile(chunk_times, [25, 50, 75])
        
        # Print detailed results
        print(f"\nDetailed Speed Comparison for {width}x{height} image")
        print(f"Image size: {megapixels:.1f} MP")
        print(f"File size: {file_size:.1f} MB")
        print("\nPIL Method:")
        print(f"  Mean: {pil_mean*1000:.2f}ms ± {pil_std*1000:.2f}ms")
        print(f"  Median: {pil_percentiles[1]*1000:.2f}ms")
        print(f"  25th-75th percentile: {pil_percentiles[0]*1000:.2f}ms - {pil_percentiles[2]*1000:.2f}ms")
        print("\nChunk Method:")
        print(f"  Mean: {chunk_mean*1000:.2f}ms ± {chunk_std*1000:.2f}ms")
        print(f"  Median: {chunk_percentiles[1]*1000:.2f}ms")
        print(f"  25th-75th percentile: {chunk_percentiles[0]*1000:.2f}ms - {chunk_percentiles[2]*1000:.2f}ms")
        print(f"\nSpeedup: {pil_mean/chunk_mean:.1f}x faster")
        
        # Verify image still valid
        with Image.open(test_image) as img:
            assert img.verify() is None
        
        # For larger images, chunk method should be significantly faster
        if width >= 1024:
            assert chunk_mean < pil_mean, "Chunk method should be faster for large images"

        # Store results in a dict
        result = {
            'megapixels': megapixels,
            'file_size': file_size,
            'width': width,
            'height': height,
            'pil_mean': pil_mean,
            'pil_std': pil_std,
            'chunk_mean': chunk_mean,
            'chunk_std': chunk_std
        }
        
        self.results.append(result)

    def _run_pil_update(self, image_path: Path) -> None:
        with Image.open(image_path) as img:
            metadata = PngInfo()
            for key, value in img.info.items():
                if isinstance(value, str):
                    metadata.add_text(key, value)
            metadata.add_text('elo_rating', str(1500.0))
            img.save(image_path, pnginfo=metadata)

    def _run_chunk_update(self, image_path: Path) -> None:
        handler = PNGMetadataHandler(image_path)
        handler.update_elo_rating(1500.0)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])