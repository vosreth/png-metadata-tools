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

    @pytest.fixture(scope="class")
    def setup_test_env(self):
        """Setup test environment once for all test cases"""
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        
        print("\nCreating test images:")
        for width, height in self.SIZES:
            # Create gradient image
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    img_array[i, j] = [(i * 255) // height, (j * 255) // width, 128]
            
            img = Image.fromarray(img_array)
            
            # Add substantial ComfyUI-like metadata
            metadata = PngInfo()
            workflow = {
                "test_node": {
                    "inputs": {
                        "seed": 42,
                        "steps": 20,
                        "cfg": 7.5,
                        "sampler_name": "euler_a",
                        "scheduler": "normal",
                        "denoise": 0.75,
                    },
                    "class_type": "TestNode",
                    "additional_data": "x" * 1000  # Bulk up metadata
                }
            }
            metadata.add_text('workflow', json.dumps(workflow))
            metadata.add_text('prompt', json.dumps({
                "positive": "a detailed test prompt " * 20,
                "negative": "test negative prompt " * 10
            }))
            
            filename = f"test_{width}x{height}.png"
            img.save(TEST_TEMP / filename, pnginfo=metadata)
            
            file_size = (TEST_TEMP / filename).stat().st_size
            print(f"{width}x{height}: {file_size / (1024*1024):.1f} MB")
        
        yield
        
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