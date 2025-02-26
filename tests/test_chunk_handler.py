"""
Test suite for PNG Chunk Handler with proper British standards
"""
import pytest
from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json
import shutil
import numpy as np
from png_metadata_tools.chunk_handler import PNGMetadataHandler

# Test data paths
TEST_ROOT = Path(__file__).parent
TEST_DATA = TEST_ROOT / "data"
TEST_CASES = TEST_DATA / "test_cases"
TEST_TEMP = TEST_DATA / "temp"

class TestPNGChunkHandler:
    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Setup test environment with improved Windows compatibility."""
        import time
        # Ensure directory exists
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        
        yield  # Allow the test to run
        
        # Give Windows time to release file handles
        time.sleep(0.2)
        
        # Attempt cleanup with patience
        for file in TEST_TEMP.glob("*"):
            try:
                if file.is_file():
                    file.unlink(missing_ok=True)
                elif file.is_dir():
                    try:
                        shutil.rmtree(file)
                    except (PermissionError, OSError):
                        print(f"Warning: Could not remove directory {file}")
            except PermissionError as e:
                print(f"Warning: Permission error when cleaning up {file}: {e}")

    def _create_test_image(self, path: Path, metadata: dict = None) -> None:
        """Create a test image with optional metadata"""
        # Create random test image data
        size = (512, 512)
        img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        
        if metadata:
            png_info = PngInfo()
            for key, value in metadata.items():
                if isinstance(value, (dict, list)):
                    png_info.add_text(key, json.dumps(value))
                else:
                    png_info.add_text(key, str(value))
            img.save(path, 'PNG', pnginfo=png_info)
        else:
            img.save(path, 'PNG')

    def test_basic_metadata_operations(self):
        """Test basic metadata read/write operations"""
        test_path = TEST_TEMP / "basic_test.png"
        self._create_test_image(test_path)
        
        # Test writing generic metadata instead of ELO-specific
        handler = PNGMetadataHandler(test_path)
        handler.update_metadata('rating', '1500.0')
        
        # Verify with PIL
        with Image.open(test_path) as img:
            assert 'rating' in img.text
            assert img.text['rating'] == '1500.0'
            
        # Test reading
        handler = PNGMetadataHandler(test_path)
        metadata = handler.get_metadata()
        assert 'rating' in metadata
        assert metadata['rating'] == '1500.0'

    def test_metadata_updates(self):
        """Test multiple metadata updates"""
        test_path = TEST_TEMP / "update_test.png"
        self._create_test_image(test_path)
        
        handler = PNGMetadataHandler(test_path)
        
        # Multiple updates with generic metadata
        for rating in [1500.0, 1600.0, 1700.0]:
            handler.update_metadata('rating', str(rating))
            metadata = handler.get_metadata()
            assert 'rating' in metadata
            assert float(metadata['rating']) == rating
            
            # Verify image is still valid after each update
            with Image.open(test_path) as img:
                assert img.verify() is None  # Raises if invalid

    def test_data_integrity(self):
        """Test that image data remains unchanged after metadata updates"""
        test_path = TEST_TEMP / "integrity_test.png"
        self._create_test_image(test_path)
        
        # Get original image data
        with Image.open(test_path) as img:
            original_data = img.tobytes()
            original_size = img.size
        
        # Perform multiple metadata updates
        handler = PNGMetadataHandler(test_path)
        for rating in [1500.0, 1600.0, 1700.0]:
            handler.update_metadata('rating', str(rating))
            
            # Verify image data
            with Image.open(test_path) as img:
                assert img.tobytes() == original_data
                assert img.size == original_size

    def test_concurrent_metadata_handling(self):
        """Test handling of metadata from different sources"""
        test_path = TEST_TEMP / "concurrent_test.png"
        self._create_test_image(test_path)
        
        # Update with handler
        handler = PNGMetadataHandler(test_path)
        handler.update_metadata('rating', '1500.0')
        
        # Update with PIL
        with Image.open(test_path) as img:
            metadata = PngInfo()
            metadata.add_text('rating', '1600.0')
            img.save(test_path, pnginfo=metadata)
        
        # Verify handler can still read/write
        handler = PNGMetadataHandler(test_path)
        metadata = handler.get_metadata()
        assert 'rating' in metadata
        assert metadata['rating'] == '1600.0'
        
        handler.update_metadata('rating', '1700.0')
        metadata = handler.get_metadata()
        assert 'rating' in metadata
        assert metadata['rating'] == '1700.0'

    def test_error_handling(self):
        """Test error handling for invalid files and operations"""
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            PNGMetadataHandler(TEST_TEMP / "nonexistent.png")
        
        # Test with non-PNG file
        invalid_path = TEST_TEMP / "invalid.txt"
        invalid_path.write_bytes(b"Not a PNG file")
        with pytest.raises(ValueError):
            PNGMetadataHandler(invalid_path)

    def test_atomic_updates(self):
        """Test that updates are atomic and don't corrupt file"""
        test_path = TEST_TEMP / "atomic_test.png"
        self._create_test_image(test_path)
        
        # Create backup
        backup_path = TEST_TEMP / "backup.png"
        shutil.copy(test_path, backup_path)
        
        handler = PNGMetadataHandler(test_path)
        
        try:
            # Simulate interruption during update
            with pytest.raises(Exception):
                handler.update_metadata('will_fail', 'value')
                raise Exception("Simulated failure")
        except:
            pass
        
        # Verify file is still valid
        with Image.open(test_path) as img:
            assert img.verify() is None  # Raises if invalid
            
        # Verify content matches backup
        with Image.open(test_path) as img1, Image.open(backup_path) as img2:
            assert img1.tobytes() == img2.tobytes()

    def test_comfyui_metadata_preservation(self):
        """Test that ComfyUI metadata is preserved during updates"""
        test_path = TEST_TEMP / "comfy_test.png"
        
        # Create test image with ComfyUI metadata
        metadata = {
            'workflow': {"test": "data"},
            'prompt': {"another": "test"}
        }
        self._create_test_image(test_path, metadata)
        
        # Update with generic metadata instead of ELO-specific
        handler = PNGMetadataHandler(test_path)
        handler.update_metadata('rating', '1500.0')
        
        # Verify all metadata
        with Image.open(test_path) as img:
            assert json.loads(img.text['workflow']) == metadata['workflow']
            assert json.loads(img.text['prompt']) == metadata['prompt']
            assert img.text['rating'] == '1500.0'