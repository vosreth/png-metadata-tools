"""
Unified Test Suite for PNG Metadata Handlers with proper British standards
"""
import pytest
from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json
import shutil
import numpy as np
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import random

from png_metadata_tools.base_handler import PNGMetadataHandlerBase
from png_metadata_tools.chunk_handler import PNGMetadataHandler
from png_metadata_tools.streaming_chunk_handler import StreamingPNGMetadataHandler

# Test data paths
TEST_ROOT = Path(__file__).parent
TEST_DATA = TEST_ROOT / "data"
TEST_TEMP = TEST_DATA / "temp" / "unified_tests"

# Define handler implementations to test
HANDLER_IMPLEMENTATIONS = [
    PNGMetadataHandler,
    StreamingPNGMetadataHandler
]

def get_implementation_name(impl):
    """Get a readable name for the implementation."""
    return impl.__name__

class TestPNGMetadataHandlers:
    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Setup test environment"""
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        yield
        # Cleanup
        for file in TEST_TEMP.glob("*"):
            file.unlink(missing_ok=True)

    def _create_test_image(self, path: Path, width: int = 512, height: int = 512, metadata: dict = None) -> None:
        """Create a test image with optional metadata"""
        # Create random test image data
        size = (width, height)
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

    def _create_large_test_image(self, path: Path, width: int = 2048, height: int = 2048, metadata: dict = None) -> None:
        """Create a large test image using a more memory-efficient approach"""
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a new image with PIL
        with Image.new('RGB', (width, height), (255, 255, 255)) as img:
            # Add some visual elements to make it a proper test
            for i in range(0, width, 256):
                for j in range(0, height, 256):
                    color = ((i * 13) % 256, (j * 17) % 256, ((i+j) * 7) % 256)
                    img.paste(Image.new('RGB', (128, 128), color), (i, j))
            
            # Add metadata if provided
            if metadata:
                png_info = PngInfo()
                for key, value in metadata.items():
                    if isinstance(value, (dict, list)):
                        png_info.add_text(key, json.dumps(value))
                    else:
                        png_info.add_text(key, str(value))
                img.save(path, 'PNG', pnginfo=png_info, optimize=True)
            else:
                img.save(path, 'PNG', optimize=True)

    @pytest.mark.parametrize("handler_class", HANDLER_IMPLEMENTATIONS, ids=get_implementation_name)
    def test_basic_metadata_operations(self, handler_class):
        """Test basic metadata read/write operations"""
        test_path = TEST_TEMP / f"basic_test_{handler_class.__name__}.png"
        self._create_test_image(test_path)
        
        # Test writing metadata
        handler = handler_class(test_path)
        handler.update_metadata('rating', '1500.0')
        
        # Verify with PIL
        with Image.open(test_path) as img:
            assert 'rating' in img.text
            assert img.text['rating'] == '1500.0'
            
        # Test reading
        handler = handler_class(test_path)
        metadata = handler.get_metadata()
        assert 'rating' in metadata
        assert metadata['rating'] == '1500.0'

    @pytest.mark.parametrize("handler_class", HANDLER_IMPLEMENTATIONS, ids=get_implementation_name)
    def test_metadata_updates(self, handler_class):
        """Test multiple metadata updates"""
        test_path = TEST_TEMP / f"update_test_{handler_class.__name__}.png"
        self._create_test_image(test_path)
        
        handler = handler_class(test_path)
        
        # Multiple updates
        for rating in [1500.0, 1600.0, 1700.0]:
            handler.update_metadata('rating', str(rating))
            metadata = handler.get_metadata()
            assert 'rating' in metadata
            assert float(metadata['rating']) == rating
            
            # Verify image is still valid after each update
            with Image.open(test_path) as img:
                assert img.verify() is None  # Raises if invalid

    @pytest.mark.parametrize("handler_class", HANDLER_IMPLEMENTATIONS, ids=get_implementation_name)
    def test_data_integrity(self, handler_class):
        """Test that image data remains unchanged after metadata updates"""
        test_path = TEST_TEMP / f"integrity_test_{handler_class.__name__}.png"
        self._create_test_image(test_path)
        
        # Get original image data
        with Image.open(test_path) as img:
            original_data = img.tobytes()
            original_size = img.size
        
        # Perform multiple metadata updates
        handler = handler_class(test_path)
        for rating in [1500.0, 1600.0, 1700.0]:
            handler.update_metadata('rating', str(rating))
            
            # Verify image data
            with Image.open(test_path) as img:
                assert img.tobytes() == original_data
                assert img.size == original_size

    @pytest.mark.parametrize("handler_class", HANDLER_IMPLEMENTATIONS, ids=get_implementation_name)
    def test_error_handling(self, handler_class):
        """Test error handling for invalid files and operations"""
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            handler_class(TEST_TEMP / f"nonexistent_{handler_class.__name__}.png")
        
        # Test with non-PNG file
        invalid_path = TEST_TEMP / f"invalid_{handler_class.__name__}.txt"
        invalid_path.write_bytes(b"Not a PNG file")
        with pytest.raises(ValueError):
            handler_class(invalid_path)
            
        # Test with truncated PNG file
        truncated_path = TEST_TEMP / f"truncated_{handler_class.__name__}.png"
        # Create a valid PNG first
        self._create_test_image(truncated_path)
        # Now truncate it
        with open(truncated_path, 'rb') as f:
            valid_start = f.read(100)  # Just enough to pass signature check
        with open(truncated_path, 'wb') as f:
            f.write(valid_start)
            
        # This should handle the truncated file gracefully
        handler = handler_class(truncated_path)
        # Reading metadata should return empty dict for truncated file
        assert handler.get_metadata() == {}

    @pytest.mark.parametrize("handler_class", HANDLER_IMPLEMENTATIONS, ids=get_implementation_name)
    def test_atomic_updates(self, handler_class):
        """Test that updates are atomic and don't corrupt file"""
        test_path = TEST_TEMP / f"atomic_test_{handler_class.__name__}.png"
        self._create_test_image(test_path)
        
        # Create backup
        backup_path = TEST_TEMP / f"backup_{handler_class.__name__}.png"
        shutil.copy(test_path, backup_path)
        
        handler = handler_class(test_path)
        
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

    @pytest.mark.parametrize("handler_class", HANDLER_IMPLEMENTATIONS, ids=get_implementation_name)
    def test_comfyui_metadata_preservation(self, handler_class):
        """Test that ComfyUI metadata is preserved during updates"""
        test_path = TEST_TEMP / f"comfy_test_{handler_class.__name__}.png"
        
        # Create test image with ComfyUI metadata
        metadata = {
            'workflow': {"test": "data"},
            'prompt': {"another": "test"}
        }
        self._create_test_image(test_path, metadata=metadata)
        
        # Update with handler
        handler = handler_class(test_path)
        handler.update_metadata('rating', '1500.0')
        
        # Verify all metadata
        with Image.open(test_path) as img:
            assert json.loads(img.text['workflow']) == metadata['workflow']
            assert json.loads(img.text['prompt']) == metadata['prompt']
            assert img.text['rating'] == '1500.0'

    @pytest.mark.parametrize("handler_class", HANDLER_IMPLEMENTATIONS, ids=get_implementation_name)
    def test_remove_metadata(self, handler_class):
        """Test removing metadata"""
        test_path = TEST_TEMP / f"remove_test_{handler_class.__name__}.png"
        
        # Create test image with metadata
        metadata = {
            'title': 'Test Image',
            'author': 'British Standards',
            'rating': '1500.0'
        }
        self._create_test_image(test_path, metadata=metadata)
        
        # Verify metadata exists
        with Image.open(test_path) as img:
            assert 'title' in img.text
            assert 'rating' in img.text
            
        # Remove one key
        handler = handler_class(test_path)
        result = handler.remove_metadata('rating')
        
        # Verify removal worked
        assert result is True
        
        # Verify with PIL
        with Image.open(test_path) as img:
            assert 'title' in img.text  # Still exists
            assert 'rating' not in img.text  # Removed
            
        # Try removing non-existent key
        result = handler.remove_metadata('nonexistent')
        assert result is False  # Should return False if key didn't exist

    @pytest.mark.parametrize("handler_class", HANDLER_IMPLEMENTATIONS, ids=get_implementation_name)
    def test_clear_metadata(self, handler_class):
        """Test clearing all metadata"""
        test_path = TEST_TEMP / f"clear_test_{handler_class.__name__}.png"
        
        # Create test image with metadata
        metadata = {
            'title': 'Test Image',
            'author': 'British Standards',
            'rating': '1500.0',
            'tags': 'test,metadata,clear'
        }
        self._create_test_image(test_path, metadata=metadata)
        
        # Verify metadata exists
        with Image.open(test_path) as img:
            assert len(img.text) >= 4
            
        # Clear all metadata
        handler = handler_class(test_path)
        handler.clear_metadata()
        
        # Verify all metadata was removed
        with Image.open(test_path) as img:
            assert not img.text  # Should be empty
            
        # Verify image is still valid
        with Image.open(test_path) as img:
            assert img.verify() is None  # Raises if invalid

    @pytest.mark.parametrize("handler_class", HANDLER_IMPLEMENTATIONS, ids=get_implementation_name)
    def test_get_metadata_keys(self, handler_class):
        """Test getting all metadata keys"""
        test_path = TEST_TEMP / f"keys_test_{handler_class.__name__}.png"
        
        # Create test image with metadata
        metadata = {
            'title': 'Test Image',
            'author': 'British Standards',
            'rating': '1500.0'
        }
        self._create_test_image(test_path, metadata=metadata)
        
        # Get keys
        handler = handler_class(test_path)
        keys = handler.get_metadata_keys()
        
        # Verify all keys are present
        assert set(keys) == set(metadata.keys())
        
        # Test with empty metadata
        empty_path = TEST_TEMP / f"empty_keys_{handler_class.__name__}.png"
        self._create_test_image(empty_path)
        
        handler = handler_class(empty_path)
        keys = handler.get_metadata_keys()
        assert len(keys) == 0

    @pytest.mark.parametrize("handler_class", HANDLER_IMPLEMENTATIONS, ids=get_implementation_name)
    def test_has_metadata_key(self, handler_class):
        """Test checking for specific metadata keys"""
        test_path = TEST_TEMP / f"has_key_test_{handler_class.__name__}.png"
        
        # Create test image with metadata
        metadata = {
            'title': 'Test Image',
            'rating': '1500.0'
        }
        self._create_test_image(test_path, metadata=metadata)
        
        # Check keys
        handler = handler_class(test_path)
        
        assert handler.has_metadata_key('title') is True
        assert handler.has_metadata_key('rating') is True
        assert handler.has_metadata_key('nonexistent') is False

    @pytest.mark.parametrize("handler_class", HANDLER_IMPLEMENTATIONS, ids=get_implementation_name)
    def test_large_file_performance(self, handler_class):
        """Test performance with larger files (basic timing, not memory profiling)"""
        # Generate large image (4MP)
        large_test_path = TEST_TEMP / f"large_perf_{handler_class.__name__}.png"
        self._create_large_test_image(large_test_path, width=2048, height=2048)
        
        # Measure time for update operation
        start_time = time.time()
        handler = handler_class(large_test_path)
        handler.update_metadata('test_key', 'test_value')
        operation_time = time.time() - start_time
        
        # Report performance
        file_size = large_test_path.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"{handler_class.__name__}: Image size: 2048x2048, File size: {file_size:.2f} MB, Update time: {operation_time*1000:.2f}ms")
        
        # No specific assertion, just collecting performance data
        # In a real test suite, you might compare against a baseline or track metrics

    @pytest.mark.parametrize("handler_class", HANDLER_IMPLEMENTATIONS, ids=get_implementation_name)
    def test_factory_method(self, handler_class):
        """Test the factory method auto-selection"""
        test_path = TEST_TEMP / f"factory_test_{handler_class.__name__}.png"
        self._create_test_image(test_path)
        
        # Force specific implementation
        handler = PNGMetadataHandlerBase.create(test_path, use_streaming=(handler_class == StreamingPNGMetadataHandler))
        
        # Verify correct implementation was chosen
        assert isinstance(handler, handler_class)
        
        # Test basic functionality to ensure factory method works
        handler.update_metadata('factory_test', 'value')
        metadata = handler.get_metadata()
        assert metadata['factory_test'] == 'value'

class TestUnifiedAPI:
    """Tests for the unified API functions in the package namespace"""
    
    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Setup test environment"""
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        yield
        # Improved cleanup with proper directory handling
        for path in TEST_TEMP.glob("*"):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                try:
                    shutil.rmtree(path)
                except (PermissionError, OSError):
                    print(f"Warning: Could not remove directory {path}")
                
    def _create_test_image(self, path: Path, metadata: dict = None) -> None:
        """Create a test image with optional metadata"""
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
            
    def test_simplified_api(self):
        """Test the simplified API functions"""
        import png_metadata_tools as pngmeta
        
        # Create test image
        test_path = TEST_TEMP / "api_test.png"
        self._create_test_image(test_path)
        
        # Test write and read
        pngmeta.write(test_path, {'name': 'Test', 'value': 42})
        metadata = pngmeta.read(test_path)
        
        assert metadata['name'] == 'Test'
        assert metadata['value'] == '42'
        
        # Test update
        pngmeta.update(test_path, 'updated', True)
        assert pngmeta.has_key(test_path, 'updated')
        
        # Test remove
        pngmeta.remove(test_path, 'name')
        assert not pngmeta.has_key(test_path, 'name')
        
        # Test clear
        pngmeta.clear(test_path)
        metadata = pngmeta.read(test_path)
        assert len(metadata) == 0
        
    def test_meta_editor(self):
        """Test the MetaEditor context manager"""
        import png_metadata_tools as pngmeta
        
        # Create test image
        test_path = TEST_TEMP / "editor_test.png"
        self._create_test_image(test_path)
        
        # Use context manager
        with pngmeta.MetaEditor(test_path) as meta:
            meta['rating'] = 1500.0
            meta['processed'] = True
            meta['author'] = 'British Standards'
            
        # Verify changes were applied
        metadata = pngmeta.read(test_path)
        assert float(metadata['rating']) == 1500.0
        assert metadata['processed'] == 'True'
        assert metadata['author'] == 'British Standards'
        
    def test_batch_editor(self):
        """Test the BatchEditor context manager"""
        import png_metadata_tools as pngmeta
        
        # Create multiple test images
        test_paths = []
        for i in range(5):
            path = TEST_TEMP / f"batch_test_{i}.png"
            self._create_test_image(path)
            test_paths.append(path)
            
        # Use batch editor
        with pngmeta.BatchEditor(workers=2) as batch:
            for path in test_paths:
                batch.update(path, {
                    'batch_processed': True,
                    'timestamp': '2025-02-25'
                })
                
        # Verify all images were updated
        for path in test_paths:
            metadata = pngmeta.read(path)
            assert metadata['batch_processed'] == 'True'
            assert metadata['timestamp'] == '2025-02-25'
            
    def test_queue_system(self):
        """Test the Queue system"""
        import png_metadata_tools as pngmeta
        
        # Create test image
        test_path = TEST_TEMP / "queue_test.png"
        self._create_test_image(test_path)
        
        # Use queue
        queue = pngmeta.Queue(workers=2)
        queue.start()
        
        queue.update(test_path, 'priority_high', 'important', priority=10)
        queue.update(test_path, 'priority_low', 'standard', priority=0)
        
        # Get results
        metadata = queue.get(test_path)
        
        # Verify updates were applied
        assert metadata['priority_high'] == 'important'
        assert metadata['priority_low'] == 'standard'
        
        # Clean up
        queue.stop()