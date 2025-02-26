"""
Test suite for PNG Metadata Batch Operations with proper British standards
"""
import pytest
from pathlib import Path
import time
import shutil
import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np

from png_metadata_tools.base_handler import PNGMetadataHandlerBase
from png_metadata_tools.batch_operations import (
    BatchEditor, BatchProcessor, 
    create_metadata_filter, create_update_operation, create_conditional_operation
)
import png_metadata_tools as pngmeta

# Test data paths
TEST_ROOT = Path(__file__).parent
TEST_DATA = TEST_ROOT / "data"
TEST_TEMP = TEST_DATA / "temp" / "batch_tests"

class TestBatchOperations:
    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Setup test environment with fresh test images"""
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        
        # Create test images
        for i in range(10):
            self._create_test_image(TEST_TEMP / f"test_{i}.png")
            
        yield
        
        # Cleanup
        if TEST_TEMP.exists():
            shutil.rmtree(TEST_TEMP)
            
    def _create_test_image(self, path: Path, metadata: dict = None) -> None:
        """Create a test image with optional metadata"""
        # Create test image data
        size = (256, 256)
        img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        
        # Add metadata if provided
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
    
    def test_basic_batch_editor(self):
        """Test basic batch editor operations"""
        # Define test paths
        test_paths = [TEST_TEMP / f"test_{i}.png" for i in range(5)]
        
        # Use batch editor (queue mode)
        with BatchEditor(workers=2, use_queue=True) as batch:
            for path in test_paths:
                batch.update(path, {
                    'batch_processed': True,
                    'timestamp': '2025-02-25'
                })
                
        # Verify all images were updated
        for path in test_paths:
            metadata = pngmeta.read(path)
            assert 'batch_processed' in metadata
            assert metadata['batch_processed'] == 'True'
            assert metadata['timestamp'] == '2025-02-25'
    
    def test_direct_batch_editor(self):
        """Test batch editor with direct operations (no queue)"""
        # Define test paths
        test_paths = [TEST_TEMP / f"test_{i}.png" for i in range(5, 10)]
        
        # Use batch editor (direct mode)
        with BatchEditor(workers=2, use_queue=False) as batch:
            for path in test_paths:
                batch.update(path, {
                    'direct_mode': 'enabled',
                    'timestamp': '2025-02-25'
                })
                
        # Verify all images were updated
        for path in test_paths:
            metadata = pngmeta.read(path)
            assert 'direct_mode' in metadata
            assert metadata['direct_mode'] == 'enabled'
            assert metadata['timestamp'] == '2025-02-25'
    
    def test_batch_remove_operation(self):
        """Test batch removal of metadata"""
        # Create images with existing metadata
        test_paths = []
        for i in range(3):
            path = TEST_TEMP / f"remove_test_{i}.png"
            self._create_test_image(path, {
                'key1': 'value1',
                'key2': 'value2',
                'key3': 'value3'
            })
            test_paths.append(path)
        
        # Use batch editor to remove keys
        with BatchEditor(workers=2) as batch:
            for path in test_paths:
                batch.remove(path, ['key1', 'key3'])
                
        # Verify keys were removed
        for path in test_paths:
            metadata = pngmeta.read(path)
            assert 'key1' not in metadata
            assert 'key2' in metadata  # Not removed
            assert 'key3' not in metadata
    
    def test_batch_clear_operation(self):
        """Test batch clearing of all metadata"""
        # Create images with existing metadata
        test_paths = []
        for i in range(3):
            path = TEST_TEMP / f"clear_test_{i}.png"
            self._create_test_image(path, {
                'key1': 'value1',
                'key2': 'value2',
                'key3': 'value3'
            })
            test_paths.append(path)
        
        # Use batch editor to clear metadata
        with BatchEditor(workers=2) as batch:
            for path in test_paths:
                batch.clear(path)
                
        # Verify all metadata was cleared
        for path in test_paths:
            metadata = pngmeta.read(path)
            assert len(metadata) == 0
    
    def test_batch_processor(self):
        """Test the advanced batch processor"""
        # Create test images with different metadata
        for i in range(3):
            path = TEST_TEMP / f"processor_test_{i}.png"
            self._create_test_image(path, {
                'category': 'test',
                'value': str(i * 100)
            })
        
        # Create a filter function
        filter_fn = create_metadata_filter(
            required_keys=['category'],
            value_conditions={
                'category': lambda v: v == 'test'
            }
        )
        
        # Create an operation that explicitly accepts batch and path
        def update_op(batch, path):
            batch.update(path, {
                'processed': True,
                'batch_id': 'test_batch'
            })
        
        # Process matching files with explicit pattern
        pattern = str(TEST_TEMP / "processor_test_*.png")
        processor = BatchProcessor(workers=2)
        results = processor.process(
            [pattern], 
            lambda path: update_op,
            filter_fn,
            recursive=False
        )
        
        # Verify results
        assert results['processed'] == 3
        
        # Verify files were updated
        for i in range(3):
            path = TEST_TEMP / f"processor_test_{i}.png"
            metadata = pngmeta.read(path)
            assert metadata['processed'] == 'True'
            assert metadata['batch_id'] == 'test_batch'
            # Original metadata should be preserved
            assert metadata['category'] == 'test'
            assert metadata['value'] == str(i * 100)
    
    def test_conditional_operations(self):
        """Test conditional batch operations"""
        # Create test images in the test directory
        for i in range(5):
            path = TEST_TEMP / f"conditional_test_{i}.png"
            self._create_test_image(path, {
                'value': str(i * 100)
            })
        
        # Create condition function
        def high_value_condition(path):
            handler = PNGMetadataHandlerBase.create(path)
            metadata = handler.get_metadata()
            return 'value' in metadata and int(metadata['value']) >= 200
        
        # Create true/false operations with explicit batch parameter
        def high_value_op(batch, path):
            batch.update(path, {'category': 'high'})
        
        def low_value_op(batch, path):
            batch.update(path, {'category': 'low'})
        
        # Create conditional operation
        def conditional_op(batch, path):
            if high_value_condition(path):
                high_value_op(batch, path)
            else:
                low_value_op(batch, path)
        
        # Process files with an explicit pattern
        pattern = str(TEST_TEMP / "conditional_test_*.png")
        processor = BatchProcessor(workers=2)
        
        # Use a lambda that matches the expected signature
        results = processor.process(
            [pattern],
            lambda path: conditional_op,
            recursive=False
        )
        
        # Verify results
        assert results['processed'] == 5
        
        # Verify conditional logic was applied
        for i in range(5):
            path = TEST_TEMP / f"conditional_test_{i}.png"
            metadata = pngmeta.read(path)
            if i >= 2:  # Value >= 200
                assert metadata['category'] == 'high'
            else:
                assert metadata['category'] == 'low'
    
    def test_update_many(self):
        """Test updating many files at once"""
        # Define test paths
        test_paths = [TEST_TEMP / f"test_{i}.png" for i in range(5)]
        
        # Use batch editor to update many
        with BatchEditor(workers=2) as batch:
            batch.update_many(test_paths, {
                'batch_type': 'many',
                'created': '2025-02-25'
            })
                
        # Verify all images were updated
        for path in test_paths:
            metadata = pngmeta.read(path)
            assert metadata['batch_type'] == 'many'
            assert metadata['created'] == '2025-02-25'
    
    def test_glob_pattern(self):
        """Test processing files with glob patterns"""
        # Create named test images
        for name in ['apple', 'banana', 'cherry', 'date', 'elderberry']:
            path = TEST_TEMP / f"{name}.png"
            self._create_test_image(path)
        
        # Define an operation function that works directly with the handler
        def set_fruit_category(path):
            fruit_name = path.stem
            # Don't use context manager for PNGMetadataHandlerBase
            handler = PNGMetadataHandlerBase.create(path)
            handler.update_metadata('category', 'fruit')
            handler.update_metadata('name', fruit_name)
            handler.update_metadata('length', str(len(fruit_name)))
        
        # Use batch editor with glob pattern - use string path
        pattern = str(TEST_TEMP / "*.png")
        with BatchEditor(workers=2, use_queue=False) as batch:
            batch.process_glob(pattern, set_fruit_category)
        
        # Verify all matching images were updated
        for name in ['apple', 'banana', 'cherry', 'date', 'elderberry']:
            path = TEST_TEMP / f"{name}.png"
            metadata = pngmeta.read(path)
            assert metadata['category'] == 'fruit'
            assert metadata['name'] == name
            assert metadata['length'] == str(len(name))
        
    def test_batch_editor_error_handling(self):
        """Test error handling in batch operations"""
        # Create test image
        valid_path = TEST_TEMP / "valid.png"
        self._create_test_image(valid_path)
        
        # Non-existent path
        invalid_path = TEST_TEMP / "nonexistent.png"
        
        # Use batch editor with both paths
        with BatchEditor(workers=2, use_queue=False) as batch:
            batch.update(valid_path, {'status': 'valid'})
            batch.update(invalid_path, {'status': 'invalid'})
            
        # Verify valid file was updated
        metadata = pngmeta.read(valid_path)
        assert metadata['status'] == 'valid'
        
        # Check that error was recorded
        assert len(batch.errors) == 1
        assert "nonexistent.png" in batch.errors[0][0]

# Run tests if script is executed directly
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])