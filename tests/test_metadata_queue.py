"""
Test suite for PNG Metadata Queue with proper British standards
"""
import pytest
import time
import threading
from pathlib import Path
import random
import shutil
import concurrent.futures
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np

from png_metadata_tools.chunk_handler import PNGMetadataHandler
from png_metadata_tools.metadata_queue import PNGMetadataQueue

# Test data paths
TEST_ROOT = Path(__file__).parent
TEST_DATA = TEST_ROOT / "data"
TEST_TEMP = TEST_DATA / "temp" / "queue_tests"


class TestPNGMetadataQueue:
    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Setup test environment with fresh test images"""
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        
        # Create test images
        for i in range(10):
            self._create_test_image(TEST_TEMP / f"test_{i}.png")
            
        # Initialize queue
        self.queue = PNGMetadataQueue(num_workers=2)
        self.queue.start()
        
        yield
        
        # Cleanup
        self.queue.stop(wait=True)
        if TEST_TEMP.exists():
            shutil.rmtree(TEST_TEMP)
            
    def _create_test_image(self, path: Path, metadata: dict = None) -> None:
        """Create a test image with optional metadata"""
        # Create random test image data
        size = (256, 256)
        img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        
        # Add metadata if provided
        if metadata:
            png_info = PngInfo()
            for key, value in metadata.items():
                png_info.add_text(key, str(value))
            img.save(path, 'PNG', pnginfo=png_info)
        else:
            img.save(path, 'PNG')
            
    def test_basic_queue_operation(self):
        """Test basic queue operations"""
        test_path = TEST_TEMP / "test_0.png"
        
        # Queue a metadata update
        self.queue.update_metadata(test_path, "test_key", "test_value")
        
        # Wait for processing
        time.sleep(0.5)
        
        # Verify metadata was written
        result = self.queue.get_metadata(test_path)
        assert "test_key" in result
        assert result["test_key"] == "test_value"
        
    def test_multiple_updates(self):
        """Test multiple metadata updates to the same file"""
        test_path = TEST_TEMP / "test_1.png"
        
        # Queue multiple updates
        for i in range(5):
            self.queue.update_metadata(test_path, f"key_{i}", f"value_{i}")
            
        # Wait for processing
        time.sleep(1.0)
        
        # Verify all metadata was written
        result = self.queue.get_metadata(test_path)
        for i in range(5):
            assert f"key_{i}" in result
            assert result[f"key_{i}"] == f"value_{i}"
            
    def test_order_preservation(self):
        """Test that operations on the same key are processed in order"""
        test_path = TEST_TEMP / "test_2.png"
        
        # Queue ordered updates to the same key
        for i in range(10):
            self.queue.update_metadata(test_path, "sequence_key", f"value_{i}")
            
        # Wait for processing
        time.sleep(1.0)
        
        # Verify final value is the last one written
        result = self.queue.get_metadata(test_path)
        assert "sequence_key" in result
        assert result["sequence_key"] == "value_9"
        
    def test_priority_handling(self):
        """Test that higher priority tasks are processed first"""
        test_path = TEST_TEMP / "test_3.png"
        
        # First set a baseline value
        self.queue.update_metadata(test_path, "priority_key", "initial")
        time.sleep(0.5)
        
        # Queue many low-priority updates
        for i in range(20):
            self.queue.update_metadata(test_path, f"low_priority_{i}", f"value_{i}", priority=0)
            
        # Queue a high-priority update
        self.queue.update_metadata(test_path, "priority_key", "high_priority", priority=10)
        
        # Small wait to allow prioritization but not full processing
        time.sleep(0.1)
        
        # Verify high-priority update gets processed quickly
        result = self.queue.get_metadata(test_path)
        assert "priority_key" in result
        assert result["priority_key"] == "high_priority"
        
    def test_concurrent_file_access(self):
        """Test concurrent operations on different files"""
        file_count = 5
        update_count = 10
        
        # Queue many updates to multiple files
        for i in range(file_count):
            for j in range(update_count):
                self.queue.update_metadata(
                    TEST_TEMP / f"test_{i}.png",
                    f"concurrent_key_{j}",
                    f"file_{i}_value_{j}"
                )
                
        # Wait for processing
        time.sleep(2.0)
        
        # Verify all updates were applied correctly
        for i in range(file_count):
            result = self.queue.get_metadata(TEST_TEMP / f"test_{i}.png")
            for j in range(update_count):
                assert f"concurrent_key_{j}" in result
                assert result[f"concurrent_key_{j}"] == f"file_{i}_value_{j}"
                
    def test_batch_operations(self):
        """Test batch operations through the queue"""
        test_path = TEST_TEMP / "test_4.png"
        
        # Create batch update
        updates = []
        for i in range(50):  # Large number to ensure batching
            updates.append((test_path, f"batch_key_{i}", f"batch_value_{i}", 0))
            
        # Queue batch update
        self.queue.batch_update(updates)
        
        # Wait for processing (allowing for batching)
        time.sleep(2.0)
        
        # Verify all updates were applied
        result = self.queue.get_metadata(test_path)
        for i in range(50):
            assert f"batch_key_{i}" in result
            assert result[f"batch_key_{i}"] == f"batch_value_{i}"
            
    def test_file_locking(self):
        """Test that file locking prevents concurrent access to the same file"""
        test_path = TEST_TEMP / "test_5.png"
        lock_acquired = threading.Event()
        lock_released = threading.Event()
        update_attempted = threading.Event()
        update_completed = threading.Event()
        
        # Create a test image if it doesn't exist
        if not test_path.exists():
            self._create_test_image(test_path)
        
        # Get file lock from the queue system
        file_lock = self.queue._get_file_lock(test_path)
        
        def block_file():
            """Function to hold the file lock for a while"""
            with file_lock:
                lock_acquired.set()
                # Hold lock until signaled
                lock_released.wait()
        
        def attempt_update():
            """Function to attempt a direct metadata update"""
            update_attempted.set()
            # Use direct handler with the same external lock
            handler = PNGMetadataHandler(test_path, external_lock=file_lock)
            handler.update_metadata("lock_test", "value")
            update_completed.set()
        
        # Start a thread to hold the lock
        blocking_thread = threading.Thread(target=block_file)
        blocking_thread.daemon = True
        blocking_thread.start()
        
        # Wait for lock to be acquired
        lock_acquired.wait()
        
        # Try to update metadata while lock is held (in a separate thread)
        update_thread = threading.Thread(target=attempt_update)
        update_thread.daemon = True
        update_thread.start()
        
        # Wait until the update is attempted
        update_attempted.wait()
        
        # Wait a short time - update should not complete while lock is held
        time.sleep(0.2)
        assert not update_completed.is_set(), "Update completed while lock was held!"
        
        # Verify metadata was not updated - use the same lock for consistency
        handler = PNGMetadataHandler(test_path, external_lock=file_lock)
        metadata = handler.get_metadata()
        assert "lock_test" not in metadata, "Metadata was updated while lock was held!"
        
        # Release the lock
        lock_released.set()
        
        # Wait for the update to complete
        update_completed.wait(timeout=2.0)
        
        # Verify the update was processed - use the same lock for consistency
        handler = PNGMetadataHandler(test_path, external_lock=file_lock)
        metadata = handler.get_metadata()
        assert "lock_test" in metadata, "Metadata was not updated after lock was released!"
        
        # Clean up
        blocking_thread.join(timeout=1.0)
        update_thread.join(timeout=1.0)
        
    def test_error_handling_and_retries(self):
        """Test error handling and retry logic"""
        test_path = TEST_TEMP / "nonexistent.png"
        
        # Queue update to nonexistent file (should fail)
        self.queue.update_metadata(test_path, "error_key", "error_value")
        
        # Wait long enough for retry attempts
        time.sleep(3.0)
        
        # Verify stats show failures
        stats = self.queue.get_stats()
        assert stats["tasks_failed"] > 0
        
        # Now create the file and queue another update
        self._create_test_image(test_path)
        self.queue.update_metadata(test_path, "recovery_key", "recovery_value")
        
        # Wait for processing
        time.sleep(0.5)
        
        # Verify update was successful
        result = self.queue.get_metadata(test_path)
        assert "recovery_key" in result
        assert result["recovery_key"] == "recovery_value"
        
    def test_handler_caching(self):
        """Test that handlers are cached for better performance"""
        test_path = TEST_TEMP / "test_6.png"
        
        # Clear any existing cache
        self.queue.handler_cache = {}
        
        # Get handler (should create new one)
        handler1 = self.queue._get_handler(test_path)
        
        # Get handler again (should return cached one)
        handler2 = self.queue._get_handler(test_path)
        
        # Verify they're the same object
        assert handler1 is handler2
        
        # Verify cache has entry
        assert str(test_path) in self.queue.handler_cache
        
        # Test cache expiration by manipulating timestamps
        current_time = time.time()
        self.queue.handler_cache[str(test_path)] = (handler1, current_time - self.queue.handler_cache_ttl - 1)
        
        # Get handler again (should create new one due to TTL expiration)
        handler3 = self.queue._get_handler(test_path)
        
        # Verify it's a different object
        assert handler1 is not handler3
        
    def test_stress_with_many_files(self):
        """Stress test with many files and concurrent operations"""
        # Create many test files
        file_count = 20
        for i in range(10, 10 + file_count):
            self._create_test_image(TEST_TEMP / f"stress_{i}.png")
            
        # Queue many operations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            
            for i in range(10, 10 + file_count):
                for j in range(10):  # 10 operations per file
                    futures.append(
                        executor.submit(
                            self.queue.update_metadata,
                            TEST_TEMP / f"stress_{i}.png",
                            f"stress_key_{j}",
                            f"stress_value_{i}_{j}",
                            random.randint(0, 5)  # Random priority
                        )
                    )
                    
            # Wait for all submissions to complete
            concurrent.futures.wait(futures)
            
        # Wait for processing with more reliable completion detection
        max_wait = 5.0  # Increased maximum wait time
        start_time = time.time()
        while time.time() - start_time < max_wait:
            stats = self.queue.get_stats()
            if stats["tasks_processed"] >= file_count * 10:
                break  # All tasks processed
            time.sleep(0.1)  # Short polling interval
            
        # Verify all updates were applied
        for i in range(10, 10 + file_count):
            result = self.queue.get_metadata(TEST_TEMP / f"stress_{i}.png")
            for j in range(10):
                assert f"stress_key_{j}" in result
                assert result[f"stress_key_{j}"] == f"stress_value_{i}_{j}"
                
        # Check queue stats with proper explanation if failing
        stats = self.queue.get_stats()
        assert stats["tasks_processed"] >= file_count * 10, \
            f"Expected at least {file_count * 10} tasks, but only {stats['tasks_processed']} were processed"
        
    def test_queue_shutdown(self):
        """Test graceful queue shutdown"""
        # Queue some operations
        for i in range(10):
            self.queue.update_metadata(
                TEST_TEMP / f"test_{i}.png",
                "shutdown_key",
                f"shutdown_value_{i}"
            )
            
        # Shutdown with wait=True
        self.queue.stop(wait=True)
        
        # Verify all operations were processed - create handlers with their own locks
        # as the queue is now stopped and we don't need cross-thread synchronization
        for i in range(10):
            metadata = PNGMetadataHandler(TEST_TEMP / f"test_{i}.png").get_metadata()
            assert "shutdown_key" in metadata
            assert metadata["shutdown_key"] == f"shutdown_value_{i}"
            
        # Verify queue is stopped
        assert not self.queue.running
        
        # Try to queue another operation (should start queue again)
        self.queue.update_metadata(TEST_TEMP / "test_0.png", "restart_key", "restart_value")
        assert self.queue.running
        
        # Wait for processing
        time.sleep(0.5)
        
        # Clean up
        self.queue.stop(wait=True)
        
    def test_dependent_operations(self):
        """
        Test that operations that depend on previous state are processed correctly.
        This simulates a case where the new value depends on reading the current value.
        """
        test_path = TEST_TEMP / "test_7.png"
        
        # Get a lock for consistent access throughout this test
        file_lock = self.queue._get_file_lock(test_path)
        
        # Initialize with start value - use consistent lock
        handler = PNGMetadataHandler(test_path, external_lock=file_lock)
        handler.update_metadata("counter", "0")
        
        # Define a function that increments based on reading the current value
        def increment_counter():
            current = int(self.queue.get_metadata(test_path).get("counter", "0"))
            new_value = str(current + 1)
            self.queue.update_metadata(test_path, "counter", new_value)
            return new_value
            
        # Queue many increment operations
        expected_results = []
        for _ in range(10):
            expected_results.append(increment_counter())
            
        # Wait for processing
        time.sleep(2.0)
        
        # Verify final counter value - use consistent lock
        result = self.queue.get_metadata(test_path)
        assert "counter" in result
        assert result["counter"] == "10"  # Should have been incremented 10 times


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])