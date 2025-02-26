"""
Test suite for PNG Metadata Backup System with proper British standards
"""
import pytest
import json
import shutil
import time
from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np

from png_metadata_tools.metadata_backup import MetadataBackupManager
from png_metadata_tools.base_handler import PNGMetadataHandlerBase

# Test data paths
TEST_ROOT = Path(__file__).parent
TEST_DATA = TEST_ROOT / "data"
TEST_IMAGES = TEST_DATA / "images"
TEST_TEMP = TEST_DATA / "temp" / "backup_tests"
TEST_BACKUPS = TEST_TEMP / "backups"


class TestMetadataBackup:
    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Setup test environment with fresh test images and backup directory"""
        # Ensure directories exist
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        TEST_BACKUPS.mkdir(parents=True, exist_ok=True)
        
        # Create test images with metadata
        self._create_test_images()
        
        # Initialize backup manager
        self.backup_mgr = MetadataBackupManager(TEST_BACKUPS)
        
        yield
        
        # Cleanup after tests
        if TEST_TEMP.exists():
            shutil.rmtree(TEST_TEMP)
            
    def _create_test_images(self):
        """Create various test images with different metadata configurations"""
        # Simple image with basic metadata
        basic_path = TEST_TEMP / "basic_metadata.png"
        self._create_test_image(basic_path, metadata={
            'title': 'Basic Test Image',
            'author': 'British Standards',
            'rating': '1500.0'
        })
        
        # Image with extensive metadata
        extensive_path = TEST_TEMP / "extensive_metadata.png"
        self._create_test_image(extensive_path, metadata={
            'title': 'Extensive Test Image',
            'author': 'British Standards Institute',
            'rating': '1700.0',
            'description': 'A test image with extensive metadata',
            'keywords': 'test,metadata,backup',
            'software': 'PNG Metadata Tools',
            'created': '2025-02-25',
            'uuid': 'd290f1ee-6c54-4b01-90e6-d701748f0851',
            'category': 'test',
            'priority': 'high'
        })
        
        # Empty metadata image
        empty_path = TEST_TEMP / "empty_metadata.png"
        self._create_test_image(empty_path)
        
        # Special characters in metadata
        special_path = TEST_TEMP / "special_chars.png"
        self._create_test_image(special_path, metadata={
            'title': 'Special Characters: £€$¥',
            'notes': 'Line 1\nLine 2\nLine 3',
            'symbols': '!@#$%^&*()'
        })
        
        # Create subdirectory with additional images
        subdir = TEST_TEMP / "subdir"
        subdir.mkdir(exist_ok=True)
        
        self._create_test_image(subdir / "sub_image_1.png", metadata={
            'location': 'subdirectory',
            'index': '1'
        })
        
        self._create_test_image(subdir / "sub_image_2.png", metadata={
            'location': 'subdirectory',
            'index': '2'
        })
            
    def _create_test_image(self, path: Path, size=(256, 256), metadata=None):
        """Create a test image with optional metadata"""
        # Create image data
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
            
    def _verify_backup_content(self, backup_path, original_path):
        """Verify that a backup file contains the correct metadata"""
        # Read the backup file
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
            
        # Check backup format
        assert 'timestamp' in backup_data
        assert 'metadata' in backup_data
        assert 'source_path' in backup_data
        assert 'file_size' in backup_data
        assert 'dimensions' in backup_data
        
        # Read original metadata
        handler = PNGMetadataHandlerBase.create(original_path)
        original_metadata = handler.get_metadata()
        
        # Compare metadata
        assert backup_data['metadata'] == original_metadata
        assert str(original_path.absolute()) == backup_data['source_path'] or \
               backup_data['source_path'].endswith(str(original_path))

    def test_create_backup(self):
        """Test creating a backup of image metadata"""
        # Test image to back up
        image_path = TEST_TEMP / "basic_metadata.png"
        
        # Create a backup
        backup_path = self.backup_mgr.create_backup(image_path)
        
        # Verify backup was created
        assert backup_path is not None
        assert backup_path.exists()
        
        # Verify backup content
        self._verify_backup_content(backup_path, image_path)
        
    def test_create_versioned_backup(self):
        """Test creating a versioned backup"""
        image_path = TEST_TEMP / "basic_metadata.png"
        
        # Create a versioned backup
        version = "test_version_1.0"
        backup_path = self.backup_mgr.create_versioned_backup(image_path, version)
        
        # Verify backup was created
        assert backup_path is not None
        assert backup_path.exists()
        assert version in str(backup_path)
        
        # Verify backup content
        self._verify_backup_content(backup_path, image_path)
        
    def test_restore_from_backup(self):
        """Test restoring metadata from a backup"""
        # Original image
        original_path = TEST_TEMP / "extensive_metadata.png"
        
        # Create a backup
        backup_path = self.backup_mgr.create_backup(original_path)
        
        # Create a test target for restoration
        target_path = TEST_TEMP / "restore_target.png"
        shutil.copy(original_path, target_path)
        
        # Clear metadata in target
        handler = PNGMetadataHandlerBase.create(target_path)
        handler.clear_metadata()
        
        # Verify metadata was cleared
        assert not handler.get_metadata()
        
        # Restore from backup
        success = self.backup_mgr.restore_from_backup(backup_path, target_path)
        
        # Verify restoration was successful
        assert success is True
        
        # Compare metadata
        original_handler = PNGMetadataHandlerBase.create(original_path)
        restored_handler = PNGMetadataHandlerBase.create(target_path)
        
        assert original_handler.get_metadata() == restored_handler.get_metadata()
        
    def test_restore_to_original_location(self):
        """Test restoring to the original location (default behavior)"""
        # Original image
        original_path = TEST_TEMP / "basic_metadata.png"
        
        # Get original metadata
        original_handler = PNGMetadataHandlerBase.create(original_path)
        original_metadata = original_handler.get_metadata()
        
        # Create a backup
        backup_path = self.backup_mgr.create_backup(original_path)
        
        # Clear metadata in original
        original_handler.clear_metadata()
        
        # Verify metadata was cleared
        assert not original_handler.get_metadata()
        
        # Restore from backup without specifying target
        success = self.backup_mgr.restore_from_backup(backup_path)
        
        # Verify restoration was successful
        assert success is True
        
        # Verify metadata was restored
        restored_handler = PNGMetadataHandlerBase.create(original_path)
        assert original_metadata == restored_handler.get_metadata()
        
    def test_batch_backup(self):
        """Test batch backup of multiple files"""
        # Get all PNG files in the test directory
        image_paths = list(TEST_TEMP.glob("*.png"))
        assert len(image_paths) >= 3  # Ensure we have multiple files
        
        # Perform batch backup
        results = self.backup_mgr.batch_backup(image_paths)
        
        # Verify results
        assert results["total"] == len(image_paths)
        assert results["successful"] == len(image_paths)
        assert results["failed"] == 0
        assert len(results["backups"]) == len(image_paths)
        
        # Verify each backup
        for backup_path_str in results["backups"]:
            backup_path = Path(backup_path_str)
            assert backup_path.exists()
            
            # Find original file from backup data
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            source_path = Path(backup_data["source_path"])
            
            # Verify source exists in our original list
            matching_paths = [p for p in image_paths if p.name == source_path.name]
            assert len(matching_paths) > 0
            
    def test_batch_restore(self):
        """Test batch restore from multiple backups"""
        # Create backups for multiple files
        image_paths = list(TEST_TEMP.glob("*.png"))[:3]  # Limit to 3 files
        
        # Create backups
        backup_paths = []
        for path in image_paths:
            backup_path = self.backup_mgr.create_backup(path)
            backup_paths.append(backup_path)
            
        # Create target directory
        target_dir = TEST_TEMP / "restore_targets"
        target_dir.mkdir(exist_ok=True)
        
        # Copy images to target with cleared metadata
        target_paths = []
        for path in image_paths:
            target_path = target_dir / path.name
            shutil.copy(path, target_path)
            
            # Clear metadata
            handler = PNGMetadataHandlerBase.create(target_path)
            handler.clear_metadata()
            
            target_paths.append(target_path)
        
        # Perform batch restore
        results = self.backup_mgr.batch_restore(backup_paths, target_dir)
        
        # Verify results
        assert results["total"] == len(backup_paths)
        assert results["successful"] == len(backup_paths)
        assert results["failed"] == 0
        
        # Verify each restoration
        for i, path in enumerate(image_paths):
            original_handler = PNGMetadataHandlerBase.create(path)
            restored_handler = PNGMetadataHandlerBase.create(target_paths[i])
            
            assert original_handler.get_metadata() == restored_handler.get_metadata()
            
    def test_list_backups(self):
        """Test listing backups"""
        # Create multiple backups
        self.backup_mgr.create_backup(TEST_TEMP / "basic_metadata.png")
        self.backup_mgr.create_backup(TEST_TEMP / "extensive_metadata.png")
        self.backup_mgr.create_versioned_backup(TEST_TEMP / "basic_metadata.png", "v1")
        
        # List all backups
        backups = self.backup_mgr.list_backups()
        
        # Verify we have the expected number
        assert len(backups) >= 3
        
        # Verify structure of backup entries
        for backup in backups:
            assert "timestamp" in backup
            assert "metadata" in backup
            assert "source_path" in backup
            assert "backup_path" in backup
            assert "date_str" in backup
            
    def test_find_backups_for_image(self):
        """Test finding backups for a specific image"""
        # Create multiple backups for the same image
        image_path = TEST_TEMP / "basic_metadata.png"
        
        self.backup_mgr.create_backup(image_path)
        time.sleep(0.1)  # Ensure different timestamps
        self.backup_mgr.create_versioned_backup(image_path, "v1")
        time.sleep(0.1)
        self.backup_mgr.create_versioned_backup(image_path, "v2")
        
        # Also create backups for other images
        self.backup_mgr.create_backup(TEST_TEMP / "extensive_metadata.png")
        
        # Find backups for the specific image
        backups = self.backup_mgr.find_backups_for_image(image_path)
        
        # Verify we found the correct number
        assert len(backups) == 3
        
        # Verify they're sorted by timestamp (newest first)
        assert backups[0]["timestamp"] > backups[1]["timestamp"]
        assert backups[1]["timestamp"] > backups[2]["timestamp"]
        
    def test_compare_backups(self):
        """Test comparing two backups"""
        # Original image
        image_path = TEST_TEMP / "basic_metadata.png"
        
        # Create test image with initial metadata
        self._create_test_image(image_path, metadata={
            'title': 'Test Image',
            'author': 'British Standards',
            'rating': '1500.0'
        })
        
        # Create first backup (older) - IMMEDIATELY after creation
        older_backup = self.backup_mgr.create_backup(image_path)
        
        # Modify metadata AFTER creating the first backup
        handler = PNGMetadataHandlerBase.create(image_path)
        handler.update_metadata("new_key", "new_value")
        handler.update_metadata("rating", "1600.0")  # Change existing key
        handler.remove_metadata("title")  # Remove a key
        
        # Create second backup (newer) - with modified metadata
        newer_backup = self.backup_mgr.create_backup(image_path)
        
        # Compare backups
        comparison = self.backup_mgr.compare_backups(newer_backup, older_backup)
        
        # Verify comparison results
        assert "new_key" in comparison["added_keys"]
        assert "title" in comparison["removed_keys"]
        assert "rating" in comparison["changed_keys"]
        assert "author" in comparison["unchanged_keys"]
        
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Non-existent image
        result = self.backup_mgr.create_backup(TEST_TEMP / "nonexistent.png")
        assert result is None
        
        # Non-existent backup file
        result = self.backup_mgr.restore_from_backup(TEST_BACKUPS / "nonexistent.json")
        assert result is False
        
        # Invalid backup file (not JSON)
        invalid_backup = TEST_BACKUPS / "invalid.json"
        invalid_backup.write_text("This is not valid JSON")
        
        result = self.backup_mgr.restore_from_backup(invalid_backup)
        assert result is False
        
    def test_daily_backup(self):
        """Test creating a daily backup"""
        image_path = TEST_TEMP / "basic_metadata.png"
        
        # Create daily backup
        backup_path = self.backup_mgr.create_daily_backup(image_path)
        
        # Verify backup was created
        assert backup_path is not None
        assert backup_path.exists()
        assert "daily" in str(backup_path)
        
        # Verify backup content
        self._verify_backup_content(backup_path, image_path)
        
    def test_recursive_backup(self):
        """Test backing up images in subdirectories"""
        # Get all PNG files recursively
        all_images = list(TEST_TEMP.glob("**/*.png"))
        
        # Verify we have images in subdirectory
        subdir_images = list(TEST_TEMP.glob("subdir/*.png"))
        assert len(subdir_images) > 0
        
        # Backup all images
        results = self.backup_mgr.batch_backup(all_images)
        
        # Verify all were backed up
        assert results["successful"] == len(all_images)
        
        # Verify subdir images were backed up
        for img_path in subdir_images:
            backups = self.backup_mgr.find_backups_for_image(img_path)
            assert len(backups) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])