"""
PNG Metadata Backup System with proper British standards
"""
import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Union, Any, Callable, Tuple
import threading

from png_metadata_tools.base_handler import PNGMetadataHandlerBase

# Configure logging
logger = logging.getLogger("png_metadata_backup")


@dataclass
class BackupMetadata:
    """Representation of a backup entry"""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, str] = field(default_factory=dict)
    source_path: str = ""
    file_size: int = 0
    dimensions: Tuple[int, int] = (0, 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "source_path": self.source_path,
            "file_size": self.file_size,
            "dimensions": self.dimensions,
            "date_str": datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackupMetadata':
        """Create instance from dictionary"""
        return cls(
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
            source_path=data.get("source_path", ""),
            file_size=data.get("file_size", 0),
            dimensions=tuple(data.get("dimensions", (0, 0)))
        )


class MetadataBackupManager:
    """
    Manager for PNG metadata backups with proper British standards.
    
    This class provides functionality for:
    - Creating backups of metadata without copying the entire image
    - Restoring metadata from backups
    - Managing backup versions
    - Batch backup and restore operations
    
    The backups are stored in a structured format that ensures durability and
    facilitates easy restoration, all while adhering to proper British standards
    of software engineering.
    """
    
    def __init__(self, backup_dir: Union[str, Path], use_streaming: bool = False):
        """
        Initialize the backup manager.
        
        Args:
            backup_dir: Directory for storing backups
            use_streaming: Whether to use streaming for large files
        """
        self.backup_dir = Path(backup_dir)
        self.use_streaming = use_streaming
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standard backup subdirectories
        self.daily_dir = self.backup_dir / "daily"
        self.versions_dir = self.backup_dir / "versions"
        self.temp_dir = self.backup_dir / "temp"
        
        # Ensure directories exist
        for directory in [self.daily_dir, self.versions_dir, self.temp_dir]:
            directory.mkdir(exist_ok=True)
            
        # Keep track of in-progress backups
        self._in_progress: Set[str] = set()
        self._in_progress_lock = threading.Lock()
    
    def _get_backup_path(self, image_path: Path, version: Optional[str] = None) -> Path:
        """
        Get the backup file path for an image.
        
        Args:
            image_path: Path to the original image
            version: Optional version identifier (if None, uses timestamp)
            
        Returns:
            Path to the backup file
        """
        # Generate a standardized filename based on the original
        relative_path = image_path.name
        if version:
            backup_filename = f"{image_path.stem}_{version}.json"
            return self.versions_dir / backup_filename
        else:
            # Use timestamp for versioning
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            backup_filename = f"{image_path.stem}_{timestamp}.json"
            return self.daily_dir / backup_filename
    
    def create_backup(self, image_path: Union[str, Path], 
                    version: Optional[str] = None) -> Optional[Path]:
        """
        Create a backup of image metadata.
        
        Args:
            image_path: Path to the image
            version: Optional version identifier
            
        Returns:
            Path to the backup file if successful, None otherwise
        """
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None
        
        # Check if backup is already in progress
        image_key = str(image_path.absolute())
        with self._in_progress_lock:
            if image_key in self._in_progress:
                logger.warning(f"Backup already in progress for {image_path}")
                return None
            self._in_progress.add(image_key)
        
        try:
            # Create handler with appropriate implementation
            handler = PNGMetadataHandlerBase.create(
                image_path, use_streaming=self.use_streaming
            )
            
            # Get metadata and create a DEEP COPY to prevent reference issues
            # This is the critical fix - we must ensure we capture the exact state
            # at this moment, not a reference that might be modified later
            metadata = handler.get_metadata()
            metadata_snapshot = metadata.copy()  # Create a completely new dictionary

            # Get image dimensions if possible
            dimensions = (0, 0)
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    dimensions = img.size
            except Exception as e:
                logger.warning(f"Could not get image dimensions: {e}")
            
            # Create backup entry with our snapshot
            backup_data = BackupMetadata(
                timestamp=time.time(),
                metadata=metadata_snapshot,  # Use our clean copy
                source_path=str(image_path.absolute()),
                file_size=image_path.stat().st_size,
                dimensions=dimensions
            )
            
            # Determine backup path
            backup_path = self._get_backup_path(image_path, version)
            
            # Write backup to temporary file first
            temp_path = self.temp_dir / f"{backup_path.name}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                backup_dict = backup_data.to_dict()
                json.dump(backup_dict, f, indent=2)
            
            # Move to final location (atomic operation)
            temp_path.replace(backup_path)
            
            logger.info(f"Backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Error creating backup for {image_path}: {e}")
            return None
        finally:
            # Remove from in-progress set
            with self._in_progress_lock:
                self._in_progress.discard(image_key)
    
    def batch_backup(self, image_paths: List[Union[str, Path]], 
                   workers: int = 4) -> Dict[str, Any]:
        """
        Create backups for multiple images in parallel.
        
        Args:
            image_paths: List of paths to images
            workers: Number of worker threads
            
        Returns:
            Dictionary with results
        """
        results = {
            "total": len(image_paths),
            "successful": 0,
            "failed": 0,
            "backups": []
        }
        
        # Convert all paths to Path objects
        paths = [Path(p) if isinstance(p, str) else p for p in image_paths]
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.create_backup, path): path for path in paths}
            
            for future in futures:
                path = futures[future]
                try:
                    backup_path = future.result()
                    if backup_path:
                        results["successful"] += 1
                        results["backups"].append(str(backup_path))
                    else:
                        results["failed"] += 1
                except Exception as e:
                    logger.error(f"Error in batch backup for {path}: {e}")
                    results["failed"] += 1
        
        return results
    
    def restore_from_backup(self, backup_path: Union[str, Path], 
                          target_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Restore metadata from a backup.
        
        Args:
            backup_path: Path to the backup file
            target_path: Path to the target image (if None, uses original path)
            
        Returns:
            True if successful, False otherwise
        """
        backup_path = Path(backup_path)
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        try:
            # Load backup data
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_dict = json.load(f)
            
            backup_data = BackupMetadata.from_dict(backup_dict)
            
            # Determine target path
            if target_path is None:
                target_path = Path(backup_data.source_path)
            else:
                target_path = Path(target_path)
            
            if not target_path.exists():
                logger.error(f"Target image not found: {target_path}")
                return False
            
            # Create handler for target image
            handler = PNGMetadataHandlerBase.create(
                target_path, use_streaming=self.use_streaming
            )
            
            # Clear existing metadata
            handler.clear_metadata()
            
            # Apply backed up metadata
            for key, value in backup_data.metadata.items():
                handler.update_metadata(key, value)
            
            logger.info(f"Metadata restored to {target_path} from {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from backup {backup_path}: {e}")
            return False
    
    def batch_restore(self, backup_paths: List[Union[str, Path]], 
                    target_dir: Optional[Union[str, Path]] = None,
                    workers: int = 4) -> Dict[str, Any]:
        """
        Restore metadata from multiple backups in parallel.
        
        Args:
            backup_paths: List of paths to backup files
            target_dir: Optional target directory (if None, uses original paths)
            workers: Number of worker threads
            
        Returns:
            Dictionary with results
        """
        results = {
            "total": len(backup_paths),
            "successful": 0,
            "failed": 0
        }
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            
            for backup_path in backup_paths:
                backup_path = Path(backup_path)
                
                # Determine target path
                target_path = None
                if target_dir is not None:
                    # Load backup to get original filename
                    try:
                        with open(backup_path, 'r', encoding='utf-8') as f:
                            backup_dict = json.load(f)
                        
                        source_path = Path(backup_dict.get("source_path", ""))
                        if source_path.name:
                            target_path = Path(target_dir) / source_path.name
                    except Exception as e:
                        logger.error(f"Error determining target path for {backup_path}: {e}")
                        results["failed"] += 1
                        continue
                
                futures.append(executor.submit(self.restore_from_backup, backup_path, target_path))
            
            for future in futures:
                try:
                    if future.result():
                        results["successful"] += 1
                    else:
                        results["failed"] += 1
                except Exception as e:
                    logger.error(f"Error in batch restore: {e}")
                    results["failed"] += 1
        
        return results
    
    def list_backups(self, filter_fn: Optional[Callable[[BackupMetadata], bool]] = None) -> List[Dict[str, Any]]:
        """
        List all backups, optionally filtered.
        
        Args:
            filter_fn: Optional filter function
            
        Returns:
            List of backup metadata dictionaries
        """
        backups = []
        
        # Search both version and daily directories
        for directory in [self.versions_dir, self.daily_dir]:
            for backup_file in directory.glob("*.json"):
                try:
                    with open(backup_file, 'r', encoding='utf-8') as f:
                        backup_dict = json.load(f)
                    
                    backup_data = BackupMetadata.from_dict(backup_dict)
                    
                    # Apply filter if provided
                    if filter_fn and not filter_fn(backup_data):
                        continue
                    
                    # Add file path to the dict
                    result = backup_data.to_dict()
                    result["backup_path"] = str(backup_file)
                    
                    backups.append(result)
                except Exception as e:
                    logger.error(f"Error reading backup {backup_file}: {e}")
        
        # Sort by timestamp, newest first
        backups.sort(key=lambda x: x["timestamp"], reverse=True)
        return backups
    
    def find_backups_for_image(self, image_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Find all backups for a specific image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            List of backup metadata dictionaries
        """
        image_path = Path(image_path)
        abs_path = str(image_path.absolute())
        
        def filter_by_source(backup: BackupMetadata) -> bool:
            return backup.source_path == abs_path or backup.source_path.endswith(str(image_path))
        
        return self.list_backups(filter_by_source)
    
    def create_daily_backup(self, image_path: Union[str, Path]) -> Optional[Path]:
        """
        Create a daily backup with standardized naming.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Path to the backup file if successful, None otherwise
        """
        # Get today's date string
        date_str = datetime.now().strftime("%Y%m%d")
        return self.create_backup(image_path, version=f"daily_{date_str}")
    
    def create_versioned_backup(self, image_path: Union[str, Path], 
                              version: str) -> Optional[Path]:
        """
        Create a versioned backup with custom identifier.
        
        Args:
            image_path: Path to the image
            version: Version identifier
            
        Returns:
            Path to the backup file if successful, None otherwise
        """
        return self.create_backup(image_path, version=version)
        
    def compare_backups(self, newer_backup: Union[str, Path], 
                       older_backup: Union[str, Path]) -> Dict[str, Any]:
        """
        Compare two backups and report differences.
        
        Args:
            newer_backup: Path to the newer backup
            older_backup: Path to the older backup
            
        Returns:
            Dictionary with comparison results containing:
            - added_keys: Keys present in newer backup but not in older backup
            - removed_keys: Keys present in older backup but not in newer backup
            - changed_keys: Keys present in both but with different values
            - unchanged_keys: Keys present in both with identical values
            - newer_info: Information about the newer backup
            - older_info: Information about the older backup
        """
        newer_path = Path(newer_backup)
        older_path = Path(older_backup)
        
        result = {
            "added_keys": [],
            "removed_keys": [],
            "changed_keys": [],
            "unchanged_keys": [],
            "newer_info": {},
            "older_info": {}
        }
        
        try:
            # Load both backups
            with open(newer_path, 'r', encoding='utf-8') as f:
                newer_dict = json.load(f)
            
            with open(older_path, 'r', encoding='utf-8') as f:
                older_dict = json.load(f)
            
            newer_data = BackupMetadata.from_dict(newer_dict)
            older_data = BackupMetadata.from_dict(older_dict)
            
            # Add debug logging to trace metadata content
            logger.debug(f"Newer metadata keys: {list(newer_data.metadata.keys())}")
            logger.debug(f"Older metadata keys: {list(older_data.metadata.keys())}")
            
            # Basic info
            result["newer_info"] = {
                "timestamp": newer_data.timestamp,
                "date": datetime.fromtimestamp(newer_data.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                "path": str(newer_path)
            }
            
            result["older_info"] = {
                "timestamp": older_data.timestamp,
                "date": datetime.fromtimestamp(older_data.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                "path": str(older_path)
            }
            
            # Compare metadata
            newer_keys = set(newer_data.metadata.keys())
            older_keys = set(older_data.metadata.keys())
            
            # Keys in newer but not in older are added
            result["added_keys"] = list(newer_keys - older_keys)
            # Keys in older but not in newer are removed
            result["removed_keys"] = list(older_keys - newer_keys)
            
            # Check for changed values
            common_keys = newer_keys.intersection(older_keys)
            for key in common_keys:
                if newer_data.metadata[key] != older_data.metadata[key]:
                    result["changed_keys"].append(key)
                else:
                    result["unchanged_keys"].append(key)
            
            # Additional debug for key differences
            logger.debug(f"Added keys: {result['added_keys']}")
            logger.debug(f"Removed keys: {result['removed_keys']}")
            logger.debug(f"Changed keys: {result['changed_keys']}")
            
            return result
        except Exception as e:
            logger.error(f"Error comparing backups: {e}")
            return {"error": str(e)}