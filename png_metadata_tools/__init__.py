"""
PNG Metadata Tools - A sophisticated system for PNG metadata operations with proper British standards
"""
from pathlib import Path
from typing import Dict, Union, Optional, List, Any
import threading
import os

__version__ = "0.1.0"
__author__ = "Proper British Engineering Ltd."

# Directly expose key functionality through the package namespace
from png_metadata_tools.base_handler import PNGMetadataHandlerBase

# Import implementations
from png_metadata_tools.chunk_handler import PNGMetadataHandler
from png_metadata_tools.streaming_chunk_handler import StreamingPNGMetadataHandler
from png_metadata_tools.metadata_queue import PNGMetadataQueue
from png_metadata_tools.png_inspector import EnhancedPNGInspector

# Update the import statement to use the full-featured implementation
from png_metadata_tools.batch_operations import (
    BatchEditor,
    BatchProcessor, 
    create_metadata_filter, 
    create_update_operation, 
    create_conditional_operation,
    QueueTaskHandler
)

# Simplified API for common operations
def read(filepath: Union[str, Path]) -> Dict[str, str]:
    """
    Read metadata from a PNG file.
    
    Args:
        filepath: Path to the PNG file.
        
    Returns:
        Dictionary of metadata key-value pairs.
    """
    handler = PNGMetadataHandlerBase.create(filepath)
    return handler.get_metadata()

def write(filepath: Union[str, Path], metadata: Dict[str, str]) -> None:
    """
    Write multiple metadata key-value pairs to a PNG file.
    
    Args:
        filepath: Path to the PNG file.
        metadata: Dictionary of metadata to write.
    """
    handler = PNGMetadataHandlerBase.create(filepath)
    for key, value in metadata.items():
        handler.update_metadata(key, str(value))

def update(filepath: Union[str, Path], key: str, value: Any) -> None:
    """
    Update a single metadata value.
    
    Args:
        filepath: Path to the PNG file.
        key: Metadata key to update.
        value: Metadata value to set (will be converted to string).
    """
    handler = PNGMetadataHandlerBase.create(filepath)
    handler.update_metadata(key, str(value))

def has_key(filepath: Union[str, Path], key: str) -> bool:
    """
    Check if a PNG file has a specific metadata key.
    
    Args:
        filepath: Path to the PNG file.
        key: Metadata key to check.
        
    Returns:
        True if the key exists, False otherwise.
    """
    handler = PNGMetadataHandlerBase.create(filepath)
    return handler.has_metadata_key(key)

def remove(filepath: Union[str, Path], key: str) -> bool:
    """
    Remove a metadata key from a PNG file.
    
    Args:
        filepath: Path to the PNG file.
        key: Metadata key to remove.
        
    Returns:
        True if the key was removed, False if it did not exist.
    """
    handler = PNGMetadataHandlerBase.create(filepath)
    return handler.remove_metadata(key)

def clear(filepath: Union[str, Path]) -> None:
    """
    Clear all metadata from a PNG file.
    
    Args:
        filepath: Path to the PNG file.
    """
    handler = PNGMetadataHandlerBase.create(filepath)
    handler.clear_metadata()

def inspect(filepath: Union[str, Path], detailed: bool = False) -> Dict[str, Any]:
    """
    Inspect a PNG file for detailed metadata and structure information.
    
    Args:
        filepath: Path to the PNG file.
        detailed: If True, includes more detailed information.
        
    Returns:
        Dictionary with detailed inspection results.
    """
    inspector = EnhancedPNGInspector(str(filepath))
    if detailed:
        inspector.print_detailed_summary()
    return inspector.metadata

class MetaEditor:
    """
    Context manager for editing PNG metadata.
    
    Example:
        with pngmeta.MetaEditor("image.png") as meta:
            meta["rating"] = 1500.0
            meta["processed"] = True
    """
    def __init__(self, filepath: Union[str, Path], use_streaming: Optional[bool] = None):
        self.filepath = filepath
        self.use_streaming = use_streaming
        self.handler = None
        self._metadata = None
        
    def __enter__(self):
        self.handler = PNGMetadataHandlerBase.create(self.filepath, self.use_streaming)
        self._metadata = self.handler.get_metadata()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Only write changes if no exception occurred
            for key, value in self._updates.items():
                self.handler.update_metadata(key, str(value))
        
    def __getitem__(self, key):
        return self._metadata.get(key)
        
    def __setitem__(self, key, value):
        self._updates[key] = value
        
    def __delitem__(self, key):
        if key in self._metadata:
            self._deletes.add(key)
            
    def __contains__(self, key):
        return key in self._metadata and key not in self._deletes
        
    @property
    def _updates(self):
        if not hasattr(self, '_update_dict'):
            self._update_dict = {}
        return self._update_dict
        
    @property
    def _deletes(self):
        if not hasattr(self, '_delete_set'):
            self._delete_set = set()
        return self._delete_set

class BatchEditor:
    """
    Context manager for batch editing multiple PNG files.
    
    Example:
        with pngmeta.BatchEditor() as batch:
            for image in folder.glob("*.png"):
                batch.update(image, {"rating": 1500.0, "processed": True})
    """
    def __init__(self, workers: int = 2):
        self.workers = workers
        self.queue = None
        
    def __enter__(self):
        self.queue = PNGMetadataQueue(num_workers=self.workers)
        self.queue.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.queue:
            self.queue.stop(wait=True)
            
    def update(self, filepath: Union[str, Path], metadata: Dict[str, Any], priority: int = 0):
        """
        Queue metadata updates for a file.
        
        Args:
            filepath: Path to the PNG file.
            metadata: Dictionary of metadata to update.
            priority: Priority of the operation (higher = processed sooner).
        """
        updates = []
        for key, value in metadata.items():
            updates.append((Path(filepath), key, str(value), priority))
        self.queue.batch_update(updates)
            
    def update_many(self, filepaths: List[Union[str, Path]], metadata: Dict[str, Any], priority: int = 0):
        """
        Queue the same metadata updates for multiple files.
        
        Args:
            filepaths: List of file paths to update.
            metadata: Dictionary of metadata to update.
            priority: Priority of the operation (higher = processed sooner).
        """
        for filepath in filepaths:
            self.update(filepath, metadata, priority)

# Extend the Queue class to handle special operations
class Queue:
    """
    Queue system for asynchronous metadata operations.
    
    Example:
        queue = pngmeta.Queue(workers=4)
        queue.start()
        queue.update("image.png", "rating", 1750.0, priority=10)
        results = queue.get("image.png")
        queue.stop()
    """
    def __init__(self, workers: int = 2):
        self.queue = PNGMetadataQueue(num_workers=workers)
        
    def start(self):
        """Start the queue processing system."""
        self.queue.start()
        
    def stop(self, wait: bool = True):
        """
        Stop the queue processing system.
        
        Args:
            wait: If True, wait for all queued tasks to complete.
        """
        self.queue.stop(wait=wait)
        
    def update(self, filepath: Union[str, Path], key: str, value: Any, priority: int = 0):
        """
        Queue a metadata update operation.
        
        Args:
            filepath: Path to the PNG file.
            key: Metadata key to update.
            value: Metadata value to set (will be converted to string).
            priority: Priority of the operation (higher = processed sooner).
        """
        self.queue.update_metadata(Path(filepath), key, str(value), priority)
        
    def get(self, filepath: Union[str, Path]) -> Dict[str, str]:
        """
        Get metadata from a file, ensuring any pending updates are processed first.
        
        Args:
            filepath: Path to the PNG file.
            
        Returns:
            Dictionary of metadata key-value pairs.
        """
        filepath = Path(filepath)
        # Process any special operations first
        metadata = self.queue.get_metadata(filepath)
        
        # Process special markers if any
        if any(k.startswith('__REMOVE__') for k in metadata) or '__CLEAR_ALL__' in metadata:
            return QueueTaskHandler.process_special_task(filepath, metadata)
            
        return metadata
        
    def update_batch(self, updates: List[tuple]):
        """
        Queue multiple metadata updates at once.
        
        Args:
            updates: List of (filepath, key, value, priority) tuples.
        """
        formatted_updates = []
        for filepath, key, value, priority in updates:
            formatted_updates.append((Path(filepath), key, str(value), priority))
        self.queue.batch_update(formatted_updates)
        
    def stats(self) -> Dict[str, Any]:
        """
        Get queue statistics.
        
        Returns:
            Dictionary with queue statistics.
        """
        return self.queue.get_stats()

__all__ = [
    # Core classes
    'PNGMetadataHandlerBase',
    'PNGMetadataHandler',
    'StreamingPNGMetadataHandler',
    'PNGMetadataQueue',
    'EnhancedPNGInspector',
    
    # Simplified API functions
    'read', 'write', 'update', 'has_key', 'remove', 'clear', 'inspect',
    
    # Editor classes
    'MetaEditor', 'BatchEditor', 'Queue',
    
    # Batch operations
    'BatchProcessor',
    'create_metadata_filter',
    'create_update_operation',
    'create_conditional_operation'
]