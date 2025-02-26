"""
PNG Metadata Batch Operations with proper British standards
"""
import threading
from pathlib import Path
from typing import Dict, List, Union, Any, Optional, Set, Callable
from concurrent.futures import ThreadPoolExecutor
import time
import logging
import os

from png_metadata_tools.base_handler import PNGMetadataHandlerBase
from png_metadata_tools.metadata_queue import PNGMetadataQueue

# Configure logging
logger = logging.getLogger("png_metadata_batch")

class BatchOperation:
    """Representation of a single operation within a batch."""
    
    def __init__(self, operation_type: str, key: str = None, value: Any = None):
        self.operation_type = operation_type  # 'update', 'remove', or 'clear'
        self.key = key
        self.value = value
        self.timestamp = time.time()
    
    def apply(self, handler: PNGMetadataHandlerBase) -> bool:
        """Apply this operation to the given handler."""
        if self.operation_type == 'update':
            handler.update_metadata(self.key, str(self.value))
            return True
        elif self.operation_type == 'remove':
            return handler.remove_metadata(self.key)
        elif self.operation_type == 'clear':
            handler.clear_metadata()
            return True
        else:
            logger.warning(f"Unknown operation type: {self.operation_type}")
            return False
    
    def __str__(self):
        if self.operation_type == 'update':
            return f"Update {self.key}={self.value}"
        elif self.operation_type == 'remove':
            return f"Remove {self.key}"
        elif self.operation_type == 'clear':
            return "Clear all metadata"
        return f"Unknown operation: {self.operation_type}"


class QueueTaskHandler:
    """Processor for special queue operations"""
    @staticmethod
    def process_special_task(file_path: Path, metadata: Dict[str, str]) -> Dict[str, str]:
        """
        Process special marker tasks and return updated metadata.
        Returns the modified metadata dict.
        """
        handler = PNGMetadataHandlerBase.create(file_path)
        original_metadata = handler.get_metadata()
        
        # Look for special removal markers
        for key in list(metadata.keys()):
            if key.startswith('__REMOVE__'):
                target_key = key[10:]  # Remove the "__REMOVE__" prefix
                handler.remove_metadata(target_key)
        
        # Look for clear all marker
        if '__CLEAR_ALL__' in metadata:
            handler.clear_metadata()
            return {}  # Return empty dict since all metadata is cleared
            
        # Return the final metadata state
        return handler.get_metadata()


class BatchEditor:
    """
    Context manager for batch editing multiple PNG files.
    
    Supports both direct handler operations and queue-based operations
    for optimal performance in different scenarios.
    
    Example:
        with BatchEditor() as batch:
            for image in folder.glob("*.png"):
                batch.update(image, {"rating": 1500.0, "processed": True})
    """
    def __init__(self, workers: int = 2, use_queue: bool = True, 
                 streaming_threshold: Optional[int] = None):
        """
        Initialize a batch editor.
        
        Args:
            workers: Number of worker threads to use
            use_queue: Whether to use queue-based processing
            streaming_threshold: File size threshold for streaming mode, None to use default
        """
        self.workers = workers
        self.use_queue = use_queue
        self.streaming_threshold = streaming_threshold
        self.queue = None
        self.operations = {}  # filepath -> list of operations
        self.processed_files = set()
        self.errors = []
        
    def __enter__(self):
        if self.use_queue:
            self.queue = PNGMetadataQueue(num_workers=self.workers)
            self.queue.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if not self.use_queue:
                # Process any remaining operations directly
                self._process_direct_operations()
            elif self.queue:
                # Process any special operations in the queue
                self._process_special_queue_operations()
                
            if self.queue:
                self.queue.stop(wait=True)
                
            logger.info(f"Batch processing completed: {len(self.processed_files)} files processed, "
                      f"{len(self.errors)} errors")
        except Exception as e:
            logger.error(f"Error during batch shutdown: {e}")
    
    def _process_special_queue_operations(self):
        """Process any special operations like removals and clears"""
        # This is now handled by post-processing each file that has special markers
        for filepath in list(self.processed_files):
            try:
                path = Path(filepath)
                if path.exists():
                    QueueTaskHandler.process_special_task(path, self.queue.get_metadata(path))
            except Exception as e:
                logger.error(f"Error processing special operations for {filepath}: {e}")
                self.errors.append((filepath, str(e)))
    
    def update(self, filepath: Union[str, Path], metadata: Dict[str, Any], 
              priority: int = 0) -> 'BatchEditor':
        """
        Queue metadata updates for a file.
        
        Args:
            filepath: Path to the PNG file.
            metadata: Dictionary of metadata to update.
            priority: Priority of the operation (higher = processed sooner).
        
        Returns:
            Self for method chaining
        """
        filepath = Path(filepath) if isinstance(filepath, str) else filepath
        
        if self.use_queue:
            updates = []
            for key, value in metadata.items():
                updates.append((filepath, key, str(value), priority))
            self.queue.batch_update(updates)
            self.processed_files.add(str(filepath))
        else:
            # Store operations for direct processing
            filepath_str = str(filepath)
            if filepath_str not in self.operations:
                self.operations[filepath_str] = []
                
            for key, value in metadata.items():
                self.operations[filepath_str].append(
                    BatchOperation('update', key, value)
                )
            self.processed_files.add(filepath_str)
        
        return self
    
    def update_many(self, filepaths: List[Union[str, Path]], 
                   metadata: Dict[str, Any], priority: int = 0) -> 'BatchEditor':
        """
        Queue the same metadata updates for multiple files.
        
        Args:
            filepaths: List of file paths to update.
            metadata: Dictionary of metadata to update.
            priority: Priority of the operation (higher = processed sooner).
            
        Returns:
            Self for method chaining
        """
        for filepath in filepaths:
            self.update(filepath, metadata, priority)
        return self
    
    def remove(self, filepath: Union[str, Path], keys: Union[str, List[str]], 
              priority: int = 0) -> 'BatchEditor':
        """
        Remove metadata keys from a file.
        
        Args:
            filepath: Path to the PNG file.
            keys: Key or list of keys to remove.
            priority: Priority of the operation.
            
        Returns:
            Self for method chaining
        """
        filepath = Path(filepath) if isinstance(filepath, str) else filepath
        keys_list = [keys] if isinstance(keys, str) else keys
        
        if self.use_queue:
            updates = []
            
            for key in keys_list:
                # We use a special marker value to indicate removal
                updates.append((filepath, f"__REMOVE__{key}", "", priority))
            
            if updates:
                self.queue.batch_update(updates)
            self.processed_files.add(str(filepath))
        else:
            # Store operations for direct processing
            filepath_str = str(filepath)
            if filepath_str not in self.operations:
                self.operations[filepath_str] = []
                
            for key in keys_list:
                self.operations[filepath_str].append(
                    BatchOperation('remove', key)
                )
            self.processed_files.add(filepath_str)
        
        return self
    
    def clear(self, filepath: Union[str, Path], priority: int = 0) -> 'BatchEditor':
        """
        Clear all metadata from a file.
        
        Args:
            filepath: Path to the PNG file.
            priority: Priority of the operation.
            
        Returns:
            Self for method chaining
        """
        filepath = Path(filepath) if isinstance(filepath, str) else filepath
        
        if self.use_queue:
            # For queue-based processing, we use a special marker
            self.queue.update_metadata(filepath, "__CLEAR_ALL__", "", priority)
            self.processed_files.add(str(filepath))
        else:
            # Store operations for direct processing
            filepath_str = str(filepath)
            if filepath_str not in self.operations:
                self.operations[filepath_str] = []
                
            self.operations[filepath_str].append(
                BatchOperation('clear')
            )
            self.processed_files.add(filepath_str)
        
        return self
    
    def process_glob(self, pattern: str, operation: Callable[[Path], None], 
                      recursive: bool = False) -> 'BatchEditor':
        """
        Process all files matching a glob pattern with a custom operation.
        
        Args:
            pattern: Glob pattern for files
            operation: Function that takes a Path and performs operations
            recursive: Whether to search directories recursively
            
        Returns:
            Self for method chaining
        """
        # Make sure we search from the current directory or absolute paths
        if os.path.isabs(pattern):
            base_path = Path(os.path.dirname(pattern))
            pattern = os.path.basename(pattern)
        else:
            base_path = Path('.')
        
        matched_files = []
        if recursive:
            for path in base_path.glob('**/' + pattern):
                if path.is_file():
                    matched_files.append(path)
        else:
            matched_files.extend(base_path.glob(pattern))
        
        logger.info(f"Found {len(matched_files)} files matching pattern: {pattern}")
        
        # Process each file with the operation
        for file_path in matched_files:
            try:
                operation(file_path)
                self.processed_files.add(str(file_path))
                logger.debug(f"Successfully processed {file_path}")
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                logger.error(error_msg)
                self.errors.append((str(file_path), str(e)))
        
        return self
    
    def _process_direct_operations(self):
        """Process stored operations directly using multiple threads."""
        if not self.operations:
            return
            
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            
            for filepath_str, ops in self.operations.items():
                futures.append(
                    executor.submit(self._process_file_operations, 
                                   filepath_str, ops)
                )
            
            # Wait for all operations to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    self.errors.append((None, str(e)))
    
    def _process_file_operations(self, filepath_str: str, operations: List[BatchOperation]):
        """Process all operations for a single file."""
        try:
            filepath = Path(filepath_str)
            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
                
            # Create handler with appropriate implementation
            handler = PNGMetadataHandlerBase.create(
                filepath, 
                use_streaming=self._should_use_streaming(filepath)
            )
            
            # Apply operations in sequence
            for operation in operations:
                operation.apply(handler)
                
            self.processed_files.add(filepath_str)
        except Exception as e:
            self.errors.append((filepath_str, str(e)))
    
    def _should_use_streaming(self, filepath: Path) -> Optional[bool]:
        """Determine if streaming should be used for this file."""
        if self.streaming_threshold is None:
            return None  # Let the factory method decide
            
        # Use streaming for files larger than the threshold
        return filepath.stat().st_size > self.streaming_threshold


class BatchProcessor:
    """
    Utility for advanced batch processing of PNG metadata.
    """
    
    def __init__(self, workers: int = 2):
        self.workers = workers
        self.results = {
            "processed": 0,
            "skipped": 0,
            "errors": [],
            "modified_files": []
        }
    
    def process(self, file_patterns: List[str], operation_factory: Callable,
               filter_fn: Optional[Callable[[Path], bool]] = None,
               recursive: bool = False) -> Dict[str, Any]:
        """
        Process files matching patterns with a custom operation.
        """
        matched_files = []
        
        # Improved pattern handling for tests
        for pattern in file_patterns:
            pattern_path = Path(pattern)
            
            # Check if it's already an absolute path or has a parent directory
            if pattern_path.is_absolute() or str(pattern_path.parent) != '.':
                # Use the pattern directly with glob
                if recursive:
                    parent = pattern_path.parent
                    glob_pattern = f"**/{pattern_path.name}"
                    for path in parent.glob(glob_pattern):
                        if path.is_file():
                            matched_files.append(path)
                else:
                    matched_files.extend(pattern_path.parent.glob(pattern_path.name))
            else:
                # Use current directory
                if recursive:
                    for path in Path().glob(f"**/{pattern}"):
                        if path.is_file():
                            matched_files.append(path)
                else:
                    matched_files.extend(Path().glob(pattern))
        
        # Deduplicate files
        matched_files = list(set(matched_files))
        
        # Debug log for troubleshooting
        logger.info(f"Matched {len(matched_files)} files for patterns: {file_patterns}")
        
        with BatchEditor(workers=self.workers, use_queue=False) as batch:
            for file_path in matched_files:
                try:
                    # Apply filter if provided
                    if filter_fn and not filter_fn(file_path):
                        self.results["skipped"] += 1
                        continue
                    
                    # Create and apply operation - pass both batch and file_path
                    operation = operation_factory(file_path)
                    operation(batch, file_path)
                    
                    self.results["processed"] += 1
                    self.results["modified_files"].append(str(file_path))
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    self.results["errors"].append((str(file_path), str(e)))
        
        return self.results


def create_metadata_filter(required_keys: Optional[List[str]] = None,
                         value_conditions: Optional[Dict[str, Callable[[str], bool]]] = None):
    """
    Create a filter function that checks metadata conditions.
    
    Args:
        required_keys: List of keys that must be present
        value_conditions: Dict mapping keys to validator functions
        
    Returns:
        Filter function for use with BatchProcessor
    """
    def filter_fn(file_path: Path) -> bool:
        try:
            # Create handler using factory method
            handler = PNGMetadataHandlerBase.create(file_path)
            metadata = handler.get_metadata()
            
            # Check required keys
            if required_keys:
                for key in required_keys:
                    if key not in metadata:
                        return False
            
            # Check value conditions
            if value_conditions:
                for key, validator in value_conditions.items():
                    if key in metadata:
                        if not validator(metadata[key]):
                            return False
            
            return True
        except Exception:
            return False
    
    return filter_fn


def create_update_operation(metadata: Dict[str, Any], priority: int = 0):
    """Create an operation function for updating metadata."""
    def operation(batch: BatchEditor, file_path: Path):
        batch.update(file_path, metadata, priority)
    return operation


def create_conditional_operation(condition_fn: Callable[[Path], bool],
                              true_op: Callable[[BatchEditor, Path], None],
                              false_op: Optional[Callable[[BatchEditor, Path], None]] = None):
    """Create an operation that applies different operations based on a condition."""
    def operation(batch: BatchEditor, file_path: Path):
        if condition_fn(file_path):
            true_op(batch, file_path)
        elif false_op:
            false_op(batch, file_path)
    return operation