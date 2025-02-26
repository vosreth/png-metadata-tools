"""
PNG Metadata Handler Base Interface with proper British standards
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, BinaryIO, Callable, Any, Iterator, Union
from pathlib import Path
import threading

class PNGMetadataHandlerBase(ABC):
    """
    Abstract base class defining the interface for all PNG metadata handlers.
    
    This interface establishes the contract that all implementations must fulfill,
    ensuring consistent behavior regardless of the underlying implementation details.
    The interface supports both standard and streaming approaches while maintaining
    proper British standards for metadata operations.
    """
    
    PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
    
    def __init__(self, filepath: Union[str, Path], external_lock: Optional[threading.Lock] = None):
        """
        Initialize the metadata handler.
        
        Args:
            filepath: Path to the PNG file.
            external_lock: Optional external lock for thread synchronization.
                           If None, a new lock will be created.
        """
        self.filepath = Path(filepath) if isinstance(filepath, str) else filepath
        self._lock = external_lock or threading.Lock()
        
        # Verify file exists
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        # Load metadata
        self._initialize_handler()
    
    @abstractmethod
    def _initialize_handler(self) -> None:
        """
        Initialize the handler-specific state.
        This is called during initialization to set up the handler.
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, str]:
        """
        Extract metadata from the PNG file.
        
        Returns:
            Dictionary mapping metadata keys to values.
        """
        pass
    
    @abstractmethod
    def update_metadata(self, key: str, value: str) -> None:
        """
        Update metadata atomically using a temporary file.
        Preserves all other chunks exactly as they are.
        
        Args:
            key: Metadata key to update.
            value: Metadata value to set.
        """
        pass
    
    @abstractmethod
    def get_metadata_keys(self) -> list[str]:
        """
        Get a list of all metadata keys present in the file.
        
        Returns:
            List of metadata key strings.
        """
        pass
    
    @abstractmethod
    def has_metadata_key(self, key: str) -> bool:
        """
        Check if a specific metadata key exists.
        
        Args:
            key: The metadata key to check.
            
        Returns:
            True if the key exists, False otherwise.
        """
        pass
    
    @abstractmethod
    def remove_metadata(self, key: str) -> bool:
        """
        Remove a metadata entry if it exists.
        
        Args:
            key: The metadata key to remove.
            
        Returns:
            True if the key was removed, False if it did not exist.
        """
        pass
    
    @abstractmethod
    def clear_metadata(self) -> None:
        """
        Remove all metadata from the PNG file.
        """
        pass
    
    @classmethod
    def create(cls, filepath: Union[str, Path], use_streaming: Optional[bool] = None, 
               external_lock: Optional[threading.Lock] = None) -> 'PNGMetadataHandlerBase':
        """
        Factory method to create the appropriate metadata handler.
        
        Args:
            filepath: Path to the PNG file.
            use_streaming: If True, forces streaming mode; if False, forces standard mode;
                           if None, selects automatically based on file size.
            external_lock: Optional external lock for thread synchronization.
            
        Returns:
            An instance of a PNGMetadataHandlerBase implementation.
        """
        from pathlib import Path
        import os
        
        path = Path(filepath) if isinstance(filepath, str) else filepath
        
        # Auto-select implementation if not specified
        if use_streaming is None:
            # Use streaming for large files (>10MB by default)
            threshold = int(os.environ.get('PNG_METADATA_STREAMING_THRESHOLD', 10_000_000))
            use_streaming = path.stat().st_size > threshold
        
        # Import implementations here to avoid circular imports
        if use_streaming:
            from png_metadata_tools.streaming_chunk_handler import StreamingPNGMetadataHandler
            return StreamingPNGMetadataHandler(path, external_lock)
        else:
            from png_metadata_tools.chunk_handler import PNGMetadataHandler
            return PNGMetadataHandler(path, external_lock)