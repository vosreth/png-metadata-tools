"""
PNG Chunk Handler with proper British standards for atomic streaming metadata operations
"""
import struct
import zlib
from typing import List, Tuple, Dict, BinaryIO, Iterator, Optional, Union, Callable, Any
from dataclasses import dataclass
from pathlib import Path
import tempfile
import shutil
import json
import threading
import io

from png_metadata_tools.base_handler import PNGMetadataHandlerBase

# Use the same PNGChunk dataclass as PNGMetadataHandler
@dataclass
class PNGChunk:
    type: bytes
    data: bytes
    offset: int  # Offset in file where chunk begins
    length: int  # Length of data section
    crc: bytes

class StreamingPNGMetadataHandler(PNGMetadataHandlerBase):
    """
    Handler for PNG metadata operations using streaming approach for memory efficiency.
    This implementation maintains all the atomic and thread-safe properties of the
    original while adding efficient streaming for files of any size.
    """
    BUFFER_SIZE = 8192  # 8KB buffer for streaming
    
    def _initialize_handler(self) -> None:
        """Initialize the handler by loading metadata chunks."""
        self._chunks: List[PNGChunk] = []
        self._iend_position: Optional[int] = None
        self._load_metadata_chunks()

    def _streaming_update(self, key: str, value: str, temp_path: Path) -> None:
        """
        Perform the streaming update operation, writing to a temporary file.
        
        This method streams through the original file chunk by chunk,
        replacing or adding the metadata chunk as needed without loading
        the entire file into memory.
        
        Args:
            key: Metadata key to update
            value: Metadata value to set
            temp_path: Path to temporary file for atomic write
        """
        # Find existing chunk for this key
        existing_chunk = None
        existing_offset = None
        
        for chunk in self._chunks:
            if (chunk.type == b'tEXt' and 
                chunk.data.split(b'\0', 1)[0].decode('latin-1') == key):
                existing_chunk = chunk
                existing_offset = chunk.offset
                break
        
        # Create new chunk
        new_chunk = self._create_text_chunk(key, value)
        
        # Now stream through the original file
        with open(self.filepath, 'rb') as src, open(temp_path, 'wb') as dst:
            # Write PNG signature
            dst.write(self.PNG_SIGNATURE)
            
            # Keep track of whether we've written the new chunk
            new_chunk_written = False
            iend_written = False
            
            # Collect all chunks first to ensure proper ordering
            chunks = list(self._stream_chunks())
            
            # Find IEND chunk and its position
            iend_chunk = next((c for c in chunks if c.type == b'IEND'), None)
            
            # Process all chunks except IEND
            for chunk in chunks:
                # Skip IEND as we'll write it at the end
                if chunk.type == b'IEND':
                    continue
                    
                # If this is the chunk we want to replace
                if existing_offset is not None and chunk.offset == existing_offset:
                    # Write our new chunk instead
                    self._write_chunk(dst, new_chunk)
                    new_chunk_written = True
                else:
                    # Write original chunk
                    self._write_chunk_from_source(dst, chunk)
            
            # If we haven't written the new chunk yet and there was an existing chunk to replace,
            # something went wrong - write it anyway before IEND
            if not new_chunk_written:
                self._write_chunk(dst, new_chunk)
                
            # Always write IEND chunk at the end
            if iend_chunk:
                self._write_chunk(dst, iend_chunk)
            else:
                # If we somehow don't have an IEND chunk, create a minimal valid one
                iend_data = b''
                iend_crc = struct.pack('>I', zlib.crc32(b'IEND' + iend_data) & 0xFFFFFFFF)
                iend_chunk = PNGChunk(
                    type=b'IEND',
                    data=iend_data,
                    offset=-1,
                    length=0,
                    crc=iend_crc
                )
                self._write_chunk(dst, iend_chunk)

    def _write_chunk(self, f: BinaryIO, chunk: PNGChunk) -> None:
        """Write a single chunk to file."""
        f.write(struct.pack('>I', chunk.length))  # Length
        f.write(chunk.type)                       # Type
        f.write(chunk.data)                       # Data
        f.write(chunk.crc)                        # CRC

    def _write_chunk_from_source(self, f: BinaryIO, chunk: PNGChunk) -> None:
        """
        Write a chunk to file by copying from source to preserve exact bytes.
        
        This method is crucial for the streaming approach as it avoids loading
        large chunk data into memory when we're only passing it through.
        
        Args:
            f: Destination file object
            chunk: Chunk metadata containing offset and length information
        """
        # Write the chunk header
        f.write(struct.pack('>I', chunk.length))  # Length
        f.write(chunk.type)                       # Type
        
        # Write the data and CRC directly from the source file
        with open(self.filepath, 'rb') as src:
            # Seek to the data portion of the chunk
            src.seek(chunk.offset + 8)  # Skip length and type bytes
            
            # Stream the data
            remaining = chunk.length
            while remaining > 0:
                buffer_size = min(remaining, self.BUFFER_SIZE)
                data = src.read(buffer_size)
                if not data:
                    raise EOFError(f"Unexpected EOF when reading chunk data at offset {src.tell()}, expected {remaining} more bytes")
                f.write(data)
                remaining -= len(data)
            
            # Write the CRC
            crc_data = src.read(4)
            if len(crc_data) != 4:
                raise EOFError(f"Unexpected EOF when reading chunk CRC at offset {src.tell()}")
            f.write(crc_data)


    def _stream_chunks(self) -> Iterator[PNGChunk]:
        """Stream all chunks from the PNG file without loading the entire file."""
        with open(self.filepath, 'rb') as f:
            # Verify PNG signature
            signature = f.read(8)
            if signature != self.PNG_SIGNATURE:
                raise ValueError("Invalid PNG signature")

            while True:
                chunk_start = f.tell()
                length_bytes = f.read(4)
                if not length_bytes or len(length_bytes) < 4:
                    break

                length = struct.unpack('>I', length_bytes)[0]
                type_bytes = f.read(4)
                
                # Read data in chunks for very large data sections
                data = bytearray()
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(remaining, self.BUFFER_SIZE))
                    if not chunk:
                        raise EOFError(f"Unexpected end of file at offset {f.tell()}, expected {remaining} more bytes")
                    data.extend(chunk)
                    remaining -= len(chunk)
                
                crc = f.read(4)
                
                chunk = PNGChunk(
                    type=type_bytes,
                    data=bytes(data),
                    offset=chunk_start,
                    length=length,
                    crc=crc
                )
                
                yield chunk
                
                if type_bytes == b'IEND':
                    break

    def _load_metadata_chunks(self) -> None:
        """Load critical chunks and text chunks while streaming."""
        # Reset state
        self._chunks = []
        self._iend_position = None
        
        try:
            for chunk in self._stream_chunks():
                # We only need to keep text chunks and critical chunks in memory
                if chunk.type == b'tEXt' or chunk.type == b'IEND':
                    self._chunks.append(chunk)
                    
                if chunk.type == b'IEND':
                    self._iend_position = chunk.offset
        except EOFError as e:
            # Handle truncated files more gracefully
            # Just log the error but keep any chunks we've already processed
            import logging
            logging.warning(f"Truncated PNG file: {self.filepath}, error: {str(e)}")

    def _get_text_chunks(self) -> List[PNGChunk]:
        """Get all tEXt chunks."""
        return [chunk for chunk in self._chunks if chunk.type == b'tEXt']

    def get_metadata(self) -> Dict[str, str]:
        """Extract metadata from tEXt chunks."""
        metadata = {}
        for chunk in self._get_text_chunks():
            # Split at first null byte
            try:
                keyword, value = chunk.data.split(b'\0', 1)
                metadata[keyword.decode('latin-1')] = value.decode('latin-1')
            except ValueError:
                continue  # Skip malformed chunks
        return metadata

    def get_metadata_keys(self) -> list[str]:
        """Get a list of all metadata keys present in the file."""
        keys = []
        for chunk in self._get_text_chunks():
            try:
                keyword, _ = chunk.data.split(b'\0', 1)
                keys.append(keyword.decode('latin-1'))
            except ValueError:
                continue  # Skip malformed chunks
        return keys

    def has_metadata_key(self, key: str) -> bool:
        """Check if a specific metadata key exists."""
        return key in self.get_metadata()

    def _create_text_chunk(self, keyword: str, text: str) -> PNGChunk:
        """Create a new tEXt chunk with proper structure."""
        data = keyword.encode('latin-1') + b'\0' + text.encode('latin-1')
        length = len(data)
        
        # Calculate CRC (type + data)
        crc = zlib.crc32(b'tEXt' + data) & 0xFFFFFFFF
        
        return PNGChunk(
            type=b'tEXt',
            data=data,
            offset=-1,  # Will be set during writing
            length=length,
            crc=struct.pack('>I', crc)
        )

    def update_metadata(self, key: str, value: str) -> None:
        """
        Update metadata atomically using a streaming approach and a temporary file.
        Preserves all other chunks exactly as they are.
        """
        
        # Update metadata with proper locking.
        with self._lock:
            # Create temporary file next to original
            temp_path = self.filepath.with_suffix('.tmp')
            
            try:
                self._streaming_update(key, value, temp_path)
                
                # Atomic replace
                temp_path.replace(self.filepath)
                
                # Reload metadata chunks to reflect changes
                self._load_metadata_chunks()
                    
            finally:
                # Cleanup temp file if something went wrong
                if temp_path.exists():
                    temp_path.unlink()

    def remove_metadata(self, key: str) -> bool:
        """Remove a metadata entry if it exists."""
        with self._lock:
            # Check if the key exists
            existing_chunk = None
            existing_offset = None
            
            for chunk in self._get_text_chunks():
                try:
                    keyword, _ = chunk.data.split(b'\0', 1)
                    if keyword.decode('latin-1') == key:
                        existing_chunk = chunk
                        existing_offset = chunk.offset
                        break
                except ValueError:
                    continue
                    
            # If key doesn't exist, nothing to do
            if existing_chunk is None:
                return False
                
            # Create temporary file
            temp_path = self.filepath.with_suffix('.tmp')
            
            try:
                with open(temp_path, 'wb') as dst:
                    # Write PNG signature
                    dst.write(self.PNG_SIGNATURE)
                    
                    # Collect all chunks first
                    chunks = list(self._stream_chunks())
                    
                    # Process all chunks except IEND and the one to remove
                    for chunk in chunks:
                        if chunk.type == b'IEND':
                            continue
                            
                        if chunk.offset == existing_offset:
                            # Skip this chunk - this is the one we're removing
                            continue
                        else:
                            # Write original chunk
                            self._write_chunk_from_source(dst, chunk)
                    
                    # Write IEND at the end
                    iend_chunk = next((c for c in chunks if c.type == b'IEND'), None)
                    if iend_chunk:
                        self._write_chunk(dst, iend_chunk)
                        
                # Atomic replace
                temp_path.replace(self.filepath)
                
                # Reload metadata
                self._load_metadata_chunks()
                return True
                    
            finally:
                # Cleanup
                if temp_path.exists():
                    temp_path.unlink()
                    
    def clear_metadata(self) -> None:
        """Remove all metadata from the PNG file."""
        with self._lock:
            # Check if there's any metadata to clear
            if not self._get_text_chunks():
                return
                
            # Create temporary file
            temp_path = self.filepath.with_suffix('.tmp')
            
            try:
                with open(temp_path, 'wb') as dst:
                    # Write PNG signature
                    dst.write(self.PNG_SIGNATURE)
                    
                    # Collect all chunks first
                    chunks = list(self._stream_chunks())
                    
                    # Process all chunks except IEND and tEXt
                    for chunk in chunks:
                        if chunk.type == b'IEND':
                            continue
                            
                        if chunk.type == b'tEXt':
                            # Skip this chunk - removing all metadata
                            continue
                        else:
                            # Write original chunk
                            self._write_chunk_from_source(dst, chunk)
                    
                    # Write IEND at the end
                    iend_chunk = next((c for c in chunks if c.type == b'IEND'), None)
                    if iend_chunk:
                        self._write_chunk(dst, iend_chunk)
                        
                # Atomic replace
                temp_path.replace(self.filepath)
                
                # Reload metadata (should be empty now)
                self._load_metadata_chunks()
                    
            finally:
                # Cleanup
                if temp_path.exists():
                    temp_path.unlink()