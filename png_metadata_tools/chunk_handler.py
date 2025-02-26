
"""
PNG Chunk Handler with proper British standards for atomic metadata operations
"""
import struct
import zlib
from typing import List, Tuple, Dict, BinaryIO, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import tempfile
import shutil
import json
import threading

from png_metadata_tools.base_handler import PNGMetadataHandlerBase

@dataclass
class PNGChunk:
    type: bytes
    data: bytes
    offset: int  # Offset in file where chunk begins
    length: int  # Length of data section
    crc: bytes

class PNGMetadataHandler(PNGMetadataHandlerBase):
    """
    Standard implementation of PNG metadata handler using complete file processing.
    Efficient for small to medium files.
    """
    
    def _initialize_handler(self) -> None:
        """Initialize handler by loading all chunks."""
        self._chunks: List[PNGChunk] = []
        self._iend_position: Optional[int] = None
        self._load_metadata_chunks()

    def _load_metadata_chunks(self) -> None:
        """Load all chunks and track IEND position."""
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
                
                # Read data for all chunks
                data = f.read(length)
                crc = f.read(4)
                
                chunk = PNGChunk(
                    type=type_bytes,
                    data=data,
                    offset=chunk_start,
                    length=length,
                    crc=crc
                )
                
                self._chunks.append(chunk)
                
                if type_bytes == b'IEND':
                    self._iend_position = chunk_start
                    break

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
        Update metadata atomically using a temporary file.
        Preserves all other chunks exactly as they are.
        """
        
        # Update metadata with proper locking.
        with self._lock:
          # Create temporary file next to original
          temp_path = self.filepath.with_suffix('.tmp')
          
          try:
              with open(self.filepath, 'rb') as src, open(temp_path, 'wb') as dst:
                  # Write PNG signature
                  dst.write(self.PNG_SIGNATURE)
                  
                  # Find existing chunk for this key
                  existing_chunk = None
                  existing_index = None
                  
                  for i, chunk in enumerate(self._chunks):
                      if (chunk.type == b'tEXt' and 
                          chunk.data.split(b'\0', 1)[0].decode('latin-1') == key):
                          existing_chunk = chunk
                          existing_index = i
                          break
                  
                  # Create new chunk
                  new_chunk = self._create_text_chunk(key, value)
                  
                  # Write all chunks except IEND
                  iend_chunk = None
                  for chunk in self._chunks:
                      if chunk.type == b'IEND':
                          iend_chunk = chunk
                          continue
                          
                      if chunk is existing_chunk:
                          # Write new chunk instead
                          self._write_chunk(dst, new_chunk)
                      else:
                          # Write original chunk
                          self._write_chunk(dst, chunk)
                  
                  # If we didn't replace an existing chunk, add the new one before IEND
                  if existing_chunk is None:
                      self._write_chunk(dst, new_chunk)
                      
                  # Write IEND chunk at the end
                  if iend_chunk:
                      self._write_chunk(dst, iend_chunk)
                      
              # Atomic replace
              temp_path.replace(self.filepath)
              
              # Update our chunk list
              if existing_chunk:
                  self._chunks[existing_index] = new_chunk
              else:
                  # Insert before IEND
                  iend_index = next((i for i, c in enumerate(self._chunks) if c.type == b'IEND'), len(self._chunks))
                  self._chunks.insert(iend_index, new_chunk)
                  
          finally:
              # Cleanup temp file if something went wrong
              if temp_path.exists():
                  temp_path.unlink()

    def remove_metadata(self, key: str) -> bool:
        """Remove a metadata entry if it exists."""
        with self._lock:
            # Find existing chunk for this key
            existing_chunk = None
            existing_index = None
            
            for i, chunk in enumerate(self._chunks):
                if (chunk.type == b'tEXt' and 
                    chunk.data.split(b'\0', 1)[0].decode('latin-1') == key):
                    existing_chunk = chunk
                    existing_index = i
                    break
            
            # If key doesn't exist, nothing to do
            if existing_chunk is None:
                return False
                
            # Create temporary file next to original
            temp_path = self.filepath.with_suffix('.tmp')
            
            try:
                with open(self.filepath, 'rb') as src, open(temp_path, 'wb') as dst:
                    # Write PNG signature
                    dst.write(self.PNG_SIGNATURE)
                    
                    # Write all chunks except the one to remove and IEND
                    iend_chunk = None
                    for chunk in self._chunks:
                        if chunk.type == b'IEND':
                            iend_chunk = chunk
                            continue
                            
                        if chunk is existing_chunk:
                            # Skip this chunk
                            continue
                        else:
                            # Write original chunk
                            self._write_chunk(dst, chunk)
                        
                    # Write IEND chunk at the end
                    if iend_chunk:
                        self._write_chunk(dst, iend_chunk)
                        
                # Atomic replace
                temp_path.replace(self.filepath)
                
                # Update our chunk list
                self._chunks.pop(existing_index)
                return True
                    
            finally:
                # Cleanup temp file if something went wrong
                if temp_path.exists():
                    temp_path.unlink()

    def clear_metadata(self) -> None:
        """Remove all metadata from the PNG file."""
        with self._lock:
            # Check if any metadata exists
            text_chunks = self._get_text_chunks()
            if not text_chunks:
                return
                
            # Create temporary file next to original
            temp_path = self.filepath.with_suffix('.tmp')
            
            try:
                with open(self.filepath, 'rb') as src, open(temp_path, 'wb') as dst:
                    # Write PNG signature
                    dst.write(self.PNG_SIGNATURE)
                    
                    # Write all chunks except tEXt and IEND
                    iend_chunk = None
                    for chunk in self._chunks:
                        if chunk.type == b'IEND':
                            iend_chunk = chunk
                            continue
                            
                        if chunk.type == b'tEXt':
                            # Skip text chunks
                            continue
                        else:
                            # Write original chunk
                            self._write_chunk(dst, chunk)
                        
                    # Write IEND chunk at the end
                    if iend_chunk:
                        self._write_chunk(dst, iend_chunk)
                        
                # Atomic replace
                temp_path.replace(self.filepath)
                
                # Update our chunk list - remove all text chunks
                self._chunks = [c for c in self._chunks if c.type != b'tEXt']
                    
            finally:
                # Cleanup temp file if something went wrong
                if temp_path.exists():
                    temp_path.unlink()

    def _write_chunk(self, f: BinaryIO, chunk: PNGChunk) -> None:
        """Write a single chunk to file."""
        f.write(struct.pack('>I', chunk.length))  # Length
        f.write(chunk.type)                       # Type
        f.write(chunk.data)                       # Data
        f.write(chunk.crc)                        # CRC