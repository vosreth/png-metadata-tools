"""
PNG Chunk Handler with proper British standards for atomic metadata operations
"""
import struct
import zlib
from typing import List, Tuple, Dict, BinaryIO
from dataclasses import dataclass
from pathlib import Path
import tempfile
import shutil
import json
from typing import Optional

@dataclass
class PNGChunk:
    type: bytes
    data: bytes
    offset: int  # Offset in file where chunk begins
    length: int  # Length of data section
    crc: bytes

class PNGMetadataHandler:
    PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
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

    def _write_chunk(self, f: BinaryIO, chunk: PNGChunk) -> None:
        """Write a single chunk to file."""
        f.write(struct.pack('>I', chunk.length))  # Length
        f.write(chunk.type)                       # Type
        f.write(chunk.data)                       # Data
        f.write(chunk.crc)                        # CRC

    def update_elo_rating(self, rating: float) -> None:
        """Update elo rating while preserving all other metadata."""
        self.update_metadata('elo_rating', str(rating))

    def get_elo_rating(self) -> Optional[float]:
        """Get elo rating if it exists."""
        metadata = self.get_metadata()
        if 'elo_rating' in metadata:
            try:
                return float(metadata['elo_rating'])
            except ValueError:
                return None
        return None