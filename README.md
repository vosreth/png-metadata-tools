# Image Rating System with PNG Metadata Storage

## Overview
A sophisticated system for storing image ratings and metadata directly within PNG files while maintaining data integrity and handling high-resolution images efficiently. Our implementation provides remarkable performance improvements over standard PIL methods, particularly for larger files.

## Key Features
- Thread-safe metadata operations
- Atomic file operations to prevent corruption
- Efficient handling of large image files (up to 36MP tested)
- Preservation of existing metadata (ComfyUI parameters)
- **Direct chunk manipulation** for significantly faster metadata operations
- **Performance optimized** with up to 10x speedup for high-resolution images

## Project Structure
```
png_metadata_tools/           # Core PNG metadata manipulation library
├── __init__.py              # Package initialization
├── chunk_handler.py         # Direct PNG chunk manipulation
└── png_inspector.py         # Enhanced metadata inspection tools

tests/                      # Comprehensive test suite
├── __init__.py
├── data/                   # Test data directory
│   ├── images/             # Test images
│   └── temp/               # Temporary test files
├── generate_test_files.py  # Test file generation utilities
├── test_chunk_handler.py   # Tests for chunk manipulation
├── test_metadata.py        # Tests for metadata operations
└── test_speed_comparison.py # Performance benchmarking
```

## Core Components

### PNG Chunk Handler
- Direct manipulation of PNG chunks without processing entire image data
- Up to 10x faster than PIL for metadata operations on large files
- Implements atomic operations using temporary files
- Preserves existing metadata during updates
- Thread-safe implementation with file locking

### Enhanced PNG Inspector
- Detailed metadata extraction and analysis
- Special handling for ComfyUI parameters
- Support for rating data extraction and validation
- Comprehensive error handling and reporting

### Speed Optimization
- Bypasses complete image loading for metadata operations
- Atomic file operations with minimal overhead
- Optimized for both small edits and large files
- Performance scales efficiently with image size

## Testing Results

### Performance Gains
Our direct chunk manipulation approach demonstrates significant performance advantages over PIL:
- Small images (0.25MP): 1.5-2x faster
- Medium images (4MP): 3-5x faster
- Large images (16MP+): 7-10x faster

These improvements become particularly significant when processing batches of images or implementing rating systems that require frequent metadata updates.

## Architectural Considerations

### Components Within This Module's Purview
- ✅ Core chunk handling (already implemented)
- ✅ Enhanced metadata inspection (already implemented)
- ✅ Atomic operations with temporary files (already implemented)
- ✅ Metadata-only backup and restoration
- ✅ Queue system for metadata operations
- ✅ Streaming approach for large files

### Better Placed in Parent/Separate Module
- ❌ Full image backup systems
- ❌ Image storage management
- ❌ User interfaces for metadata editing
- ❌ Integration with image processing pipelines

This arrangement maintains the module's focused purpose while providing comprehensive metadata capabilities that can be integrated into any larger system. By limiting scope to metadata operations exclusively, we ensure the module remains independent and reusable across various projects.

## Implementation Status

### Completed Features
- ✅ Core PNG chunk manipulation implementation
- ✅ Atomic metadata operations with validation
- ✅ Comprehensive test suite with various image sizes
- ✅ Performance benchmarking and comparison with PIL
- ✅ ComfyUI metadata preservation
- ✅ Detailed metadata inspection tools

### Planned Enhancements
- 🔄 Metadata-only backup service
- 🔄 Queue system for concurrent metadata operations
- 🔄 Streaming approach for extremely large files
- 🔄 Extended API for batch operations
- 🔄 Integrity verification with checksums

## Safety Measures
1. File locking during operations
2. Temporary file usage for atomic writes
3. Validation before saves
4. Metadata-only backup capability
5. Integrity checks after operations

## Getting Started

### Installation
```bash
# Installation instructions will be added when package is ready for distribution
```

### Basic Usage
```python
from png_metadata_tools.chunk_handler import PNGMetadataHandler

# Read metadata
handler = PNGMetadataHandler("path/to/image.png")
current_rating = handler.get_elo_rating()

# Update metadata (atomic operation)
handler.update_elo_rating(1500.0)

# Get all metadata
all_metadata = handler.get_metadata()
```

## Contributing
Contributions are welcome! Please ensure your code follows our development guidelines and includes appropriate tests.

---

*Delivering proper British standards for digital excellence since 2025*