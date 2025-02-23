# Image Rating System with PNG Metadata Storage

## Overview
A sophisticated system for storing image ratings and metadata directly within PNG files while maintaining data integrity and handling high-resolution images efficiently.

## Key Features
- Thread-safe metadata operations
- Atomic file operations to prevent corruption
- Backup system for data integrity
- Efficient handling of large image files
- Preservation of existing metadata (ComfyUI parameters)

## Project Structure
```
backend/
├── image_handler/          # Core image handling functionality
├── storage/               # Image and backup storage
├── models/               # Data models and schemas
└── tests/               # Comprehensive test suite
```

## Core Components

### Metadata Manager
- Handles reading and writing of PNG metadata
- Implements atomic operations using temporary files
- Preserves existing metadata during updates
- Includes validation and error handling

### Queue System
- Thread-safe queue for image operations
- Background processing of metadata updates
- Priority handling for read operations
- Monitoring and logging capabilities

### Backup Service
- Automated backup scheduling
- Incremental backup system
- Validation of backup integrity
- Recovery procedures

## Testing Requirements

### Unit Tests
1. Metadata Read/Write Operations
   - Test atomic operations
   - Verify metadata preservation
   - Check error handling
   - Validate thread safety

2. Queue System
   - Test concurrent operations
   - Verify operation ordering
   - Check resource cleanup
   - Test error propagation

3. Backup System
   - Verify backup creation
   - Test recovery procedures
   - Check integrity validation
   - Test incremental backups

### Integration Tests
1. End-to-End Operations
   - Full metadata update cycle
   - Concurrent user operations
   - Recovery from simulated failures
   - Performance under load

2. Data Integrity
   - Verify ComfyUI parameters preservation
   - Check rating data consistency
   - Test backup/restore operations
   - Validate file locking mechanisms

## Development Guidelines
1. All file operations must be atomic
2. Existing metadata must be preserved
3. Implement proper error handling and logging
4. Include comprehensive test coverage
5. Document all public interfaces
6. Follow PEP 8 style guidelines

## Safety Measures
1. File locking during operations
2. Temporary file usage for atomic writes
3. Validation before saves
4. Automatic backup creation
5. Integrity checks after operations

## Future Enhancements
1. Automated backup rotation
2. Performance optimization for bulk operations
3. Advanced metadata compression
4. Enhanced recovery tools
5. Monitoring dashboard

## Getting Started
[Development setup instructions will go here]

## Contributing
[Contribution guidelines will go here]