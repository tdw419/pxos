# pxOS Filesystem Specification v0.1

## Abstract

The pxOS (Pixel Operating System) filesystem is a spatial, content-addressable storage system where every file is represented as a single pixel on an infinite 2D map. Each pixel serves as both a visual representation and a compressed blueprint that can reconstruct the original file.

## Design Principles

1. **Infinite Spatial Canvas**: Unlimited storage space via infinite 2D coordinate system
2. **Pixel-as-File**: Every file compresses to exactly one pixel
3. **Nested Hierarchy**: Directories are container pixels containing child pixels
4. **Content-Addressable**: Files deduplicated via cryptographic hashing
5. **Bootable**: System can boot directly from pixel representation

## Pixel Types

### FILE_PIXEL
A standard file compressed to a single pixel.

**Properties:**
- `type`: "file"
- `position`: {x: int, y: int} - Absolute coordinates on infinite map
- `visual`: {r: 0-255, g: 0-255, b: 0-255, a: 0-255} - Display color
- `metadata`: File attributes (name, size, permissions, timestamps)
- `data_hash`: SHA-256 hash of file content
- `backend`: Storage backend identifier

**Visual Encoding:**
- **R channel**: File type category
  - 255: Executable/binary
  - 200: Image/media
  - 150: Text/document
  - 100: Archive/compressed
  - 50: Configuration
  - 0: Other
- **G channel**: Size class (logarithmic scale)
  - 255: > 1GB
  - 200: 100MB - 1GB
  - 150: 10MB - 100MB
  - 100: 1MB - 10MB
  - 50: < 1MB
  - 0: Empty/symlink
- **B channel**: Access frequency/recency
  - 255: Accessed within 24h
  - 128: Accessed within week
  - 0: Rarely accessed
- **Alpha**: Always 255 (fully opaque)

### DIRECTORY_PIXEL
A container pixel representing a directory.

**Properties:**
- `type`: "directory"
- `position`: {x: int, y: int}
- `visual`: Border/outline color, larger than FILE_PIXEL
- `metadata`: Directory attributes
- `children`: Array of child pixel positions (relative coordinates)
- `bounds`: {width: int, height: int} - Space occupied by children

**Layout Strategy:**
- Children positioned relative to parent
- Grid layout: 16x16 pixels per directory by default
- Nested directories expand on zoom/click

### EXECUTABLE_PIXEL
A special FILE_PIXEL that can execute/boot.

**Properties:**
- All FILE_PIXEL properties plus:
- `executable_type`: "elf64", "script", "kernel", etc.
- `entry_point`: Offset for execution start
- `dependencies`: Array of required pixel hashes

**Visual Encoding:**
- Pulsing animation or distinct border
- Higher priority in rendering

## Coordinate System

### Infinite Map
- **Origin**: (0, 0) represents root `/`
- **Quadrants**: Signed 64-bit integers (±9,223,372,036,854,775,807)
- **Granularity**: Each coordinate = 1 pixel unit
- **Chunking**: Map divided into 256x256 chunks for efficient loading

### Spatial Allocation
```
/ (root at 0,0)
├─ /boot     → (0, -1000)
├─ /home     → (0, 1000)
├─ /etc      → (1000, 0)
├─ /usr      → (2000, 0)
├─ /var      → (0, 2000)
└─ /tmp      → (3000, 0)
```

Each top-level directory gets a 1000x1000 region.

## Data Storage

### Content-Addressable Store
```
data/
├─ objects/
│   ├─ ab/
│   │   └─ cdef1234...  (raw file bytes)
│   └─ 12/
│       └─ 3456abcd...
└─ index.db  (hash → metadata mapping)
```

**Hash Function**: SHA-256
**Deduplication**: Automatic - identical files share same hash
**Compression**: Optional gzip/zstd before storage

### Metadata Database
SQLite or LanceDB storing:
- Pixel position → hash mapping
- Spatial index for fast queries
- Directory structure
- File metadata

## File Operations

### Compression (File → Pixel)
```
1. Read file content
2. Compute SHA-256 hash
3. Check if hash exists in store
   - If yes: Reuse existing, only create new pixel metadata
   - If no: Store content in objects/
4. Extract metadata (size, type, permissions)
5. Compute visual encoding (RGB values)
6. Allocate spatial position
7. Create pixel record
```

### Decompression (Pixel → File)
```
1. Read pixel metadata
2. Retrieve data_hash
3. Lookup hash in object store
4. Read raw bytes
5. Verify hash integrity
6. Apply metadata (permissions, timestamps)
7. Reconstruct file
```

### Directory Listing
```
1. Query pixels with parent = directory_id
2. Render child pixels in relative positions
3. Support zoom levels (overview vs detailed)
```

## Pixel Format (JSON Schema)

```json
{
  "$schema": "pxos-pixel-v0.1",
  "type": "file|directory|executable",
  "id": "uuid-v4",
  "position": {
    "x": 0,
    "y": 0,
    "chunk_id": "0,0"
  },
  "visual": {
    "r": 255,
    "g": 128,
    "b": 64,
    "a": 255,
    "size": 1
  },
  "metadata": {
    "name": "example.txt",
    "path": "/home/user/example.txt",
    "size": 1048576,
    "permissions": "0644",
    "owner": 1000,
    "group": 1000,
    "created": "2025-11-16T00:00:00Z",
    "modified": "2025-11-16T12:00:00Z",
    "accessed": "2025-11-16T13:00:00Z",
    "mime_type": "text/plain"
  },
  "data": {
    "hash": "sha256:abcdef1234567890...",
    "backend": "local",
    "compressed": true,
    "compression": "zstd"
  },
  "children": [],
  "parent_id": "uuid-of-parent-directory"
}
```

## Boot Process

### Stage 1: Pixel Bootloader
```
1. Traditional BIOS/UEFI loads minimal bootloader
2. Bootloader initializes pixel map reader
3. Locate kernel EXECUTABLE_PIXEL at known coordinates
4. Decompress kernel from pixel
```

### Stage 2: Pixel Kernel Init
```
1. Kernel mounts pixel filesystem as root
2. Initialize spatial index
3. Load init system from /sbin/init pixel
4. Boot continues using pixel-based file access
```

### Stage 3: Runtime
```
- All file I/O intercepts and translates to pixel operations
- VFS layer bridges traditional syscalls to pixel map
- Lazy loading: Only decompress files when accessed
```

## Performance Considerations

### Caching
- LRU cache for recently accessed pixels
- Chunk-based loading (256x256 regions)
- Prefetch adjacent pixels on directory access

### Indexing
- Spatial R-tree for fast coordinate queries
- Hash index for content lookup
- B-tree for metadata searches

### Scalability
- Sparse storage: Only allocate chunks with pixels
- Distributed: Map can shard across multiple backends
- Compression: Dedupe + zstd = significant space savings

## Compatibility Layer

### FUSE Driver
Expose pixel filesystem as traditional directory tree:
```
/pxos/mount/
├─ home/
├─ etc/
└─ ...
```

### Translation
- File path → pixel coordinate lookup
- open() → pixel decompression
- write() → pixel update + rehash
- mkdir() → create DIRECTORY_PIXEL

## Future Extensions

- **Versioning**: Git-like object storage for file history
- **Encryption**: Encrypt data objects, pixels hold keys
- **Distribution**: P2P pixel sharing across machines
- **Compression**: Neural compression for extreme ratios
- **3D Maps**: Extend to (x, y, z) coordinates

## Implementation Checklist

- [ ] Pixel data structures
- [ ] Content-addressable storage
- [ ] Spatial indexing
- [ ] Compression/decompression
- [ ] Directory scanning
- [ ] Visual renderer
- [ ] FUSE mount
- [ ] Boot loader integration
- [ ] Performance testing
- [ ] Ubuntu system conversion

---

**Version**: 0.1
**Status**: Draft
**Last Updated**: 2025-11-16
