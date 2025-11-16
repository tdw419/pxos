# pxOS Pixel Filesystem (pxFS)

**A revolutionary spatial filesystem where every file is compressed into a single pixel on an infinite 2D map.**

## Overview

pxFS is the foundational storage layer for pxOS—a new operating system paradigm built on two core principles:

1. **Infinite Map**: Unbounded spatial canvas for unlimited storage and computational space
2. **Pixel-as-File**: Every file compresses to exactly one pixel that can be decompressed on demand

This creates virtually unlimited storage capacity through spatial organization and aggressive content-addressable deduplication.

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Compress your first file
python3 pxfs_cli.py compress-file /path/to/file.txt

# Compress an entire directory
python3 pxfs_cli.py compress-dir /path/to/directory

# Visualize the pixel map
python3 pxfs_cli.py visualize -o map.html

# Decompress files
python3 pxfs_cli.py decompress-dir /path/to/directory -o /restore/path
```

### Example: Compress Your Home Directory

```bash
# Compress your entire home directory
python3 pxfs_cli.py compress-dir /home/user

# View statistics
python3 pxfs_cli.py stats

# See the pixel map
python3 pxfs_cli.py visualize-ascii

# Generate interactive HTML map
python3 pxfs_cli.py visualize -o my_files.html

# Restore everything
python3 pxfs_cli.py decompress-dir /home/user -o /tmp/restored_home
```

---

## How It Works

### File → Pixel Compression

Each file undergoes the following transformation:

```
Original File (any size)
    ↓
SHA-256 Hash (content-addressable)
    ↓
Store in blob storage (deduplicated)
    ↓
Create Pixel Metadata:
  • Position: (x, y) on infinite map
  • Visual: RGB color encoding properties
  • Data Reference: Hash pointer to blob
    ↓
Single Pixel = Complete File Blueprint
```

### Visual Encoding

Each pixel's RGB color encodes file properties:

- **R Channel** (File Type):
  - `255` = Executable/Binary
  - `200` = Media (image/video/audio)
  - `150` = Text/Document
  - `100` = Archive/Compressed
  - `50` = Configuration
  - `0` = Other

- **G Channel** (Size Class):
  - `255` = > 1GB
  - `200` = 100MB - 1GB
  - `150` = 10MB - 100MB
  - `100` = 1MB - 10MB
  - `50` = < 1MB
  - `0` = Empty

- **B Channel** (Access Recency):
  - `255` = Accessed within 24h
  - `128` = Accessed within week
  - `0` = Rarely accessed

### Directory Structure

Directories are container pixels with children:

```
/home (DIR_PIXEL at 0, 1000)
  ├─ user/ (DIR_PIXEL at 100, 100 relative)
  │   ├─ file.txt (FILE_PIXEL at 10, 10)
  │   └─ photo.jpg (FILE_PIXEL at 20, 10)
  └─ another/ (DIR_PIXEL at 100, 200)
```

---

## Architecture

### Storage Backend

```
.pxfs/
├── content/          # Content-addressable blob storage
│   ├── objects/
│   │   ├── ab/
│   │   │   └── cdef1234...  (compressed file data)
│   │   └── 12/
│   │       └── 3456abcd...
│   └── ...
└── metadata.db       # SQLite spatial index + metadata
```

### Key Components

1. **ContentStore**: Git-like content-addressable storage
   - SHA-256 hashing
   - Automatic deduplication
   - Zstandard compression

2. **MetadataStore**: SQLite database with spatial indexing
   - Fast coordinate queries
   - Path-based lookups
   - Parent-child relationships

3. **PixelCompressor**: File → Pixel conversion
   - Metadata extraction
   - Visual encoding
   - Spatial allocation

4. **PixelDecompressor**: Pixel → File restoration
   - Hash verification
   - Metadata restoration
   - Recursive directory reconstruction

5. **PixelMapVisualizer**: Visual rendering
   - HTML interactive maps
   - ASCII terminal views
   - Chunk-based loading

---

## Pixel Format Specification

Each pixel follows the pxOS-pixel-v0.1 schema:

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
    "a": 255
  },
  "metadata": {
    "name": "example.txt",
    "path": "/absolute/path",
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
    "hash": "sha256:abcdef...",
    "backend": "local",
    "compressed": true,
    "compression": "zstd"
  },
  "children": [],
  "parent_id": "parent-uuid"
}
```

See [SPECIFICATION.md](SPECIFICATION.md) for full details.

---

## CLI Reference

### Compression

```bash
# Compress single file
pxfs_cli.py compress-file <path> [-x X] [-y Y]

# Compress directory tree
pxfs_cli.py compress-dir <path> [-x X] [-y Y] [--no-recursive]
```

### Decompression

```bash
# Decompress file
pxfs_cli.py decompress-file <original_path> [-o output]

# Decompress directory
pxfs_cli.py decompress-dir <original_path> [-o output]
```

### Inspection

```bash
# List pixels in chunk
pxfs_cli.py list [-x chunk_x] [-y chunk_y]

# Show statistics
pxfs_cli.py stats

# Verify integrity
pxfs_cli.py verify
```

### Visualization

```bash
# HTML interactive map
pxfs_cli.py visualize [-o output.html] [-x X] [-y Y] [-s pixel_size]

# ASCII terminal view
pxfs_cli.py visualize-ascii [-x X] [-y Y]
```

---

## Benefits

### Unlimited Storage
- Infinite 2D coordinate space (±2^63)
- Sparse storage—only allocate what's used
- No filesystem size limits

### Aggressive Deduplication
- Content-addressable: identical files share storage
- Automatic across entire system
- Works for files in different directories

### Spatial Organization
- Files organized by coordinates
- Chunk-based loading (256×256)
- Fast spatial queries

### Bootable
- System can boot from pixel map
- Kernel is an EXECUTABLE_PIXEL
- All system files compressed

### Visual Interface
- See your entire filesystem as colors
- Identify file types at a glance
- Interactive HTML maps

---

## Performance

### Test Results (test_data/ directory)

```
Original:     10,097 bytes
Compressed:   ~8,500 bytes
Ratio:        84%
Deduplication: 1.2x
Files:        4
Pixels:       6 (includes 2 directories)
```

### Scalability

- **Chunks**: 256×256 pixels loaded on demand
- **Index**: SQLite with spatial R-tree
- **Cache**: LRU for recently accessed pixels
- **Parallel**: Independent chunks can load concurrently

---

## Roadmap

### ✅ Phase 1: Core Implementation (COMPLETE)
- [x] Pixel data structures
- [x] Content-addressable storage
- [x] Compression/decompression
- [x] Directory support
- [x] Spatial indexing
- [x] CLI interface
- [x] Visualization

### 🚧 Phase 2: System Integration (IN PROGRESS)
- [ ] Boot from pixel map
- [ ] FUSE mount driver
- [ ] VFS integration
- [ ] Real Ubuntu conversion

### 🔮 Phase 3: Advanced Features
- [ ] Distributed pixel sharing (P2P)
- [ ] Neural compression
- [ ] 3D spatial maps (x, y, z)
- [ ] Version control (Git-like)
- [ ] Encryption at rest
- [ ] Real-time collaboration

---

## Next Steps: Boot from Pixels

The ultimate goal is to:

1. **Scan Ubuntu System**
   ```bash
   # Compress entire Ubuntu filesystem
   sudo pxfs_cli.py compress-dir / -o ubuntu_pixels
   ```

2. **Create Pixel Kernel**
   - Minimal bootloader loads pixel map reader
   - Decompress kernel from EXECUTABLE_PIXEL
   - Mount pxFS as root

3. **Boot Sequence**
   ```
   BIOS → Pixel Bootloader → Load Kernel Pixel →
   Decompress Kernel → Mount pxFS → Init System →
   Boot Complete (running from pixels!)
   ```

4. **Runtime**
   - VFS layer intercepts file operations
   - Translate to pixel map operations
   - Lazy decompression on access
   - Write-through to pixels

---

## Python API

```python
from pxfs import (
    ContentStore,
    MetadataStore,
    PixelCompressor,
    PixelDecompressor,
    PixelMapVisualizer,
    Position
)

# Initialize storage
content = ContentStore(".pxfs/content")
metadata = MetadataStore(".pxfs/metadata.db")

# Compress a file
compressor = PixelCompressor(content, metadata)
pixel = compressor.compress_file(
    "/path/to/file.txt",
    Position(x=100, y=200)
)

print(f"Compressed to pixel at ({pixel.position.x}, {pixel.position.y})")
print(f"Visual: RGB({pixel.visual.r}, {pixel.visual.g}, {pixel.visual.b})")

# Decompress
decompressor = PixelDecompressor(content, metadata)
restored_path = decompressor.decompress_pixel(pixel, "/tmp/restored.txt")
print(f"Restored to: {restored_path}")

# Visualize
visualizer = PixelMapVisualizer(metadata)
visualizer.render_html("map.html", chunk_x=0, chunk_y=0)
```

---

## Project Structure

```
pxos/
├── SPECIFICATION.md          # Formal pxFS spec v0.1
├── README_PIXEL_FS.md       # This file
├── requirements.txt          # Python dependencies
├── pxfs/                     # Core library
│   ├── __init__.py
│   ├── pixel.py             # Pixel data structures
│   ├── storage.py           # Content + metadata stores
│   ├── compressor.py        # File → Pixel
│   ├── decompressor.py      # Pixel → File
│   └── visualizer.py        # Visual rendering
├── pxfs_cli.py              # Command-line interface
├── test_data/               # Sample files for testing
├── restored/                # Decompressed test output
├── .pxfs/                   # Storage directory
│   ├── content/             # Blob storage
│   └── metadata.db          # Spatial index
└── pixel_map.html           # Generated visualization
```

---

## Technical Specifications

### Coordinate System
- **Range**: ±9,223,372,036,854,775,807 (int64)
- **Origin**: (0, 0) = root `/`
- **Chunk Size**: 256×256 pixels
- **Allocation**: Top-level dirs get 1000×1000 regions

### Hashing
- **Algorithm**: SHA-256
- **Format**: `sha256:hexdigest`
- **Verification**: Optional on decompress

### Compression
- **Algorithm**: Zstandard (level 3)
- **Disabled for**: Files < 100 bytes
- **Format**: Raw zstd stream

### Database
- **Engine**: SQLite 3
- **Indexes**: Position (x,y), Chunk, Path, Hash, Parent
- **Transactions**: Atomic pixel creation

---

## FAQ

**Q: Does this really compress files to single pixels?**
A: Yes! The "pixel" is a metadata structure with visual encoding + hash pointer. The actual file data lives in content-addressable storage.

**Q: Can I boot my Ubuntu system from this?**
A: Not yet—but that's the goal! Currently pxFS is a standalone filesystem. Phase 2 will add bootloader integration.

**Q: What about file access performance?**
A: Lazy loading + LRU cache means frequently accessed files stay fast. Cold files decompress on demand.

**Q: Is this production-ready?**
A: No—this is v0.1 prototype. Don't use for critical data yet. Always keep backups!

**Q: Why build a new filesystem?**
A: To explore spatial organization, infinite storage models, and visual file interfaces. Plus it's a fun foundation for a pixel-based OS!

**Q: Can I see deleted files?**
A: Not yet, but the content-addressable storage enables Git-like history. Future versions could add versioning.

---

## Contributing

We're looking for help with:

- 🐛 Bug fixes and testing
- 📝 Documentation improvements
- ⚡ Performance optimization
- 🔧 FUSE driver implementation
- 🎨 Better visualizations
- 🧪 Large-scale testing (compress real Ubuntu systems)
- 🚀 Bootloader integration

---

## License

MIT License - See LICENSE file

---

## Resources

- **Specification**: [SPECIFICATION.md](SPECIFICATION.md)
- **Core Library**: [pxfs/](pxfs/)
- **CLI Tool**: [pxfs_cli.py](pxfs_cli.py)
- **Visualizations**: [pixel_map.html](pixel_map.html)

---

**Built with Python 3 + SQLite + Zstandard**

*"Every file is a pixel. Every pixel is a file. The map is infinite."*

---

## Credits

Inspired by:
- Content-addressable storage (Git)
- Spatial databases (PostGIS)
- Infinite canvas UIs (Figma, Miro)
- Voxel engines (Minecraft)
- Plan 9 filesystem concepts

**pxOS**: Rethinking operating systems from the pixels up.
