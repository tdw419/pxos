#!/usr/bin/env python3
"""
Pixel Archive - Single-File Repository Storage

Creates a single .pxa (pixel archive) file containing the entire codebase.

THIS IS THE EVOLUTION:
  Before: Many .pxi files + manifest.json
  After:  ONE .pxa file with everything inside

Like a pixel-native TAR/ZIP, but substrate-native.
One cartridge. One file. Everything.

Archive Format (.pxa):
┌─────────────────────────────────────┐
│ Archive Header (128 bytes)          │
│  - Magic: "PXAR"                    │
│  - Version, file count, sizes       │
├─────────────────────────────────────┤
│ File Index (directory structure)    │
│  - File paths, offsets, sizes       │
├─────────────────────────────────────┤
│ File Data (concatenated)            │
│  - All files back-to-back           │
└─────────────────────────────────────┘

Philosophy:
"One pixel file to rule them all.
 One cartridge. One truth.
 Everything you need in one place."
"""

import struct
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

# Bootstrap
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pixel_llm.core.pixelfs import PixelFS


# Archive constants
ARCHIVE_MAGIC = b'PXAR'
ARCHIVE_VERSION = 1
HEADER_SIZE = 128


class PixelArchive:
    """
    Pixel Archive - stores multiple files in a single pixel image.

    This is like tar/zip but pixel-native:
    - One .pxa file contains entire codebase
    - Directory structure preserved
    - Can be extracted or mounted
    - Memory-mapped for efficiency
    - True cartridge model
    """

    def __init__(self):
        self.files: List[Tuple[str, bytes]] = []  # (path, data)

    def add_file(self, path: str, data: bytes):
        """Add a file to the archive"""
        self.files.append((path, data))

    def add_file_from_disk(self, disk_path: Path, archive_path: str):
        """Add a file from disk with a custom archive path"""
        data = disk_path.read_bytes()
        self.add_file(archive_path, data)

    def pack(self) -> bytes:
        """
        Pack all files into a single archive byte stream.

        Returns:
            Archive bytes ready to write to .pxa
        """
        # Calculate index size
        index_data = bytearray()

        # File entries in index
        file_entries = []
        current_offset = HEADER_SIZE

        # First pass: build index
        for path, data in self.files:
            path_bytes = path.encode('utf-8')
            path_len = len(path_bytes)
            file_size = len(data)

            # Calculate offset for index
            index_entry_size = 2 + path_len + 8 + 8 + 16  # path_len + path + offset + size + reserved
            current_offset += index_entry_size

            file_entries.append({
                'path': path,
                'path_bytes': path_bytes,
                'path_len': path_len,
                'data': data,
                'size': file_size,
            })

        # current_offset is now past the index
        data_start_offset = current_offset

        # Second pass: assign data offsets
        current_data_offset = data_start_offset
        for entry in file_entries:
            entry['offset'] = current_data_offset
            current_data_offset += entry['size']

        # Build index
        for entry in file_entries:
            # Path length (uint16)
            index_data.extend(struct.pack('<H', entry['path_len']))
            # Path (UTF-8)
            index_data.extend(entry['path_bytes'])
            # Offset (uint64)
            index_data.extend(struct.pack('<Q', entry['offset']))
            # Size (uint64)
            index_data.extend(struct.pack('<Q', entry['size']))
            # Reserved (16 bytes)
            index_data.extend(b'\x00' * 16)

        # Build header
        header = bytearray(HEADER_SIZE)

        # Magic
        header[0:4] = ARCHIVE_MAGIC

        # Version (uint32)
        struct.pack_into('<I', header, 4, ARCHIVE_VERSION)

        # Number of files (uint32)
        struct.pack_into('<I', header, 8, len(self.files))

        # Index offset (uint64) - right after header
        struct.pack_into('<Q', header, 12, HEADER_SIZE)

        # Index size (uint64)
        struct.pack_into('<Q', header, 20, len(index_data))

        # Total archive size (uint64)
        total_size = HEADER_SIZE + len(index_data) + sum(entry['size'] for entry in file_entries)
        struct.pack_into('<Q', header, 28, total_size)

        # Assemble archive
        archive = bytearray()
        archive.extend(header)
        archive.extend(index_data)

        # Append all file data
        for entry in file_entries:
            archive.extend(entry['data'])

        return bytes(archive)

    def save(self, output_path: str):
        """Pack and save archive to a pixel image"""
        archive_bytes = self.pack()

        # Write to pixel image via PixelFS
        fs = PixelFS()
        fs.write(output_path, archive_bytes)

        print(f"✅ Saved pixel archive: {output_path}")
        print(f"   Files: {len(self.files)}")
        print(f"   Size: {len(archive_bytes):,} bytes")


class PixelArchiveReader:
    """
    Read files from a pixel archive.

    Can extract files or provide direct access (virtual filesystem).
    """

    def __init__(self, archive_path: str):
        self.archive_path = archive_path
        self.fs = PixelFS()

        # Load archive
        self.archive_bytes = self.fs.read(archive_path)

        # Parse header
        self._parse_header()

        # Parse index
        self._parse_index()

    def _parse_header(self):
        """Parse archive header"""
        if len(self.archive_bytes) < HEADER_SIZE:
            raise ValueError(f"Archive too small: {len(self.archive_bytes)} < {HEADER_SIZE}")

        # Check magic
        magic = self.archive_bytes[0:4]
        if magic != ARCHIVE_MAGIC:
            raise ValueError(f"Invalid archive magic: {magic} != {ARCHIVE_MAGIC}")

        # Version
        self.version = struct.unpack('<I', self.archive_bytes[4:8])[0]

        # Number of files
        self.num_files = struct.unpack('<I', self.archive_bytes[8:12])[0]

        # Index offset
        self.index_offset = struct.unpack('<Q', self.archive_bytes[12:20])[0]

        # Index size
        self.index_size = struct.unpack('<Q', self.archive_bytes[20:28])[0]

        # Total size
        self.total_size = struct.unpack('<Q', self.archive_bytes[28:36])[0]

    def _parse_index(self):
        """Parse file index"""
        self.files: Dict[str, Dict] = {}

        offset = self.index_offset
        for _ in range(self.num_files):
            # Path length
            path_len = struct.unpack('<H', self.archive_bytes[offset:offset+2])[0]
            offset += 2

            # Path
            path_bytes = self.archive_bytes[offset:offset+path_len]
            path = path_bytes.decode('utf-8')
            offset += path_len

            # Data offset
            data_offset = struct.unpack('<Q', self.archive_bytes[offset:offset+8])[0]
            offset += 8

            # Size
            size = struct.unpack('<Q', self.archive_bytes[offset:offset+8])[0]
            offset += 8

            # Reserved (skip)
            offset += 16

            self.files[path] = {
                'offset': data_offset,
                'size': size,
            }

    def list_files(self) -> List[str]:
        """List all files in archive"""
        return sorted(self.files.keys())

    def read_file(self, path: str) -> bytes:
        """Read a file from the archive"""
        if path not in self.files:
            raise FileNotFoundError(f"File not in archive: {path}")

        entry = self.files[path]
        offset = entry['offset']
        size = entry['size']

        return self.archive_bytes[offset:offset+size]

    def extract_all(self, output_dir: Path):
        """Extract all files to a directory"""
        output_dir.mkdir(parents=True, exist_ok=True)

        for path in self.files:
            file_data = self.read_file(path)

            # Create parent directories
            file_path = output_dir / path
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            file_path.write_bytes(file_data)

            print(f"  Extracted: {path} ({len(file_data):,} bytes)")

        print(f"\n✅ Extracted {len(self.files)} files to {output_dir}")


def main():
    """CLI for pixel archive tool"""
    if len(sys.argv) < 2:
        print("Pixel Archive Tool")
        print()
        print("Usage:")
        print("  python3 pixel_llm/core/pixel_archive.py pack <output.pxa> <files...>")
        print("  python3 pixel_llm/core/pixel_archive.py list <archive.pxa>")
        print("  python3 pixel_llm/core/pixel_archive.py extract <archive.pxa> <output_dir>")
        print("  python3 pixel_llm/core/pixel_archive.py read <archive.pxa> <file_path>")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'list':
        archive_path = sys.argv[2]
        reader = PixelArchiveReader(archive_path)

        print(f"Archive: {archive_path}")
        print(f"Files: {reader.num_files}")
        print(f"Size: {reader.total_size:,} bytes")
        print()
        print("Contents:")
        for path in reader.list_files():
            entry = reader.files[path]
            print(f"  {path:<60} {entry['size']:>10,} bytes")

    elif command == 'extract':
        archive_path = sys.argv[2]
        output_dir = Path(sys.argv[3])

        reader = PixelArchiveReader(archive_path)
        reader.extract_all(output_dir)

    elif command == 'read':
        archive_path = sys.argv[2]
        file_path = sys.argv[3]

        reader = PixelArchiveReader(archive_path)
        data = reader.read_file(file_path)

        # Try to decode as text
        try:
            text = data.decode('utf-8')
            print(text)
        except UnicodeDecodeError:
            print(f"Binary file ({len(data)} bytes)")
            print(data.hex()[:200] + "...")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
