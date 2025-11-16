#!/usr/bin/env python3
"""
PixelFS: Pixel-Based File System

Stores arbitrary data as RGB pixel values, enabling:
- GPU-native data representation
- Spatial data layout
- Memory-mapped access to large files (like LLM weights)
- Visual inspection of data (it's literally pixels!)

Key Concepts:
    - Each pixel (RGB) stores 3 bytes of data
    - Files become 2D images
    - Large files (like GGUF models) are memory-mapped
    - Supports chunked loading for efficiency

Example:
    >>> fs = PixelFS("pixel_storage")
    >>> fs.write("model.pxi", model_bytes)
    >>> data = fs.read("model.pxi", offset=1024, length=4096)
"""

import numpy as np
import mmap
import struct
from pathlib import Path
from typing import Optional, Tuple, Union, BinaryIO
from dataclasses import dataclass
from PIL import Image
import hashlib


@dataclass
class PixelFileHeader:
    """
    Header for .pxi (Pixel eXternal Image) files

    Format:
        Magic: 4 bytes "PXIF" (Pixel eXternal Image File)
        Version: 2 bytes (major.minor)
        Original size: 8 bytes (uint64)
        Width: 4 bytes (uint32)
        Height: 4 bytes (uint32)
        Compression: 1 byte (0=none, 1=RLE, 2=LZ4)
        Checksum: 32 bytes (SHA256)
        Reserved: 15 bytes
        Total: 64 bytes
    """
    MAGIC = b'PXIF'
    HEADER_SIZE = 64
    VERSION = (1, 0)

    original_size: int
    width: int
    height: int
    compression: int = 0
    checksum: bytes = b''

    def pack(self) -> bytes:
        """Serialize header to bytes"""
        # Ensure checksum is 32 bytes
        checksum = self.checksum if len(self.checksum) == 32 else self.checksum.ljust(32, b'\x00')

        # Format: 4s BB Q I I B 32s 9s = 4+2+8+4+4+1+32+9 = 64 bytes
        header = struct.pack(
            '>4s BB Q I I B 32s 9s',  # > = big-endian
            self.MAGIC,
            self.VERSION[0],
            self.VERSION[1],
            self.original_size,
            self.width,
            self.height,
            self.compression,
            checksum,
            b'\x00' * 9  # reserved
        )
        assert len(header) == self.HEADER_SIZE, f"Header size mismatch: {len(header)} != {self.HEADER_SIZE}"
        return header

    @classmethod
    def unpack(cls, data: bytes) -> 'PixelFileHeader':
        """Deserialize header from bytes"""
        if len(data) < cls.HEADER_SIZE:
            raise ValueError(f"Header too short: {len(data)} < {cls.HEADER_SIZE}")

        parts = struct.unpack('>4s BB Q I I B 32s 9s', data[:cls.HEADER_SIZE])

        magic, ver_major, ver_minor, original_size, width, height, compression, checksum, _ = parts

        if magic != cls.MAGIC:
            raise ValueError(f"Invalid magic: {magic} != {cls.MAGIC}")

        return cls(
            original_size=original_size,
            width=width,
            height=height,
            compression=compression,
            checksum=checksum
        )


class PixelFS:
    """
    Pixel-based file system that stores data as RGB images.

    Features:
        - Store arbitrary binary data as pixels
        - Memory-mapped access for large files
        - Spatial layout control
        - Visual data inspection
        - Efficient chunked loading
    """

    BYTES_PER_PIXEL = 3  # RGB
    DEFAULT_WIDTH = 1024  # Default image width in pixels

    def __init__(self, root: Union[str, Path] = "pixel_storage"):
        """
        Initialize PixelFS.

        Args:
            root: Root directory for pixel storage
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache = {}  # Memory-mapped file cache

    def write(
        self,
        filename: str,
        data: bytes,
        width: Optional[int] = None,
        compression: int = 0,
        visualize: bool = False
    ) -> Path:
        """
        Write binary data as a pixel file.

        Args:
            filename: Output filename (will add .pxi extension)
            data: Binary data to store
            width: Image width in pixels (auto-calculated if None)
            compression: Compression mode (0=none)
            visualize: Also save a viewable PNG

        Returns:
            Path to created .pxi file
        """
        if not filename.endswith('.pxi'):
            filename += '.pxi'

        filepath = self.root / filename

        # Calculate dimensions
        data_size = len(data)
        width = width or self.DEFAULT_WIDTH

        # Calculate height needed (round up)
        pixels_needed = (data_size + self.BYTES_PER_PIXEL - 1) // self.BYTES_PER_PIXEL
        height = (pixels_needed + width - 1) // width

        # Pad data to fit image dimensions
        total_bytes = width * height * self.BYTES_PER_PIXEL
        padded_data = data + b'\x00' * (total_bytes - data_size)

        # Create pixel array (reshape as RGB image)
        pixel_array = np.frombuffer(padded_data, dtype=np.uint8)
        pixel_array = pixel_array.reshape((height, width, self.BYTES_PER_PIXEL))

        # Calculate checksum
        checksum = hashlib.sha256(data).digest()

        # Create header
        header = PixelFileHeader(
            original_size=data_size,
            width=width,
            height=height,
            compression=compression,
            checksum=checksum
        )

        # Write file
        with open(filepath, 'wb') as f:
            f.write(header.pack())
            f.write(pixel_array.tobytes())

        # Optional: save visualization
        if visualize:
            vis_path = filepath.with_suffix('.png')
            img = Image.fromarray(pixel_array, mode='RGB')
            img.save(vis_path)
            print(f"Visualization saved: {vis_path}")

        print(f"PixelFS: Wrote {data_size:,} bytes as {width}x{height} pixel image")
        print(f"  File: {filepath}")

        # Calculate efficiency (avoid division by zero for empty files)
        if width * height > 0:
            efficiency = (data_size / (width * height * 3)) * 100
            print(f"  Efficiency: {efficiency:.1f}%")
        else:
            print(f"  Efficiency: N/A (empty file)")

        return filepath

    def read(
        self,
        filename: str,
        offset: int = 0,
        length: Optional[int] = None,
        use_mmap: bool = True
    ) -> bytes:
        """
        Read data from pixel file.

        Args:
            filename: Pixel file to read
            offset: Byte offset to start reading
            length: Number of bytes to read (None = all)
            use_mmap: Use memory mapping for large files

        Returns:
            Binary data
        """
        if not filename.endswith('.pxi'):
            filename += '.pxi'

        filepath = self.root / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Pixel file not found: {filepath}")

        # Read header
        with open(filepath, 'rb') as f:
            header_bytes = f.read(PixelFileHeader.HEADER_SIZE)
            header = PixelFileHeader.unpack(header_bytes)

        # Determine read strategy
        file_size = filepath.stat().st_size
        use_mmap = use_mmap and file_size > 10 * 1024 * 1024  # Use mmap for files > 10MB

        if use_mmap:
            data = self._read_mmap(filepath, header, offset, length)
        else:
            data = self._read_direct(filepath, header, offset, length)

        return data

    def _read_direct(
        self,
        filepath: Path,
        header: PixelFileHeader,
        offset: int,
        length: Optional[int]
    ) -> bytes:
        """Direct file read (for small files)"""
        with open(filepath, 'rb') as f:
            # Skip header
            f.seek(PixelFileHeader.HEADER_SIZE)

            # Read pixel data
            pixel_data = f.read()

        # Extract original data (remove padding)
        data = pixel_data[:header.original_size]

        # Apply offset and length
        if length is None:
            return data[offset:]
        else:
            return data[offset:offset + length]

    def _read_mmap(
        self,
        filepath: Path,
        header: PixelFileHeader,
        offset: int,
        length: Optional[int]
    ) -> bytes:
        """Memory-mapped read (for large files)"""
        # Check cache
        cache_key = str(filepath)
        if cache_key not in self.cache:
            # Open and mmap file
            f = open(filepath, 'rb')
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.cache[cache_key] = (f, mm)

        _, mm = self.cache[cache_key]

        # Calculate pixel data region
        pixel_start = PixelFileHeader.HEADER_SIZE
        pixel_end = pixel_start + header.original_size

        # Read requested region
        start = pixel_start + offset
        if length is None:
            end = pixel_end
        else:
            end = min(pixel_start + offset + length, pixel_end)

        return bytes(mm[start:end])

    def get_info(self, filename: str) -> PixelFileHeader:
        """Get metadata about a pixel file"""
        if not filename.endswith('.pxi'):
            filename += '.pxi'

        filepath = self.root / filename

        with open(filepath, 'rb') as f:
            header_bytes = f.read(PixelFileHeader.HEADER_SIZE)
            return PixelFileHeader.unpack(header_bytes)

    def verify(self, filename: str) -> bool:
        """Verify file integrity using checksum"""
        data = self.read(filename)
        info = self.get_info(filename)

        actual_checksum = hashlib.sha256(data).digest()
        return actual_checksum == info.checksum

    def list_files(self) -> list[Path]:
        """List all pixel files"""
        return list(self.root.glob("*.pxi"))

    def close(self):
        """Close all memory-mapped files"""
        for f, mm in self.cache.values():
            mm.close()
            f.close()
        self.cache.clear()

    def __del__(self):
        """Cleanup on deletion"""
        self.close()


# Visualization utilities
def visualize_pixel_file(pixelfs: PixelFS, filename: str, output: Optional[str] = None):
    """
    Create a visualization of a pixel file.

    This shows what the data "looks like" as an image.
    """
    info = pixelfs.get_info(filename)

    filepath = pixelfs.root / (filename if filename.endswith('.pxi') else filename + '.pxi')

    with open(filepath, 'rb') as f:
        f.seek(PixelFileHeader.HEADER_SIZE)
        pixel_data = f.read()

    # Reshape as image
    pixel_array = np.frombuffer(pixel_data[:info.width * info.height * 3], dtype=np.uint8)
    pixel_array = pixel_array.reshape((info.height, info.width, 3))

    # Create image
    img = Image.fromarray(pixel_array, mode='RGB')

    if output:
        img.save(output)
        print(f"Visualization saved: {output}")

    return img


# CLI for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("PixelFS Demo")
        print("\nUsage:")
        print("  python pixelfs.py write <file> [output_name]")
        print("  python pixelfs.py read <pixel_file>")
        print("  python pixelfs.py info <pixel_file>")
        print("  python pixelfs.py verify <pixel_file>")
        print("  python pixelfs.py list")
        print("  python pixelfs.py demo")
        sys.exit(1)

    cmd = sys.argv[1]
    fs = PixelFS()

    if cmd == "demo":
        # Demo: Store some text
        print("\n=== PixelFS Demo ===\n")

        # Create test data
        test_data = b"Hello, Pixel World! " * 100
        test_data += b"\nThis is data stored as pixels.\n" * 50

        print(f"Demo data: {len(test_data)} bytes")

        # Write
        filepath = fs.write("demo", test_data, visualize=True)

        # Read back
        read_data = fs.read("demo")

        # Verify
        is_valid = fs.verify("demo")
        print(f"\nVerification: {'✓ PASS' if is_valid else '✗ FAIL'}")
        print(f"Data matches: {test_data == read_data}")

        # Info
        info = fs.get_info("demo")
        print(f"\nFile info:")
        print(f"  Original size: {info.original_size:,} bytes")
        print(f"  Dimensions: {info.width}x{info.height} pixels")
        print(f"  Compression: {info.compression}")

    elif cmd == "write":
        input_file = sys.argv[2]
        output_name = sys.argv[3] if len(sys.argv) > 3 else Path(input_file).stem

        with open(input_file, 'rb') as f:
            data = f.read()

        fs.write(output_name, data, visualize=True)

    elif cmd == "read":
        pixel_file = sys.argv[2]
        data = fs.read(pixel_file)

        # Try to decode as text
        try:
            text = data.decode('utf-8')
            print(text[:500])
            if len(data) > 500:
                print(f"\n... ({len(data) - 500} more bytes)")
        except:
            print(f"Binary data: {len(data)} bytes")
            print(data[:100].hex())

    elif cmd == "info":
        pixel_file = sys.argv[2]
        info = fs.get_info(pixel_file)

        print(f"\nPixel File Info: {pixel_file}")
        print(f"  Original size: {info.original_size:,} bytes")
        print(f"  Dimensions: {info.width}x{info.height} pixels")
        print(f"  Compression: {info.compression}")
        print(f"  Checksum: {info.checksum.hex()[:16]}...")

    elif cmd == "verify":
        pixel_file = sys.argv[2]
        is_valid = fs.verify(pixel_file)
        print(f"Verification: {'✓ PASS' if is_valid else '✗ FAIL'}")

    elif cmd == "list":
        files = fs.list_files()
        print(f"\nPixel Files ({len(files)}):")
        for f in files:
            info = fs.get_info(f.name)
            size_mb = info.original_size / 1024 / 1024
            print(f"  {f.name:30s} {size_mb:8.2f} MB  {info.width}x{info.height}")
