#!/usr/bin/env python3
"""
PixelFS - Pixel-based File System

Stores arbitrary data as RGB pixel images (.pxi format).
Each pixel stores 3 bytes (RGB), enabling compact binary storage.

File Format (.pxi):
- Header: 128 bytes metadata (magic, version, size, checksum)
- Data: RGB pixels packed with binary data (3 bytes per pixel)

Features:
- Memory-mapped access for large files
- SHA256 integrity verification
- Efficient storage (3 bytes per pixel)
- Supports visualization of data as images
"""

import struct
import hashlib
import mmap
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image


# Magic number for .pxi files (ASCII: "PXFS")
PXI_MAGIC = 0x50584653

# Current format version
PXI_VERSION = 1

# Header size (fixed 128 bytes)
PXI_HEADER_SIZE = 128


@dataclass
class PixelFileHeader:
    """Header metadata for .pxi files"""
    magic: int  # Magic number (0x50584653)
    version: int  # Format version
    data_size: int  # Original data size in bytes
    width: int  # Image width in pixels
    height: int  # Image height in pixels
    checksum: bytes  # SHA256 hash of data (32 bytes)
    reserved: bytes  # Reserved for future use

    def to_bytes(self) -> bytes:
        """Serialize header to bytes"""
        # Format: magic(4) version(4) data_size(8) width(4) height(4) checksum(32) reserved(72)
        header = struct.pack(
            '<IIQII32s72s',
            self.magic,
            self.version,
            self.data_size,
            self.width,
            self.height,
            self.checksum,
            self.reserved
        )
        assert len(header) == PXI_HEADER_SIZE
        return header

    @staticmethod
    def from_bytes(data: bytes) -> 'PixelFileHeader':
        """Deserialize header from bytes"""
        if len(data) < PXI_HEADER_SIZE:
            raise ValueError(f"Header too short: {len(data)} < {PXI_HEADER_SIZE}")

        magic, version, data_size, width, height, checksum, reserved = struct.unpack(
            '<IIQII32s72s',
            data[:PXI_HEADER_SIZE]
        )

        if magic != PXI_MAGIC:
            raise ValueError(f"Invalid magic number: 0x{magic:08X} (expected 0x{PXI_MAGIC:08X})")

        if version != PXI_VERSION:
            raise ValueError(f"Unsupported version: {version} (expected {PXI_VERSION})")

        return PixelFileHeader(
            magic=magic,
            version=version,
            data_size=data_size,
            width=width,
            height=height,
            checksum=checksum,
            reserved=reserved
        )


class PixelFS:
    """
    PixelFS - Stores binary data as RGB pixel images

    Usage:
        # Write data
        pfs = PixelFS()
        pfs.write("data.pxi", b"Hello, pixels!")

        # Read data
        data = pfs.read("data.pxi")
        print(data)  # b"Hello, pixels!"

        # Read with verification
        data = pfs.read("data.pxi", verify=True)
    """

    def __init__(self):
        """Initialize PixelFS"""
        pass

    def write(self, filepath: str, data: bytes, width: int = 1024, visualize: bool = False) -> Tuple[int, int]:
        """
        Write binary data to a .pxi file

        Args:
            filepath: Path to .pxi file
            data: Binary data to write
            width: Image width in pixels (height auto-calculated)
            visualize: If True, also save a .png visualization

        Returns:
            (width, height) of the created image
        """
        filepath = Path(filepath)

        # Calculate dimensions
        data_size = len(data)
        bytes_per_pixel = 3
        total_pixels = (data_size + bytes_per_pixel - 1) // bytes_per_pixel
        height = (total_pixels + width - 1) // width

        # Pad data to fill complete pixels
        padded_size = width * height * bytes_per_pixel
        padded_data = data + b'\x00' * (padded_size - data_size)

        # Calculate checksum
        checksum = hashlib.sha256(data).digest()

        # Create header
        header = PixelFileHeader(
            magic=PXI_MAGIC,
            version=PXI_VERSION,
            data_size=data_size,
            width=width,
            height=height,
            checksum=checksum,
            reserved=b'\x00' * 72
        )

        # Convert data to RGB image
        pixels = np.frombuffer(padded_data, dtype=np.uint8)
        pixels = pixels.reshape((height, width, 3))

        # Write .pxi file (header + image data)
        with open(filepath, 'wb') as f:
            f.write(header.to_bytes())
            # Write pixels as raw RGB data
            f.write(pixels.tobytes())

        # Optional: Save visualization as PNG
        if visualize:
            png_path = filepath.with_suffix('.png')
            img = Image.fromarray(pixels, mode='RGB')
            img.save(png_path)
            print(f"PixelFS: Saved visualization to {png_path}")

        print(f"PixelFS: Wrote {data_size:,} bytes as {width}x{height} pixel image")
        print(f"  File: {filepath}")
        if width * height > 0:
            efficiency = (data_size / (width * height * 3)) * 100
            print(f"  Efficiency: {efficiency:.1f}%")

        return width, height

    def read(self, filepath: str, verify: bool = False) -> bytes:
        """
        Read binary data from a .pxi file

        Args:
            filepath: Path to .pxi file
            verify: If True, verify checksum

        Returns:
            Original binary data

        Raises:
            ValueError: If header is invalid or checksum mismatch
        """
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            # Read header
            header_bytes = f.read(PXI_HEADER_SIZE)
            header = PixelFileHeader.from_bytes(header_bytes)

            # Read pixel data
            pixel_bytes = header.width * header.height * 3
            pixel_data = f.read(pixel_bytes)

            if len(pixel_data) != pixel_bytes:
                raise ValueError(f"Unexpected EOF: expected {pixel_bytes} bytes, got {len(pixel_data)}")

            # Extract original data (remove padding)
            data = pixel_data[:header.data_size]

            # Verify checksum if requested
            if verify:
                actual_checksum = hashlib.sha256(data).digest()
                if actual_checksum != header.checksum:
                    raise ValueError("Checksum mismatch: file may be corrupted")

            return data

    def read_mmap(self, filepath: str, verify: bool = False) -> bytes:
        """
        Read binary data using memory mapping (efficient for large files)

        Args:
            filepath: Path to .pxi file
            verify: If True, verify checksum

        Returns:
            Original binary data
        """
        filepath = Path(filepath)

        with open(filepath, 'r+b') as f:
            # Memory-map the file
            with mmap.mmap(f.fileno(), 0) as mm:
                # Read header
                header_bytes = mm[:PXI_HEADER_SIZE]
                header = PixelFileHeader.from_bytes(header_bytes)

                # Read data (without copying if possible)
                start = PXI_HEADER_SIZE
                end = start + header.data_size
                data = bytes(mm[start:end])

                # Verify checksum if requested
                if verify:
                    actual_checksum = hashlib.sha256(data).digest()
                    if actual_checksum != header.checksum:
                        raise ValueError("Checksum mismatch: file may be corrupted")

                return data

    def get_info(self, filepath: str) -> dict:
        """
        Get metadata about a .pxi file without reading all data

        Args:
            filepath: Path to .pxi file

        Returns:
            Dictionary with file metadata
        """
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            header_bytes = f.read(PXI_HEADER_SIZE)
            header = PixelFileHeader.from_bytes(header_bytes)

            return {
                'format': 'PXI',
                'version': header.version,
                'data_size': header.data_size,
                'width': header.width,
                'height': header.height,
                'total_pixels': header.width * header.height,
                'efficiency': (header.data_size / (header.width * header.height * 3)) * 100,
                'checksum': header.checksum.hex(),
                'file_size': filepath.stat().st_size
            }

    def visualize(self, filepath: str, output_path: Optional[str] = None) -> Image.Image:
        """
        Create a visual representation of the pixel data

        Args:
            filepath: Path to .pxi file
            output_path: Optional path to save PNG

        Returns:
            PIL Image object
        """
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            # Read header
            header_bytes = f.read(PXI_HEADER_SIZE)
            header = PixelFileHeader.from_bytes(header_bytes)

            # Read pixel data
            pixel_bytes = header.width * header.height * 3
            pixel_data = f.read(pixel_bytes)

            # Convert to image
            pixels = np.frombuffer(pixel_data, dtype=np.uint8)
            pixels = pixels.reshape((header.height, header.width, 3))
            img = Image.fromarray(pixels, mode='RGB')

            # Save if requested
            if output_path:
                img.save(output_path)
                print(f"PixelFS: Saved visualization to {output_path}")

            return img

    def batch_write(self, base_dir: str, files: dict, width: int = 1024) -> None:
        """
        Write multiple files efficiently

        Args:
            base_dir: Base directory for output
            files: Dict of {filename: data}
            width: Image width for all files
        """
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)

        for filename, data in files.items():
            filepath = base_path / filename
            self.write(str(filepath), data, width=width)

    def verify_integrity(self, filepath: str) -> bool:
        """
        Verify file integrity (checksum)

        Args:
            filepath: Path to .pxi file

        Returns:
            True if valid, False otherwise
        """
        try:
            self.read(filepath, verify=True)
            return True
        except ValueError:
            return False


def demo():
    """Demonstration of PixelFS capabilities"""
    import tempfile
    import shutil

    print("=" * 70)
    print("PixelFS Demonstration")
    print("=" * 70)
    print()

    # Create temporary directory
    tmpdir = tempfile.mkdtemp()
    print(f"Working directory: {tmpdir}\n")

    try:
        pfs = PixelFS()

        # Demo 1: Simple write/read
        print("Demo 1: Simple write and read")
        print("-" * 70)
        test_data = b"Hello, PixelFS! This is a test of pixel-based storage."
        pfs.write(f"{tmpdir}/hello.pxi", test_data, width=16, visualize=True)
        read_data = pfs.read(f"{tmpdir}/hello.pxi", verify=True)
        assert read_data == test_data
        print(f"✅ Data verified: {len(read_data)} bytes\n")

        # Demo 2: Large file
        print("Demo 2: Large file (1MB)")
        print("-" * 70)
        large_data = bytes(range(256)) * 4096  # 1MB repeating pattern
        pfs.write(f"{tmpdir}/large.pxi", large_data, width=1024)
        read_large = pfs.read_mmap(f"{tmpdir}/large.pxi", verify=True)
        assert read_large == large_data
        print(f"✅ Large file verified: {len(read_large):,} bytes\n")

        # Demo 3: File info
        print("Demo 3: File metadata")
        print("-" * 70)
        info = pfs.get_info(f"{tmpdir}/large.pxi")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()

        # Demo 4: Batch write
        print("Demo 4: Batch write")
        print("-" * 70)
        files = {
            'file1.pxi': b'Data for file 1',
            'file2.pxi': b'Data for file 2' * 100,
            'file3.pxi': b'Data for file 3' * 1000,
        }
        pfs.batch_write(f"{tmpdir}/batch", files, width=64)
        print(f"✅ Wrote {len(files)} files\n")

        # Demo 5: Integrity check
        print("Demo 5: Integrity verification")
        print("-" * 70)
        is_valid = pfs.verify_integrity(f"{tmpdir}/hello.pxi")
        print(f"  hello.pxi integrity: {'✅ VALID' if is_valid else '❌ INVALID'}")
        print()

        print("=" * 70)
        print("All demos completed successfully! ✅")
        print("=" * 70)

    finally:
        # Cleanup
        shutil.rmtree(tmpdir)
        print(f"\nCleaned up: {tmpdir}")


if __name__ == "__main__":
    demo()
