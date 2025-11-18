#!/usr/bin/env python3
"""
PXCARTRIDGE Maker v0.1

Convert binary files into pixel cartridge format (.pxcart.png)

This tool creates standardized pixel-based containers for foreign binaries,
enabling systematic porting to pxOS.

Usage:
    make_pxcart.py input.bin --isa x86_32 --abi raw_bin --entry 0x7C00 -o output.pxcart.png
"""

import argparse
import struct
import hashlib
import zlib
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: This tool requires PIL (Pillow) and numpy")
    print("Install with: pip install pillow numpy")
    exit(1)


# ============================================================================
# Constants
# ============================================================================

MAGIC = b"PXCT"
VERSION = b"0001"
HEADER_ROWS = 16
WIDTH_OPTIONS = [256, 512, 1024]

# ISA identifiers
ISA_TYPES = {
    "x86_32", "x86_64", "arm32", "arm64", "mips",
    "riscv32", "riscv64", "wasm32", "unknown"
}

# ABI identifiers
ABI_TYPES = {
    "elf_linux", "pe_win32", "pe_win64", "macho",
    "raw_bin", "dos_com", "dos_exe",
    "pxos_prim", "pxos_native"
}

# Compression types
COMPRESSION_TYPES = {
    "none", "zlib", "gzip", "lz4", "lzma"
}

# Flags
FLAG_COMPRESSED = 0x01
FLAG_HAS_IR = 0x02
FLAG_PARTIAL_PORT = 0x04

# Porting status (bits 8-15)
STATUS_RAW = 0x00
STATUS_ANALYZED = 0x01
STATUS_LIFTED = 0x02
STATUS_PORTED = 0x03
STATUS_TESTED = 0x04


# ============================================================================
# Cartridge Metadata
# ============================================================================

@dataclass
class CartridgeMetadata:
    """Metadata for a pixel cartridge"""
    isa: str
    abi: str
    entry_point: int
    binary_size: int
    compressed: bool = False
    compression_type: str = "none"
    has_ir: bool = False
    partial_port: bool = False
    status: int = STATUS_RAW
    dependencies: List[str] = None
    license: str = ""
    author: str = ""

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

        # Validate
        if self.isa not in ISA_TYPES:
            raise ValueError(f"Invalid ISA: {self.isa}. Must be one of {ISA_TYPES}")
        if self.abi not in ABI_TYPES:
            raise ValueError(f"Invalid ABI: {self.abi}. Must be one of {ABI_TYPES}")
        if self.compression_type not in COMPRESSION_TYPES:
            raise ValueError(f"Invalid compression: {self.compression_type}")

    def get_flags(self) -> int:
        """Calculate flags bitfield"""
        flags = 0
        if self.compressed:
            flags |= FLAG_COMPRESSED
        if self.has_ir:
            flags |= FLAG_HAS_IR
        if self.partial_port:
            flags |= FLAG_PARTIAL_PORT
        flags |= (self.status << 8)
        return flags


# ============================================================================
# Pixel Encoding
# ============================================================================

def encode_string(s: str, max_bytes: int) -> bytes:
    """Encode string as null-terminated UTF-8, padded to max_bytes"""
    encoded = s.encode('utf-8')[:max_bytes-1]
    return encoded + b'\0' + b'\0' * (max_bytes - len(encoded) - 1)


def encode_u32(value: int) -> bytes:
    """Encode 32-bit unsigned int as little-endian"""
    return struct.pack('<I', value)


def encode_u64(value: int) -> bytes:
    """Encode 64-bit unsigned int as little-endian"""
    return struct.pack('<Q', value)


def bytes_to_pixels(data: bytes, width: int) -> np.ndarray:
    """Convert bytes to RGBA pixel array"""
    # Pad to multiple of 4
    padded_size = (len(data) + 3) // 4 * 4
    padded = data + b'\0' * (padded_size - len(data))

    # Reshape to RGBA pixels
    pixels_count = len(padded) // 4
    rgba = np.frombuffer(padded, dtype=np.uint8).reshape(pixels_count, 4)

    # Calculate height
    height = (pixels_count + width - 1) // width

    # Pad to full image
    total_pixels = width * height
    if pixels_count < total_pixels:
        padding = np.zeros((total_pixels - pixels_count, 4), dtype=np.uint8)
        rgba = np.vstack([rgba, padding])

    # Reshape to image
    return rgba.reshape(height, width, 4)


# ============================================================================
# Cartridge Builder
# ============================================================================

class CartridgeBuilder:
    """Build a pixel cartridge image"""

    def __init__(self, binary_path: Path, metadata: CartridgeMetadata, width: int = 512):
        self.binary_path = binary_path
        self.metadata = metadata
        self.width = width

        if width not in WIDTH_OPTIONS:
            raise ValueError(f"Width must be one of {WIDTH_OPTIONS}")

        # Load binary
        self.binary_data = binary_path.read_bytes()

        # Compress if requested
        if metadata.compressed:
            if metadata.compression_type == "zlib":
                self.payload = zlib.compress(self.binary_data)
            else:
                # For now, only zlib is implemented
                self.payload = self.binary_data
        else:
            self.payload = self.binary_data

        # Update metadata with actual binary size
        self.metadata.binary_size = len(self.binary_data)

    def build_header(self) -> bytes:
        """Build header bytes (16 rows × width × 4 bytes)"""
        header_size = HEADER_ROWS * self.width * 4
        header = bytearray(header_size)

        offset = 0

        # Row 0: Magic & Version
        header[offset:offset+4] = MAGIC
        offset += 4
        header[offset:offset+4] = VERSION
        offset += 4
        # Rest of row 0: reserved/padding
        offset = self.width * 4

        # Row 1: ISA & ABI
        header[offset:offset+32] = encode_string(self.metadata.isa, 32)
        offset += 32
        header[offset:offset+32] = encode_string(self.metadata.abi, 32)
        offset += 32
        # Rest of row 1: padding
        offset = 2 * self.width * 4

        # Row 2: Entry point & Size
        header[offset:offset+8] = encode_u64(self.metadata.entry_point)
        offset += 8
        header[offset:offset+8] = encode_u64(self.metadata.binary_size)
        offset += 8
        offset = 3 * self.width * 4

        # Row 3: Flags & Compression
        flags = self.metadata.get_flags()
        header[offset:offset+4] = encode_u32(flags)
        offset += 4
        header[offset:offset+16] = encode_string(self.metadata.compression_type, 16)
        offset += 16
        offset = 4 * self.width * 4

        # Row 4-7: Dependencies (comma-separated)
        deps_str = ",".join(self.metadata.dependencies)
        deps_bytes = encode_string(deps_str, 4 * self.width * 4)
        header[offset:offset+len(deps_bytes)] = deps_bytes
        offset = 8 * self.width * 4

        # Row 8-11: License & Author
        license_bytes = encode_string(self.metadata.license, 2 * self.width * 4)
        header[offset:offset+len(license_bytes)] = license_bytes
        offset += len(license_bytes)
        author_bytes = encode_string(self.metadata.author, 2 * self.width * 4)
        header[offset:offset+len(author_bytes)] = author_bytes
        offset = 12 * self.width * 4

        # Row 12-15: Checksums
        payload_hash = hashlib.sha256(self.payload).digest()
        header[offset:offset+32] = payload_hash
        offset += 32

        # CRC32 of header (calculated after header is built)
        header_crc = zlib.crc32(bytes(header[:offset]))
        header[offset:offset+4] = encode_u32(header_crc)

        return bytes(header)

    def build(self) -> Image.Image:
        """Build complete cartridge image"""
        # Build header
        header_bytes = self.build_header()
        header_img = bytes_to_pixels(header_bytes, self.width)

        # Build payload
        payload_img = bytes_to_pixels(self.payload, self.width)

        # Build checksum row
        checksum_row = np.zeros((1, self.width, 4), dtype=np.uint8)
        # Full image CRC will be calculated after assembly
        # For now, leave as zeros

        # Assemble
        full_img = np.vstack([header_img, payload_img, checksum_row])

        # Create PIL image
        img = Image.fromarray(full_img, mode='RGBA')

        # Calculate and embed final CRC
        # (In a real implementation, we'd calculate CRC of the PNG bytes)
        # For simplicity, we skip this for v0.1

        return img

    def save(self, output_path: Path) -> None:
        """Build and save cartridge"""
        img = self.build()
        img.save(output_path, format='PNG', optimize=False)

        print(f"Created pixel cartridge: {output_path}")
        print(f"  ISA: {self.metadata.isa}")
        print(f"  ABI: {self.metadata.abi}")
        print(f"  Entry point: 0x{self.metadata.entry_point:08X}")
        print(f"  Binary size: {self.metadata.binary_size} bytes")
        print(f"  Payload size: {len(self.payload)} bytes")
        print(f"  Image size: {img.width} × {img.height} pixels")
        print(f"  Compression: {self.metadata.compression_type}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create pixel cartridge from binary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic cartridge
  make_pxcart.py hello.bin --isa x86_32 --abi raw_bin --entry 0x7C00 -o hello.pxcart.png

  # With compression
  make_pxcart.py app.exe --isa x86_32 --abi pe_win32 --entry 0x400000 --compress zlib -o app.pxcart.png

  # With metadata
  make_pxcart.py prog --isa x86_64 --abi elf_linux --entry 0x401000 \\
      --license MIT --author "Jane Doe" -o prog.pxcart.png
        """
    )

    parser.add_argument("input", type=Path, help="Input binary file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output .pxcart.png file")
    parser.add_argument("--isa", required=True, choices=ISA_TYPES, help="Instruction set architecture")
    parser.add_argument("--abi", required=True, choices=ABI_TYPES, help="ABI/OS type")
    parser.add_argument("--entry", type=lambda x: int(x, 0), required=True, help="Entry point address (hex or decimal)")
    parser.add_argument("--width", type=int, choices=WIDTH_OPTIONS, default=512, help="Image width in pixels")
    parser.add_argument("--compress", choices=["none", "zlib"], default="none", help="Compression type")
    parser.add_argument("--license", default="", help="License identifier")
    parser.add_argument("--author", default="", help="Author name")
    parser.add_argument("--deps", nargs="*", default=[], help="Dependencies (space-separated)")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found")
        return 1

    # Build metadata
    metadata = CartridgeMetadata(
        isa=args.isa,
        abi=args.abi,
        entry_point=args.entry,
        binary_size=0,  # Will be set by builder
        compressed=(args.compress != "none"),
        compression_type=args.compress,
        license=args.license,
        author=args.author,
        dependencies=args.deps
    )

    # Build cartridge
    builder = CartridgeBuilder(args.input, metadata, args.width)
    builder.save(args.output)

    return 0


if __name__ == "__main__":
    exit(main())
