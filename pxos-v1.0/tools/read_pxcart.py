#!/usr/bin/env python3
"""
PXCARTRIDGE Reader v0.1

Read and extract pixel cartridge format (.pxcart.png)

This tool reads pixel cartridges, displays metadata, verifies integrity,
and extracts the original binary.

Usage:
    read_pxcart.py cartridge.pxcart.png --info
    read_pxcart.py cartridge.pxcart.png --extract output.bin
    read_pxcart.py cartridge.pxcart.png --verify
"""

import argparse
import struct
import hashlib
import zlib
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: This tool requires PIL (Pillow) and numpy")
    print("Install with: pip install pillow numpy")
    exit(1)


# ============================================================================
# Constants (same as make_pxcart.py)
# ============================================================================

MAGIC = b"PXCT"
VERSION = b"0001"
HEADER_ROWS = 16

FLAG_COMPRESSED = 0x01
FLAG_HAS_IR = 0x02
FLAG_PARTIAL_PORT = 0x04

STATUS_NAMES = {
    0x00: "Raw",
    0x01: "Analyzed",
    0x02: "Lifted",
    0x03: "Ported",
    0x04: "Tested"
}


# ============================================================================
# Decoding Functions
# ============================================================================

def decode_string(data: bytes) -> str:
    """Decode null-terminated UTF-8 string"""
    null_pos = data.find(b'\0')
    if null_pos >= 0:
        data = data[:null_pos]
    return data.decode('utf-8', errors='replace')


def decode_u32(data: bytes) -> int:
    """Decode 32-bit little-endian unsigned int"""
    return struct.unpack('<I', data[:4])[0]


def decode_u64(data: bytes) -> int:
    """Decode 64-bit little-endian unsigned int"""
    return struct.unpack('<Q', data[:8])[0]


def pixels_to_bytes(img_array: np.ndarray) -> bytes:
    """Convert RGBA pixel array to bytes"""
    return img_array.flatten().tobytes()


# ============================================================================
# Cartridge Reader
# ============================================================================

@dataclass
class CartridgeInfo:
    """Parsed cartridge metadata"""
    magic: bytes
    version: bytes
    isa: str
    abi: str
    entry_point: int
    binary_size: int
    flags: int
    compression_type: str
    dependencies: list
    license: str
    author: str
    payload_hash: bytes
    header_crc: int

    @property
    def is_compressed(self) -> bool:
        return bool(self.flags & FLAG_COMPRESSED)

    @property
    def has_ir(self) -> bool:
        return bool(self.flags & FLAG_HAS_IR)

    @property
    def is_partial_port(self) -> bool:
        return bool(self.flags & FLAG_PARTIAL_PORT)

    @property
    def status(self) -> int:
        return (self.flags >> 8) & 0xFF

    @property
    def status_name(self) -> str:
        return STATUS_NAMES.get(self.status, "Unknown")


class CartridgeReader:
    """Read pixel cartridge images"""

    def __init__(self, cart_path: Path):
        self.cart_path = cart_path

        # Load image
        img = Image.open(cart_path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        self.img_array = np.array(img)
        self.height, self.width, _ = self.img_array.shape

        # Parse header
        self.info = self._parse_header()

        # Extract payload
        self.payload = self._extract_payload()

    def _parse_header(self) -> CartridgeInfo:
        """Parse header rows into metadata"""
        # Extract header rows
        header_array = self.img_array[:HEADER_ROWS]
        header_bytes = pixels_to_bytes(header_array)

        offset = 0

        # Row 0: Magic & Version
        magic = header_bytes[offset:offset+4]
        offset += 4
        version = header_bytes[offset:offset+4]
        offset += 4
        offset = self.width * 4

        if magic != MAGIC:
            raise ValueError(f"Invalid magic number: {magic} (expected {MAGIC})")

        # Row 1: ISA & ABI
        isa = decode_string(header_bytes[offset:offset+32])
        offset += 32
        abi = decode_string(header_bytes[offset:offset+32])
        offset += 32
        offset = 2 * self.width * 4

        # Row 2: Entry point & Size
        entry_point = decode_u64(header_bytes[offset:offset+8])
        offset += 8
        binary_size = decode_u64(header_bytes[offset:offset+8])
        offset += 8
        offset = 3 * self.width * 4

        # Row 3: Flags & Compression
        flags = decode_u32(header_bytes[offset:offset+4])
        offset += 4
        compression_type = decode_string(header_bytes[offset:offset+16])
        offset += 16
        offset = 4 * self.width * 4

        # Row 4-7: Dependencies
        deps_str = decode_string(header_bytes[offset:offset+4*self.width*4])
        dependencies = [d.strip() for d in deps_str.split(',') if d.strip()]
        offset = 8 * self.width * 4

        # Row 8-11: License & Author
        license_str = decode_string(header_bytes[offset:offset+2*self.width*4])
        offset += 2 * self.width * 4
        author_str = decode_string(header_bytes[offset:offset+2*self.width*4])
        offset = 12 * self.width * 4

        # Row 12-15: Checksums
        payload_hash = header_bytes[offset:offset+32]
        offset += 32
        header_crc = decode_u32(header_bytes[offset:offset+4])

        return CartridgeInfo(
            magic=magic,
            version=version,
            isa=isa,
            abi=abi,
            entry_point=entry_point,
            binary_size=binary_size,
            flags=flags,
            compression_type=compression_type,
            dependencies=dependencies,
            license=license_str,
            author=author_str,
            payload_hash=payload_hash,
            header_crc=header_crc
        )

    def _extract_payload(self) -> bytes:
        """Extract payload bytes from image"""
        # Payload starts after header, ends before checksum row
        payload_array = self.img_array[HEADER_ROWS:-1]
        payload_bytes = pixels_to_bytes(payload_array)

        # Decompress if needed
        if self.info.is_compressed:
            if self.info.compression_type == "zlib":
                return zlib.decompress(payload_bytes)
            else:
                print(f"Warning: Unknown compression type: {self.info.compression_type}")
                return payload_bytes
        else:
            return payload_bytes

    def verify(self) -> bool:
        """Verify checksums"""
        # Verify payload hash
        actual_hash = hashlib.sha256(self.payload).digest()
        if actual_hash != self.info.payload_hash:
            print("FAIL: Payload hash mismatch")
            print(f"  Expected: {self.info.payload_hash.hex()}")
            print(f"  Actual:   {actual_hash.hex()}")
            return False

        print("PASS: Payload hash verified")

        # TODO: Verify header CRC and full image CRC
        # (Requires reconstructing header bytes and comparing)

        return True

    def print_info(self) -> None:
        """Print cartridge information"""
        print("=" * 60)
        print("PIXEL CARTRIDGE INFO")
        print("=" * 60)
        print(f"File:            {self.cart_path}")
        print(f"Image size:      {self.width} × {self.height} pixels")
        print()
        print("HEADER")
        print("-" * 60)
        print(f"Magic:           {self.info.magic.decode('ascii', errors='replace')}")
        print(f"Version:         {self.info.version.decode('ascii', errors='replace')}")
        print()
        print("BINARY INFO")
        print("-" * 60)
        print(f"ISA:             {self.info.isa}")
        print(f"ABI:             {self.info.abi}")
        print(f"Entry point:     0x{self.info.entry_point:08X}")
        print(f"Binary size:     {self.info.binary_size} bytes")
        print()
        print("FLAGS")
        print("-" * 60)
        print(f"Flags:           0x{self.info.flags:08X}")
        print(f"  Compressed:    {self.info.is_compressed}")
        print(f"  Has IR:        {self.info.has_ir}")
        print(f"  Partial port:  {self.info.is_partial_port}")
        print(f"  Status:        {self.info.status_name} ({self.info.status})")
        print()
        print("COMPRESSION")
        print("-" * 60)
        print(f"Type:            {self.info.compression_type}")
        print(f"Payload size:    {len(self.payload)} bytes")
        print()
        print("METADATA")
        print("-" * 60)
        print(f"License:         {self.info.license or '(none)'}")
        print(f"Author:          {self.info.author or '(none)'}")
        print(f"Dependencies:    {', '.join(self.info.dependencies) or '(none)'}")
        print()
        print("CHECKSUMS")
        print("-" * 60)
        print(f"Payload SHA256:  {self.info.payload_hash.hex()}")
        print(f"Header CRC32:    0x{self.info.header_crc:08X}")
        print("=" * 60)

    def extract_binary(self, output_path: Path) -> None:
        """Extract binary to file"""
        # Trim to actual binary size
        binary = self.payload[:self.info.binary_size]

        output_path.write_bytes(binary)
        print(f"Extracted binary: {output_path}")
        print(f"  Size: {len(binary)} bytes")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Read and extract pixel cartridges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show info
  read_pxcart.py hello.pxcart.png --info

  # Extract binary
  read_pxcart.py hello.pxcart.png --extract hello.bin

  # Verify checksums
  read_pxcart.py hello.pxcart.png --verify

  # Do all
  read_pxcart.py hello.pxcart.png --info --verify --extract hello.bin
        """
    )

    parser.add_argument("input", type=Path, help="Input .pxcart.png file")
    parser.add_argument("--info", action="store_true", help="Show cartridge info")
    parser.add_argument("--verify", action="store_true", help="Verify checksums")
    parser.add_argument("--extract", type=Path, help="Extract binary to file")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found")
        return 1

    # If no action specified, default to --info
    if not (args.info or args.verify or args.extract):
        args.info = True

    try:
        reader = CartridgeReader(args.input)

        if args.info:
            reader.print_info()

        if args.verify:
            print()
            reader.verify()

        if args.extract:
            print()
            reader.extract_binary(args.extract)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
