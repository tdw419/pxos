#!/usr/bin/env python3
"""
PixelFS: Pixel-Based File System
"""

import hashlib
import math
import struct
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

class PixelFS:
    MAGIC = b'PXFS'
    VERSION = (1, 0)
    HEADER_SIZE = 64

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)

    def write(self, name: str, data: bytes, compression: int = 0) -> Path:
        original_size = len(data)
        checksum = hashlib.sha256(data).digest()

        width = 1024
        num_pixels = math.ceil(original_size / 3)
        height = math.ceil(num_pixels / width)

        padded_size = width * height * 3
        padded_data = data.ljust(padded_size, b'\x00')

        header = self.pack_header(original_size, width, height, compression, checksum)

        img_array = np.frombuffer(padded_data, dtype=np.uint8).reshape((height, width, 3))
        img = Image.fromarray(img_array, 'RGB')

        filepath = self.root / f"{name}.pxi"
        meta_path = self.root / f"{name}.meta"

        img.save(filepath, format='PNG')
        with meta_path.open('wb') as f:
            f.write(header)

        return filepath

    def read(self, name: str, offset: int = 0, length: Optional[int] = None) -> bytes:
        filepath = self.root / f"{name}.pxi"
        meta_path = self.root / f"{name}.meta"

        with meta_path.open('rb') as f:
            header_data = f.read(self.HEADER_SIZE)
            header = self.unpack_header(header_data)

        img = Image.open(filepath)
        img_array = np.array(img)

        data = img_array.tobytes()

        if length is None:
            length = header['original_size'] - offset

        return data[offset:offset+length]

    def pack_header(self, original_size, width, height, compression, checksum):
        return struct.pack(
            '>4s BB Q I I B 32s 9s',
            self.MAGIC,
            self.VERSION[0],
            self.VERSION[1],
            original_size,
            width,
            height,
            compression,
            checksum,
            b'\x00' * 9
        )

    def unpack_header(self, data):
        magic, v_maj, v_min, size, w, h, comp, chk, _ = struct.unpack('>4s BB Q I I B 32s 9s', data)
        if magic != self.MAGIC:
            raise ValueError("Invalid PixelFS file format")
        return {
            "version": (v_maj, v_min),
            "original_size": size,
            "width": w,
            "height": h,
            "compression": comp,
            "checksum": chk
        }

def demo():
    print("=== PixelFS Demo ===")
    fs = PixelFS("pixel_storage_demo")
    demo_data = b"Hello, Pixel World!" * 200
    fs.write("demo_file", demo_data)
    read_data = fs.read("demo_file")
    assert read_data == demo_data[:len(read_data)]
    print("✓ Demo complete!")

if __name__ == "__main__":
    demo()
