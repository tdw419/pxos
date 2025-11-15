#!/usr/bin/env python3
"""
PXI Compression System
Compress pixel programs down to minimal size using:
1. zlib compression
2. Self-extracting bootstrap
3. Fractal/procedural generation (future)

Goal: Compress 1,048,576 pixels → 4 pixels → 1 pixel
"""

from PIL import Image
import zlib
import struct
import math
from typing import Tuple

# Bootstrap opcodes (same as pxi_cpu.py)
OP_LOAD = 0x10
OP_HALT = 0xFF
OP_DECOMPRESS = 0xD0  # New: decompress embedded data


class PXICompressor:
    """Compress pixel programs to minimal form"""

    def __init__(self):
        self.compression_ratio = 0.0

    def compress_image(self, img: Image.Image, level: int = 9) -> bytes:
        """Compress image data with zlib"""
        # Convert image to raw RGBA bytes
        raw_data = img.tobytes()
        compressed = zlib.compress(raw_data, level=level)

        print(f"Original size: {len(raw_data):,} bytes")
        print(f"Compressed size: {len(compressed):,} bytes")
        self.compression_ratio = len(compressed) / len(raw_data)
        print(f"Compression ratio: {self.compression_ratio:.2%}")

        return compressed

    def create_self_extracting_image(
        self,
        original_img: Image.Image,
        output_path: str = "compressed.png"
    ) -> Image.Image:
        """
        Create a self-extracting pixel program:
        - First N pixels: bootstrap decompressor
        - Remaining pixels: compressed payload encoded as pixel data
        """

        # Compress the original image
        compressed_data = self.compress_image(original_img)
        width, height = original_img.size

        # Calculate how many pixels we need for the payload
        # Each pixel holds 4 bytes (RGBA)
        bytes_per_pixel = 4
        payload_pixels_needed = math.ceil(len(compressed_data) / bytes_per_pixel)

        # Bootstrap program size (in pixels)
        bootstrap_size = 32  # Reserve 32 pixels for decompressor code

        # Total pixels needed
        total_pixels = bootstrap_size + payload_pixels_needed

        # Create output image (make it square, power of 2)
        output_size = 2 ** math.ceil(math.log2(math.sqrt(total_pixels)))
        output_img = Image.new("RGBA", (output_size, output_size), (0, 0, 0, 255))

        print(f"\nSelf-extracting image:")
        print(f"  Bootstrap: {bootstrap_size} pixels")
        print(f"  Payload: {payload_pixels_needed} pixels")
        print(f"  Total: {total_pixels} pixels")
        print(f"  Output size: {output_size}x{output_size}")
        print(f"  Compression: {original_img.size[0] * original_img.size[1]} → {total_pixels} pixels")
        print(f"  Reduction: {total_pixels / (original_img.size[0] * original_img.size[1]):.2%}")

        # Write bootstrap code
        self._write_bootstrap(output_img, bootstrap_size, width, height, len(compressed_data))

        # Write compressed payload into pixels
        self._write_payload(output_img, compressed_data, bootstrap_size)

        output_img.save(output_path)
        print(f"\n✓ Saved to {output_path}")

        return output_img

    def _write_bootstrap(
        self,
        img: Image.Image,
        bootstrap_size: int,
        orig_width: int,
        orig_height: int,
        payload_size: int
    ):
        """Write bootstrap decompressor code"""

        def set_pixel(pc, r, g, b, a=0):
            x = pc % img.width
            y = pc // img.width
            img.putpixel((x, y), (r, g, b, a))

        pc = 0

        # Pixel 0-3: Metadata (original dimensions + payload size)
        # Store as RGBA values
        set_pixel(pc, 0x4D, 0x45, 0x54, 0x41)  # "META" marker
        pc += 1

        # Original width (split across 2 bytes)
        set_pixel(pc, orig_width >> 8, orig_width & 0xFF, 0, 0)
        pc += 1

        # Original height
        set_pixel(pc, orig_height >> 8, orig_height & 0xFF, 0, 0)
        pc += 1

        # Payload size (4 bytes across 1 pixel)
        set_pixel(pc,
                  (payload_size >> 24) & 0xFF,
                  (payload_size >> 16) & 0xFF,
                  (payload_size >> 8) & 0xFF,
                  payload_size & 0xFF)
        pc += 1

        # Pixels 4-31: Decompressor code
        # (In real implementation, this would be PXI bytecode)
        # For now, just mark it as decompressor region
        for i in range(pc, bootstrap_size):
            set_pixel(i, OP_DECOMPRESS, 0, 0, 0)

    def _write_payload(self, img: Image.Image, data: bytes, offset: int):
        """Write compressed data into pixels starting at offset"""
        pc = offset

        for i in range(0, len(data), 4):
            chunk = data[i:i+4]
            # Pad with zeros if needed
            r = chunk[0] if len(chunk) > 0 else 0
            g = chunk[1] if len(chunk) > 1 else 0
            b = chunk[2] if len(chunk) > 2 else 0
            a = chunk[3] if len(chunk) > 3 else 0

            x = pc % img.width
            y = pc // img.width
            img.putpixel((x, y), (r, g, b, a))
            pc += 1

    def extract(self, compressed_img: Image.Image) -> Image.Image:
        """Extract original image from self-extracting archive"""

        # Read metadata from first 4 pixels
        def get_pixel(pc):
            x = pc % compressed_img.width
            y = pc // compressed_img.width
            return compressed_img.getpixel((x, y))

        # Check META marker
        r, g, b, a = get_pixel(0)
        if (r, g, b, a) != (0x4D, 0x45, 0x54, 0x41):
            raise ValueError("Not a valid self-extracting PXI image (missing META marker)")

        # Read original dimensions
        r1, g1, _, _ = get_pixel(1)
        orig_width = (r1 << 8) | g1

        r2, g2, _, _ = get_pixel(2)
        orig_height = (r2 << 8) | g2

        # Read payload size
        r3, g3, b3, a3 = get_pixel(3)
        payload_size = (r3 << 24) | (g3 << 16) | (b3 << 8) | a3

        print(f"Extracting:")
        print(f"  Original dimensions: {orig_width}x{orig_height}")
        print(f"  Payload size: {payload_size:,} bytes")

        # Read compressed payload
        bootstrap_size = 32
        payload_data = bytearray()

        pc = bootstrap_size
        for i in range(0, payload_size, 4):
            r, g, b, a = get_pixel(pc)
            payload_data.extend([r, g, b, a])
            pc += 1

        # Trim to exact size
        payload_data = bytes(payload_data[:payload_size])

        # Decompress
        decompressed = zlib.decompress(payload_data)
        expected_size = orig_width * orig_height * 4  # RGBA
        print(f"  Decompressed: {len(decompressed):,} bytes")

        if len(decompressed) != expected_size:
            raise ValueError(f"Decompressed size mismatch: {len(decompressed)} != {expected_size}")

        # Reconstruct image
        reconstructed = Image.frombytes("RGBA", (orig_width, orig_height), decompressed)
        print(f"✓ Extraction complete")

        return reconstructed


def create_test_program() -> Image.Image:
    """Create a test program to compress"""
    size = 128
    img = Image.new("RGBA", (size, size), (0, 0, 0, 255))

    # Create a pattern with lots of repetition (compresses well)
    for y in range(size):
        for x in range(size):
            # Checkerboard pattern
            if (x // 8 + y // 8) % 2 == 0:
                img.putpixel((x, y), (255, 255, 255, 255))
            else:
                img.putpixel((x, y), (0, 0, 0, 255))

    return img


if __name__ == "__main__":
    print("=== PXI Compression System ===\n")

    # Create test program
    print("Creating test program (128x128 with pattern)...")
    test_img = create_test_program()
    test_img.save("/home/user/pxos/test_original.png")
    print(f"Original: {test_img.size[0] * test_img.size[1]} pixels\n")

    # Compress it
    compressor = PXICompressor()
    compressed = compressor.create_self_extracting_image(
        test_img,
        "/home/user/pxos/test_compressed.png"
    )

    print(f"\n{'='*50}")
    print("COMPRESSION COMPLETE")
    print(f"{'='*50}")
    print(f"Original: 128x128 = 16,384 pixels")
    print(f"Compressed: {compressed.size[0]}x{compressed.size[1]} = {compressed.size[0] * compressed.size[1]} pixels")
    reduction = (1 - compressed.size[0] * compressed.size[1] / 16384) * 100
    print(f"Reduction: {reduction:.1f}%")

    # Test extraction
    print(f"\n{'='*50}")
    print("TESTING EXTRACTION")
    print(f"{'='*50}\n")

    extracted = compressor.extract(Image.open("/home/user/pxos/test_compressed.png"))
    extracted.save("/home/user/pxos/test_extracted.png")

    # Verify
    if list(extracted.getdata()) == list(test_img.getdata()):
        print("\n✓ PERFECT RECONSTRUCTION - Images match exactly!")
    else:
        print("\n✗ Reconstruction error")
