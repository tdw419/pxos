#!/usr/bin/env python3
"""
pxi_compress.py - PXI Image Compressor

This utility takes a PXI program image and compresses it into a
raw binary data blob using zlib. This is the first step in
creating a self-extracting "Genesis Pixel" for pxOS.
"""

import zlib
from PIL import Image
import argparse

def compress_pxi(input_path: str, output_path: str):
    """
    Compresses a PXI image file into a zlib-compressed binary blob.
    """
    try:
        img = Image.open(input_path).convert("RGBA")
        img_bytes = img.tobytes()
        compressed_bytes = zlib.compress(img_bytes, level=9)

        with open(output_path, 'wb') as f:
            f.write(compressed_bytes)

        print(f"Successfully compressed '{input_path}' ({len(img_bytes)} bytes)")
        print(f"to '{output_path}' ({len(compressed_bytes)} bytes).")
        print(f"Compression ratio: {len(img_bytes) / len(compressed_bytes):.2f}x")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="PXI Image Compressor")
    parser.add_argument("input_file", help="Path to the input PXI image (.png)")
    parser.add_argument("output_file", help="Path for the output compressed data (.bin)")
    args = parser.parse_args()

    compress_pxi(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
