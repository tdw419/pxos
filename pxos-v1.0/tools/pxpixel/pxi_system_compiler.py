#!/usr/bin/env python3
# pxi_system_compiler.py
# A tool to compile high-level system operations into pixel-encoded drivers (PXI).

from PIL import Image
import sys

def generate_serial_driver(text, output_path):
    """
    Generates a PXI (PNG) image where each pixel's Red channel encodes a character
    of the input text. This represents a simple, pixel-encoded serial driver.
    """
    width = len(text)
    height = 1
    img = Image.new('RGBA', (width, height))
    pixels = img.load()

    for i, char in enumerate(text):
        ascii_val = ord(char)
        # R channel = ASCII value
        # G, B, A = 0 for this simple driver
        pixels[i, 0] = (ascii_val, 0, 0, 0)

    img.save(output_path)
    print(f"✓ Successfully generated '{output_path}' ({width}x{height})")

def main():
    print("pxpixel: PXI System Compiler v0.1")

    if len(sys.argv) != 3:
        print("Usage: ./pxi_system_compiler.py <text> <output_path>")
        print("Example: ./pxi_system_compiler.py \"Hello from GPU!\" serial_driver.pxi")
        sys.exit(1)

    text_to_encode = sys.argv[1]
    output_file = sys.argv[2]

    generate_serial_driver(text_to_encode, output_file)

if __name__ == "__main__":
    main()
