#!/usr/bin/env python3
"""
Create pixel-encoded serial driver (PXI format)

Each pixel encodes one UART operation:
- R: Character/data byte (0-255)
- G: Baud rate divisor (0x01 = 115200)
- B: Line control (0x03 = 8N1)
- A: Operation type (0x01 = write, 0x02 = read)
"""

import numpy as np
from PIL import Image
import argparse

def encode_serial_char(char, baud_divisor=0x01, line_control=0x03):
    """
    Encode a single character as RGBA pixel
    """
    return (ord(char), baud_divisor, line_control, 0x01)

def create_driver_image(message):
    """
    Create a PXI image from a string message.
    """
    width = len(message)
    height = 1

    # Create an empty image
    img_data = np.zeros((height, width, 4), dtype=np.uint8)

    # Encode each character into a pixel
    for i, char in enumerate(message):
        img_data[0, i] = encode_serial_char(char)

    return Image.fromarray(img_data, 'RGBA')

def visualize_driver(image, message):
    """
    Print a textual visualization of the pixel-encoded driver.
    """
    print("--- Pixel Driver Visualization ---")
    print(f"Message: '{message}'")
    print(f"Image size: {image.width}x{image.height}")

    pixels = image.load()
    for i in range(image.width):
        r, g, b, a = pixels[i, 0]
        char = chr(r)
        print(f"  Pixel {i}: Char='{char}' (R={r}, G={g}, B={b}, A={a})")
    print("---------------------------------")

def main():
    parser = argparse.ArgumentParser(description="Create a pixel-encoded serial driver.")
    parser.add_argument("-m", "--message", default="Hello from GPU!", help="Message to encode in the driver.")
    parser.add_argument("-o", "--output", default="serial_driver.pxi", help="Output file path for the PXI driver (PNG).")
    parser.add_argument("-v", "--visualize", action="store_true", help="Print a visualization of the encoded driver.")

    args = parser.parse_args()

    print(f"Encoding message: '{args.message}'")

    # Create the PXI image
    driver_image = create_driver_image(args.message)

    # Save the image
    driver_image.save(args.output)
    print(f"✓ Saved pixel-encoded driver to '{args.output}'")

    # Visualize if requested
    if args.visualize:
        visualize_driver(driver_image, args.message)

if __name__ == "__main__":
    main()
