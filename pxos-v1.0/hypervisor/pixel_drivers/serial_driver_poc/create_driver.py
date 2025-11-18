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

    Args:
        char: ASCII character (int or str)
        baud_divisor: Baud rate (0x01 = 115200)
        line_control: Line control (0x03 = 8N1)

    Returns:
        (R, G, B, A) tuple
    """
    if isinstance(char, str):
        char = ord(char)

    return (
        char,           # R: Character
        baud_divisor,   # G: Baud rate
        line_control,   # B: Line control
        0x01            # A: Write operation
    )


def create_serial_driver(message, width=256, height=1):
    """
    Create pixel-encoded serial driver from message string

    Args:
        message: String to encode
        width: Image width (pixels per row)
        height: Image height (total rows)

    Returns:
        PIL Image object (RGBA)
    """
    # Calculate total pixels
    total_pixels = width * height

    # Encode message as pixels
    pixels = []
    for char in message:
        pixels.append(encode_serial_char(char))

    # Pad remaining pixels with NOPs (A=0x00)
    while len(pixels) < total_pixels:
        pixels.append((0, 0, 0, 0))  # NOP operation

    # Truncate if too long
    pixels = pixels[:total_pixels]

    # Convert to numpy array and reshape
    pixel_array = np.array(pixels, dtype=np.uint8)
    pixel_array = pixel_array.reshape((height, width, 4))

    # Create PIL image
    img = Image.fromarray(pixel_array, mode='RGBA')

    return img


def save_pxi(filename, message, width=256, height=1):
    """
    Save pixel-encoded serial driver as PNG image

    Args:
        filename: Output filename (e.g., 'serial_driver.pxi')
        message: String to encode
        width: Image width
        height: Image height
    """
    img = create_serial_driver(message, width, height)
    img.save(filename, format='PNG', compress_level=0)

    print(f"Created {filename}:")
    print(f"  Message: {repr(message)}")
    print(f"  Size: {width}x{height} pixels")
    print(f"  Characters: {len(message)}")
    print(f"  File size: {len(open(filename, 'rb').read())} bytes")


def visualize_driver(filename):
    """
    Visualize pixel-encoded driver (for debugging)

    Args:
        filename: PXI file to visualize
    """
    img = Image.open(filename)
    pixel_array = np.array(img)

    print(f"\nDriver visualization ({filename}):")
    print("Idx  R(char) G(baud) B(ctrl) A(op)  ASCII")
    print("=" * 50)

    height, width, _ = pixel_array.shape
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixel_array[y, x]

            # Skip NOP operations
            if a == 0:
                continue

            # Display character
            char = chr(r) if 32 <= r <= 126 else f"\\x{r:02x}"
            op_name = "WRITE" if a == 1 else "READ" if a == 2 else f"OP{a}"

            idx = y * width + x
            print(f"{idx:3d}  0x{r:02x}    0x{g:02x}    0x{b:02x}    0x{a:02x}   "
                  f"{char!r:6s} ({op_name})")


def main():
    parser = argparse.ArgumentParser(
        description='Create pixel-encoded serial driver (PXI format)'
    )
    parser.add_argument(
        '-m', '--message',
        default='Hello from GPU!',
        help='Message to encode (default: "Hello from GPU!")'
    )
    parser.add_argument(
        '-o', '--output',
        default='serial_driver.pxi',
        help='Output filename (default: serial_driver.pxi)'
    )
    parser.add_argument(
        '-w', '--width',
        type=int,
        default=256,
        help='Image width (default: 256)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=1,
        help='Image height (default: 1)'
    )
    parser.add_argument(
        '-v', '--visualize',
        action='store_true',
        help='Visualize the created driver'
    )

    args = parser.parse_args()

    # Create driver
    save_pxi(args.output, args.message, args.width, args.height)

    # Visualize if requested
    if args.visualize:
        visualize_driver(args.output)


if __name__ == '__main__':
    main()
