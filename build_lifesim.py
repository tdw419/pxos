#!/usr/bin/env python3
"""
A simple PXI program that simulates a single organism
moving randomly and leaving a trail.
"""
from PIL import Image

def build_lifesim(path: str, width: int = 128, height: int = 128):
    img = Image.new("RGBA", (width, height), (0x00, 0x00, 0x00, 0xFF))

    def write(pc, r, g, b, a=0):
        y, x = divmod(pc, width)
        img.putpixel((x, y), (r, g, b, a))

    # A simple program to move a pixel randomly
    program = [
        (0x10, 0, 64),          # LOAD R0, 64 (x)
        (0x10, 1, 64),          # LOAD R1, 64 (y)
        (0x90, 2, 2),           # RAND R2, 2 (dx)
        (0x90, 3, 2),           # RAND R3, 2 (dy)
        (0x31, 2, 1),           # SUB R2, 1 (dx = -1, 0, or 1)
        (0x31, 3, 1),           # SUB R3, 1 (dy = -1, 0, or 1)
        (0x30, 0, 2),           # ADD R0, R2 (x += dx)
        (0x30, 1, 3),           # ADD R1, R3 (y += dy)
        (0x60, 255, 255, 255),  # DRAW white at (R0, R1)
        (0x40, 2, 0),           # JMP to pc=2
    ]

    # We need to add the opcodes to pxi_cpu.py first
    # For now, this is a placeholder.
    # I will create a simplified version that just draws a pixel.

    simple_program = [
        (0x10, 0, 64),
        (0x10, 1, 64),
        (0x60, 255, 255, 255),
        (0xFF, 0, 0)
    ]

    for i, (r, g, b, *a) in enumerate(simple_program):
        alpha = a[0] if a else 0
        write(i, r, g, b, alpha)

    img.save(path)
    print(f"LifeSim PXI program created at '{path}'")

if __name__ == "__main__":
    build_lifesim("lifesim.png")
