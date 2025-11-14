#!/usr/bin/env python3
"""
vram_builder.py - The pxOS Assembler/Painter
This tool creates a .vram file by painting instructions into a numpy array.
"""
import numpy as np
import argparse

VRAM_WIDTH, VRAM_HEIGHT = 800, 600

# Hardcoded constants for the CPU architecture
OP_HALT = 0x00
OP_PSET = 0x01

def create_vram():
    """Creates an empty VRAM array."""
    return np.zeros((VRAM_HEIGHT, VRAM_WIDTH, 4), dtype=np.uint8)

def save_vram(vram, filename):
    """Saves the VRAM numpy array to a binary .vram file."""
    with open(filename, 'wb') as f:
        f.write(vram.tobytes())
    print(f"VRAM saved to {filename}")

def paint_instruction(vram, index, opcode, args=None):
    """
    Paints an instruction and its arguments into the VRAM.
    Returns the index for the next instruction.
    """
    y = index // VRAM_WIDTH
    x = index % VRAM_WIDTH

    # Write the opcode to the R channel of the instruction pixel
    vram[y, x, 0] = opcode
    index += 1

    if args:
        for arg_pixel in args:
            y = index // VRAM_WIDTH
            x = index % VRAM_WIDTH
            vram[y, x] = arg_pixel
            index += 1

    return index

def assemble_boot_program():
    """
    Assembles a simple boot program that draws one pixel and halts.
    """
    vram = create_vram()
    pc = 0

    # Instruction 1: PSET(100, 150, color=(255, 0, 0, 255))
    x, y = 100, 150
    color = (255, 0, 0, 255) # R, G, B, A

    # Argument for PSET is 2 pixels:
    # 1. Coordinates: [x_hi, x_lo, y_hi, y_lo]
    # 2. Color: [R, G, B, A]
    pset_args = [
        ( (x >> 8) & 0xFF, x & 0xFF, (y >> 8) & 0xFF, y & 0xFF ),
        color
    ]
    pc = paint_instruction(vram, pc, OP_PSET, args=pset_args)

    # Instruction 2: HALT
    pc = paint_instruction(vram, pc, OP_HALT)

    return vram

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pxOS VRAM Builder")
    parser.add_argument('output_file', default='boot.vram', nargs='?', help="The output .vram file.")
    args = parser.parse_args()

    print("Assembling boot program...")
    boot_vram = assemble_boot_program()
    save_vram(boot_vram, args.output_file)
    print("Done.")
