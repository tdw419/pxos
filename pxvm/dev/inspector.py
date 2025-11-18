#!/usr/bin/env python3
"""
pxvm/dev/inspector.py

Pixel Program Inspector - View and understand .pxi programs.

Philosophy:
- This is a LENS to view pixels, not a replacement for pixels
- Read-only operations - never modifies the program
- Helps humans and LLMs understand what's in the pixels

Usage:
    from pxvm.dev import PixelInspector

    inspector = PixelInspector("program.pxi")
    inspector.summary()           # Program overview
    inspector.instructions()      # List all instructions
    inspector.matrix_info("W_hidden")  # Show matrix details
    inspector.health_check()      # Detect issues
"""
from __future__ import annotations

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from PIL import Image

from pxvm.core.opcodes import OP_HALT, OP_DOT_RGB, OP_ADD, OP_RELU, OP_MATMUL

OPCODE_MAP = {
    OP_HALT: "HALT",
    OP_DOT_RGB: "DOT_RGB",
    OP_ADD: "ADD",
    OP_RELU: "RELU",
    OP_MATMUL: "MATMUL",
}

class PixelInspector:
    """
    A read-only tool to inspect and understand .pxi programs.
    """
    def __init__(self, program_path: str | Path):
        self.program_path = Path(program_path)
        if not self.program_path.exists():
            raise FileNotFoundError(f"Program not found: {self.program_path}")
        self.img = np.array(Image.open(self.program_path).convert('RGBA'))
        self.height, self.width, _ = self.img.shape

    def summary(self):
        """Prints a high-level summary of the program."""
        print(f"PIXEL PROGRAM: {self.program_path.name}")
        print("=" * 70)
        print(f"Dimensions: {self.width}x{self.height} RGBA ({self.program_path.stat().st_size / 1024:.1f}KB)")
        instructions = self.get_instructions()
        print(f"Instructions: {len(instructions)} opcodes")

    def get_instructions(self) -> List[Tuple[int, np.ndarray]]:
        """Extracts all instructions from the program."""
        instructions = []
        for x in range(self.width):
            instr_pixel = self.img[0, x]
            opcode = int(instr_pixel[0])
            if opcode == OP_HALT and x > 0: # a halt at 0 is an empty program
                instructions.append((x, instr_pixel))
                break
            if opcode != 0 : # if not a NOP
                 instructions.append((x, instr_pixel))
        return instructions

    def instructions(self):
        """Prints a formatted list of all instructions."""
        print("INSTRUCTIONS (Row 0):")
        for x, instr_pixel in self.get_instructions():
            opcode = int(instr_pixel[0])
            args = instr_pixel[1:]
            op_name = OPCODE_MAP.get(opcode, "UNKNOWN")
            print(f"  [{x:03d}]: {op_name:8s} ({opcode}) args=({args[0]:3d}, {args[1]:3d}, {args[2]:3d})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 -m pxvm.dev.inspector <program.pxi>")
        sys.exit(1)

    inspector = PixelInspector(sys.argv[1])
    inspector.summary()
    inspector.instructions()
