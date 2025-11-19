#!/usr/bin/env python3
"""
pxOS Pixel Program Generator
Creates os.pxi - operating system encoded as PNG pixels
Each RGBA pixel = one instruction (opcode + 3 operands)
"""

import numpy as np
from PIL import Image
import struct

# PXI Instruction Set (v0.1)
OP_NOP              = 0x00
OP_HALT             = 0xFF
OP_PRINT_CHAR       = 0x01
OP_PRINT_STR        = 0x02
OP_SET_CURSOR       = 0x03
OP_CLEAR_SCREEN     = 0x04
OP_LOAD             = 0x10
OP_STORE            = 0x11
OP_ADD              = 0x20
OP_SUB              = 0x21
OP_JMP              = 0x30
OP_JE               = 0x31
OP_JNE              = 0x32
OP_MMIO_WRITE_UART  = 0x80
OP_CPU_HALT         = 0xFE

class PXIProgram:
    def __init__(self, width=256, height=1):
        self.width = width
        self.height = height
        self.instructions = []

    def emit(self, opcode, arg1=0, arg2=0, arg3=0):
        self.instructions.append((opcode, arg1, arg2, arg3))

    def save(self, filename):
        pixel_array = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        for i, inst in enumerate(self.instructions):
            y = i // self.width
            x = i % self.width
            pixel_array[y, x] = inst

        img = Image.fromarray(pixel_array, mode='RGBA')
        img.save(filename, format='PNG')
        print(f"Generated {filename}:")
        print(f"  Dimensions: {self.width}x{self.height}")
        print(f"  Instructions: {len(self.instructions)}")

def create_hello_world_os():
    prog = PXIProgram()
    message = "Hello from GPU OS!"
    for char in message:
        prog.emit(OP_PRINT_CHAR, ord(char))
    prog.emit(OP_HALT)
    prog.save("os_poc.pxi")

if __name__ == "__main__":
    create_hello_world_os()
