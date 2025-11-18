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
# System Control
OP_NOP              = 0x00
OP_HALT             = 0xFF

# Output Operations
OP_PRINT_CHAR       = 0x01
OP_PRINT_STR        = 0x02
OP_SET_CURSOR       = 0x03
OP_CLEAR_SCREEN     = 0x04

# Memory Operations
OP_LOAD             = 0x10
OP_STORE            = 0x11
OP_MOVE             = 0x12

# Arithmetic Operations
OP_ADD              = 0x20
OP_SUB              = 0x21
OP_INC              = 0x22
OP_DEC              = 0x23

# Control Flow
OP_JMP              = 0x30
OP_JZ               = 0x31
OP_JNZ              = 0x32
OP_CALL             = 0x33
OP_RET              = 0x34

# CPU-GPU Communication
OP_CPU_REQ          = 0xF0
OP_YIELD            = 0xF1

# Extended opcodes for Phase 1 POC
OP_MMIO_WRITE_UART  = 0x80  # Write to serial port via CPU
OP_CPU_HALT         = 0x8F  # Signal CPU to halt (debugging)

class PXIProgram:
    """Builder for pixel-encoded programs"""

    def __init__(self, width=256):
        self.width = width
        self.pixels = []

    def emit(self, opcode, arg1=0, arg2=0, arg3=0):
        """Emit a single instruction as RGBA pixel"""
        self.pixels.append((opcode, arg1, arg2, arg3))
        return self

    def nop(self):
        return self.emit(OP_NOP)

    def halt(self):
        return self.emit(OP_HALT)

    def print_char(self, char, color=0x0F, flags=0):
        """Print ASCII character with color"""
        if isinstance(char, str):
            char = ord(char)
        return self.emit(OP_PRINT_CHAR, char, color, flags)

    def print_string(self, text, color=0x0F):
        """Print string as sequence of PRINT_CHAR instructions"""
        for char in text:
            self.print_char(char, color)
        return self

    def clear_screen(self, color=0x00):
        return self.emit(OP_CLEAR_SCREEN, color)

    def set_cursor(self, x, y):
        return self.emit(OP_SET_CURSOR, x, y)

    def mmio_write_uart(self, char, port=0x3F8):
        """Write character to serial port (requires CPU)"""
        if isinstance(char, str):
            char = ord(char)
        # Port encoded as 16-bit: high byte in arg2, low byte in arg3
        port_hi = (port >> 8) & 0xFF
        port_lo = port & 0xFF
        return self.emit(OP_MMIO_WRITE_UART, char, port_hi, port_lo)

    def cpu_halt(self):
        """Request CPU to halt (debugging)"""
        return self.emit(OP_CPU_HALT)

    def load(self, dest_reg, addr_hi, addr_lo):
        return self.emit(OP_LOAD, dest_reg, addr_hi, addr_lo)

    def store(self, src_reg, addr_hi, addr_lo):
        return self.emit(OP_STORE, src_reg, addr_hi, addr_lo)

    def move(self, dest_reg, src_reg):
        return self.emit(OP_MOVE, dest_reg, src_reg)

    def add(self, dest, src1, src2):
        return self.emit(OP_ADD, dest, src1, src2)

    def inc(self, reg):
        return self.emit(OP_INC, reg)

    def jmp(self, target):
        """Jump to instruction index"""
        target_hi = (target >> 8) & 0xFF
        target_lo = target & 0xFF
        return self.emit(OP_JMP, target_hi, target_lo)

    def jz(self, reg, target):
        """Jump if register is zero"""
        target_hi = (target >> 8) & 0xFF
        target_lo = target & 0xFF
        return self.emit(OP_JZ, reg, target_hi, target_lo)

    def save(self, filename):
        """Save program as PNG image"""
        # Calculate dimensions
        num_pixels = len(self.pixels)
        height = (num_pixels + self.width - 1) // self.width

        # Pad to fill last row
        while len(self.pixels) < height * self.width:
            self.pixels.append((OP_NOP, 0, 0, 0))

        # Convert to numpy array
        pixel_array = np.array(self.pixels, dtype=np.uint8)
        pixel_array = pixel_array.reshape((height, self.width, 4))

        # Save as PNG (force PNG format even if extension is .pxi)
        img = Image.fromarray(pixel_array, mode='RGBA')
        img.save(filename, format='PNG')

        print(f"Generated {filename}:")
        print(f"  Dimensions: {self.width}x{height}")
        print(f"  Instructions: {num_pixels}")
        print(f"  Size: {num_pixels * 4} bytes")

        return img


def create_hello_world():
    """Phase 1 POC: Hello World from GPU!"""
    prog = PXIProgram(width=256)

    # Clear screen
    prog.clear_screen(0x00)

    # Set cursor to position (0, 0)
    prog.set_cursor(0, 0)

    # Print "Hello from GPU OS!\n"
    message = "Hello from GPU OS!\n"

    # Method 1: Print to VGA text mode (GPU direct rendering)
    for char in message:
        if char == '\n':
            continue  # Skip newline for VGA
        prog.print_char(char, color=0x0A)  # Green text

    # Method 2: Also send to serial port (via CPU)
    for char in message:
        prog.mmio_write_uart(char, port=0x3F8)

    # Halt GPU execution
    prog.halt()

    # Save as os.pxi
    prog.save('build/os.pxi')

    return prog


def create_counter_demo():
    """Demonstrate loops and arithmetic"""
    prog = PXIProgram(width=256)

    # Clear screen
    prog.clear_screen(0x00)

    # Initialize counter in R0
    prog.move(0, 0)  # R0 = 0

    # Loop: print digits 0-9
    # Instruction 3: loop start
    loop_start = 3

    for i in range(10):
        prog.print_char(ord('0') + i, color=0x0E)  # Yellow
        prog.inc(0)  # R0++

    # Halt
    prog.halt()

    prog.save('build/counter.pxi')
    return prog


def analyze_pxi(filename):
    """Debug tool: Analyze PXI file"""
    img = Image.open(filename)
    pixels = np.array(img)

    print(f"\nAnalyzing {filename}:")
    print(f"Dimensions: {img.width}x{img.height}")
    print(f"Total instructions: {img.width * img.height}")
    print()

    # Decode and display first 20 instructions
    print("First 20 instructions:")
    print("  Idx  | Opcode | Arg1 | Arg2 | Arg3 | Mnemonic")
    print("-------|--------|------|------|------|------------------")

    opcode_names = {
        0x00: "NOP",
        0x01: "PRINT_CHAR",
        0x02: "PRINT_STR",
        0x03: "SET_CURSOR",
        0x04: "CLEAR_SCREEN",
        0x10: "LOAD",
        0x11: "STORE",
        0x12: "MOVE",
        0x20: "ADD",
        0x22: "INC",
        0x30: "JMP",
        0x31: "JZ",
        0x80: "MMIO_WRITE_UART",
        0x8F: "CPU_HALT",
        0xFF: "HALT",
    }

    for i in range(min(20, img.width * img.height)):
        y = i // img.width
        x = i % img.width
        r, g, b, a = pixels[y, x]

        opcode_name = opcode_names.get(r, f"UNK_{r:02X}")

        # Special formatting for PRINT_CHAR
        if r == 0x01 and 32 <= g <= 126:
            char_repr = f"'{chr(g)}'"
            print(f" {i:5d} | 0x{r:02X}   | 0x{g:02X} | 0x{b:02X} | 0x{a:02X} | {opcode_name} {char_repr}")
        else:
            print(f" {i:5d} | 0x{r:02X}   | 0x{g:02X} | 0x{b:02X} | 0x{a:02X} | {opcode_name}")

    # Count opcode usage
    print("\nOpcode usage statistics:")
    opcode_counts = {}
    for y in range(img.height):
        for x in range(img.width):
            opcode = pixels[y, x, 0]
            opcode_counts[opcode] = opcode_counts.get(opcode, 0) + 1

    for opcode in sorted(opcode_counts.keys()):
        count = opcode_counts[opcode]
        name = opcode_names.get(opcode, f"UNKNOWN_0x{opcode:02X}")
        percentage = (count / (img.width * img.height)) * 100
        print(f"  {name:20s}: {count:5d} ({percentage:5.1f}%)")


if __name__ == '__main__':
    import sys
    import os

    # Create build directory
    os.makedirs('build', exist_ok=True)

    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        # Analyze mode
        if len(sys.argv) < 3:
            print("Usage: create_os_pxi.py analyze <file.pxi>")
            sys.exit(1)
        analyze_pxi(sys.argv[2])
    else:
        # Generate mode
        print("=" * 50)
        print("pxOS Pixel Program Generator v0.1")
        print("=" * 50)
        print()

        print("Creating Phase 1 POC programs...")
        print()

        # Create hello world
        create_hello_world()
        print()

        # Create counter demo
        create_counter_demo()
        print()

        print("=" * 50)
        print("Programs generated successfully!")
        print()
        print("To analyze a program:")
        print("  python3 create_os_pxi.py analyze build/os.pxi")
        print()
        print("To view as image:")
        print("  display build/os.pxi  # (ImageMagick)")
        print("  eog build/os.pxi      # (Eye of GNOME)")
        print("=" * 50)
