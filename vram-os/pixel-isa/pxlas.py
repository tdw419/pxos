#!/usr/bin/env python3
"""
pxlas.py - Pixel ISA Assembler
Converts Pixel ISA assembly (.pxl) to executable PNG images (.px)
"""

import sys
import re
import struct
from io import BytesIO
from typing import List, Dict, Tuple, Optional
import argparse

try:
    from PIL import Image
except ImportError:
    print("Error: PIL (Pillow) is required. Install with: pip install Pillow")
    sys.exit(1)


class PixelInstruction:
    """Represents a single 32-bit Pixel ISA instruction encoded as RGBA"""

    def __init__(self, opcode: int, arg0: int = 0, arg1: int = 0, flags: int = 0):
        self.opcode = opcode & 0xFF
        self.arg0 = arg0 & 0xFF
        self.arg1 = arg1 & 0xFF
        self.flags = flags & 0xFF

    def to_pixel(self) -> Tuple[int, int, int, int]:
        """Convert instruction to RGBA pixel tuple"""
        return (self.opcode, self.arg0, self.arg1, self.flags)

    def to_bytes(self) -> bytes:
        """Convert instruction to 4 bytes"""
        return struct.pack('BBBB', self.opcode, self.arg0, self.arg1, self.flags)

    def __repr__(self):
        return f"PixelInstruction(R={self.opcode:02X}, G={self.arg0:02X}, B={self.arg1:02X}, A={self.flags:02X})"


class PixelAssembler:
    """Assembler for Pixel ISA"""

    # Instruction set opcodes
    OPCODES = {
        'NOP': 0x00,
        'LOAD': 0x01, 'STORE': 0x02, 'LOADI': 0x03, 'MOV': 0x04,
        'ADD': 0x05, 'SUB': 0x06, 'MUL': 0x07, 'DIV': 0x08,
        'AND': 0x09, 'OR': 0x0A, 'XOR': 0x0B, 'NOT': 0x0C,
        'SHL': 0x0D, 'SHR': 0x0E, 'CMP': 0x0F,
        'JMP': 0x10, 'JEQ': 0x11, 'JNE': 0x12, 'JLT': 0x13,
        'JGT': 0x14, 'JLE': 0x15, 'JGE': 0x16,
        'CALL': 0x17, 'RET': 0x18, 'PUSH': 0x19, 'POP': 0x1A,
        'HALT': 0x20, 'SYSCALL': 0x21, 'INT': 0x22,
        'IRET': 0x23, 'CLI': 0x24, 'STI': 0x25,
        'VPIX': 0x30, 'VGET': 0x31, 'VRECT': 0x32,
        'VLINE': 0x33, 'VBLIT': 0x34, 'VBLEND': 0x35,
    }

    # Register names mapping
    REGISTERS = {
        'R0': 0, 'R1': 1, 'R2': 2, 'R3': 3,
        'R4': 4, 'R5': 5, 'R6': 6, 'R7': 7,
    }

    def __init__(self):
        self.instructions: List[PixelInstruction] = []
        self.labels: Dict[str, int] = {}
        self.data: Dict[int, bytes] = {}
        self.constants: Dict[str, int] = {}
        self.current_address = 0
        self.origin = 0x1000  # Default program start

    def parse_register(self, token: str) -> int:
        """Parse register name to register number"""
        token = token.upper().strip()
        if token in self.REGISTERS:
            return self.REGISTERS[token]
        raise ValueError(f"Invalid register: {token}")

    def parse_immediate(self, token: str) -> int:
        """Parse immediate value (supports #42, 0x2A, constants)"""
        token = token.strip()

        # Remove '#' prefix if present
        if token.startswith('#'):
            token = token[1:]

        # Check if it's a constant
        if token in self.constants:
            return self.constants[token]

        # Check if it's a label (for addresses)
        if token in self.labels:
            return self.labels[token]

        # Parse hex
        if token.startswith('0x') or token.startswith('0X'):
            return int(token, 16)

        # Parse binary
        if token.startswith('0b') or token.startswith('0B'):
            return int(token, 2)

        # Parse decimal
        return int(token)

    def parse_address(self, token: str) -> int:
        """Parse memory address or label"""
        token = token.strip()

        # Handle [R1] or [R1 + 10] syntax
        if token.startswith('[') and token.endswith(']'):
            # This is indirect addressing - return register number
            inner = token[1:-1].strip()
            if '+' in inner:
                # [R1 + offset] - we'll encode register in arg0, offset in arg1
                parts = inner.split('+')
                reg = self.parse_register(parts[0].strip())
                return reg
            else:
                return self.parse_register(inner)

        # Direct address or label
        return self.parse_immediate(token)

    def first_pass(self, source: str):
        """First pass: collect labels and calculate addresses"""
        address = self.origin

        for line_num, line in enumerate(source.split('\n'), 1):
            # Remove comments
            if ';' in line:
                line = line[:line.index(';')]
            line = line.strip()

            if not line:
                continue

            # Check for label
            if ':' in line:
                label, rest = line.split(':', 1)
                label = label.strip()
                self.labels[label] = address
                line = rest.strip()

                if not line:
                    continue

            # Check for directives
            if line.upper().startswith('ORG'):
                parts = line.split()
                if len(parts) >= 2:
                    self.origin = self.parse_immediate(parts[1])
                    address = self.origin
                continue

            if line.upper().startswith('EQU'):
                # Define constant: UART EQU 0x10000000
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[0].upper()
                    value = self.parse_immediate(parts[2])
                    self.constants[name] = value
                continue

            if line.upper().startswith('DATA'):
                # String data - will be encoded later
                address += len(line) // 4 + 1  # Rough estimate
                continue

            # Regular instruction - increment address
            address += 4  # Each instruction = 4 bytes = 1 pixel

    def assemble_instruction(self, line: str) -> Optional[PixelInstruction]:
        """Assemble a single instruction"""
        parts = [p.strip() for p in re.split(r'[,\s]+', line) if p.strip()]

        if not parts:
            return None

        opcode_str = parts[0].upper()

        if opcode_str not in self.OPCODES:
            return None  # Skip unknown opcodes

        opcode = self.OPCODES[opcode_str]
        arg0 = 0
        arg1 = 0
        flags = 0

        # Parse arguments based on instruction type
        if opcode_str in ['NOP', 'RET', 'HALT', 'IRET', 'CLI', 'STI']:
            # No arguments
            pass

        elif opcode_str in ['LOAD', 'STORE', 'LOADI', 'MOV', 'ADD', 'SUB', 'MUL', 'DIV',
                            'AND', 'OR', 'XOR', 'SHL', 'SHR', 'CMP']:
            # Two arguments: reg, reg or reg, imm
            if len(parts) >= 2:
                arg0 = self.parse_register(parts[1])
            if len(parts) >= 3:
                try:
                    arg1 = self.parse_register(parts[2])
                except ValueError:
                    # Not a register, must be immediate
                    arg1 = self.parse_immediate(parts[2]) & 0xFF

        elif opcode_str in ['JMP', 'JEQ', 'JNE', 'JLT', 'JGT', 'JLE', 'JGE', 'CALL']:
            # Jump instructions: address
            if len(parts) >= 2:
                addr = self.parse_immediate(parts[1])
                arg0 = (addr >> 8) & 0xFF  # High byte
                arg1 = addr & 0xFF          # Low byte

        elif opcode_str in ['PUSH', 'POP', 'NOT']:
            # Single register argument
            if len(parts) >= 2:
                arg0 = self.parse_register(parts[1])

        elif opcode_str in ['SYSCALL', 'INT']:
            # System call number
            if len(parts) >= 2:
                arg0 = self.parse_immediate(parts[1]) & 0xFF

        elif opcode_str in ['VPIX', 'VGET', 'VBLIT']:
            # Graphics instructions - simplified encoding
            if len(parts) >= 2:
                arg0 = self.parse_immediate(parts[1]) & 0xFF
            if len(parts) >= 3:
                arg1 = self.parse_immediate(parts[2]) & 0xFF

        return PixelInstruction(opcode, arg0, arg1, flags)

    def second_pass(self, source: str):
        """Second pass: generate machine code"""
        for line in source.split('\n'):
            # Remove comments
            if ';' in line:
                line = line[:line.index(';')]
            line = line.strip()

            if not line:
                continue

            # Skip labels (already processed)
            if ':' in line:
                _, line = line.split(':', 1)
                line = line.strip()
                if not line:
                    continue

            # Skip directives
            if line.upper().startswith(('ORG', 'EQU', 'DATA', 'ALIGN')):
                continue

            # Assemble instruction
            instruction = self.assemble_instruction(line)
            if instruction:
                self.instructions.append(instruction)

    def create_metadata_row(self) -> List[Tuple[int, int, int, int]]:
        """Create the metadata row (row 0 of PNG)"""
        metadata = []

        # Pixel 0: Magic number "PXL\0"
        metadata.append((0x50, 0x58, 0x4C, 0x00))

        # Pixel 1: Version 2.0
        metadata.append((0x02, 0x00, 0x00, 0x00))

        # Pixel 2: Entry point (default: after metadata)
        metadata.append((0x00, 0x00, 0x00, 0x08))

        # Pixel 3: Program size (number of instructions)
        size = len(self.instructions)
        metadata.append(((size >> 24) & 0xFF, (size >> 16) & 0xFF,
                        (size >> 8) & 0xFF, size & 0xFF))

        # Pixel 4: Required VRAM size (64KB default)
        metadata.append((0x00, 0x01, 0x00, 0x00))

        # Pixels 5-7: Reserved
        for _ in range(3):
            metadata.append((0, 0, 0, 0))

        return metadata

    def assemble(self, source: str) -> List[Tuple[int, int, int, int]]:
        """Main assembly function"""
        self.first_pass(source)
        self.second_pass(source)

        # Create pixel list
        pixels = self.create_metadata_row()

        # Add instructions
        for inst in self.instructions:
            pixels.append(inst.to_pixel())

        return pixels

    def create_png(self, pixels: List[Tuple[int, int, int, int]], width: int = 64) -> Image.Image:
        """Convert pixel list to PNG image"""
        height = (len(pixels) + width - 1) // width  # Ceiling division

        # Create image
        img = Image.new('RGBA', (width, height), (0, 0, 0, 255))

        # Write pixels
        for i, pixel in enumerate(pixels):
            x = i % width
            y = i // width
            img.putpixel((x, y), pixel)

        return img


def main():
    parser = argparse.ArgumentParser(description='Pixel ISA Assembler')
    parser.add_argument('input', help='Input assembly file (.pxl)')
    parser.add_argument('-o', '--output', help='Output PNG file (.px)')
    parser.add_argument('-w', '--width', type=int, default=64,
                       help='PNG width in pixels (default: 64)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        output_file = args.input.rsplit('.', 1)[0] + '.px'

    # Read source file
    try:
        with open(args.input, 'r') as f:
            source = f.read()
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found")
        sys.exit(1)

    # Assemble
    assembler = PixelAssembler()
    try:
        pixels = assembler.assemble(source)
    except Exception as e:
        print(f"Assembly error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    if args.verbose:
        print(f"Assembled {len(assembler.instructions)} instructions")
        print(f"Labels: {assembler.labels}")
        print(f"Constants: {assembler.constants}")

    # Create PNG
    img = assembler.create_png(pixels, width=args.width)
    img.save(output_file, 'PNG')

    print(f"✓ Assembled '{args.input}' → '{output_file}'")
    print(f"  Size: {img.width}×{img.height} pixels ({len(pixels)} instructions)")


if __name__ == '__main__':
    main()
