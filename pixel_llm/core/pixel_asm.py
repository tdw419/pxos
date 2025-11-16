#!/usr/bin/env python3
"""
Pixel Assembly Language - Human-readable syntax for Pixel IR

Compiles .pxasm text files to Pixel IR bytecode (.pxi).

Example:
    ; Count down from 5
    PUSH 5
    loop_start:
        DUP
        PRINT
        PUSH 1
        SUB
        DUP
        JNZ loop_start
    HALT

Philosophy:
"Assembly is the middle ground:
 - More readable than raw bytes
 - More explicit than high-level languages
 - Perfect for understanding the substrate"
"""

import re
import struct
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Bootstrap path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@dataclass
class Instruction:
    """Parsed assembly instruction"""
    label: Optional[str]
    opcode: str
    operands: List[str]
    line_num: int
    comment: Optional[str]


class PixelAssembler:
    """Assembles Pixel Assembly to Pixel IR bytecode"""

    # Opcode definitions (name → byte value)
    OPCODES = {
        # Stack
        'PUSH': 0x01,
        'POP': 0x02,
        'DUP': 0x11,
        'SWAP': 0x12,

        # Arithmetic
        'ADD': 0x03,
        'SUB': 0x04,
        'MUL': 0x05,
        'DIV': 0x06,
        'MOD': 0x07,

        # Comparison
        'EQ': 0x20,
        'LT': 0x21,
        'GT': 0x22,

        # Control flow
        'JMP': 0x30,
        'JZ': 0x31,
        'JNZ': 0x32,
        'CALL': 0x33,
        'RET': 0x34,

        # Memory
        'LOAD': 0x40,
        'STORE': 0x41,

        # I/O
        'PRINT': 0x10,
        'PRINT_STR': 0x13,

        # Host calls
        'HOST_CALL': 0x50,

        # System
        'HALT': 0xFF,
        'NOP': 0xFE,
    }

    # Opcodes that take operands
    OPERAND_TYPES = {
        'PUSH': ['int'],
        'PRINT_STR': ['string'],
        'JMP': ['label'],
        'JZ': ['label'],
        'JNZ': ['label'],
        'CALL': ['label'],
        'HOST_CALL': ['int'],
    }

    def __init__(self):
        self.labels: Dict[str, int] = {}
        self.instructions: List[Instruction] = []
        self.bytecode = bytearray()

    def assemble(self, source: str) -> bytes:
        """Assemble source code to bytecode"""
        self.labels.clear()
        self.instructions.clear()
        self.bytecode.clear()

        # Pass 1: Parse and collect labels
        self._parse(source)
        self._calculate_label_offsets()

        # Pass 2: Generate bytecode
        self._generate_bytecode()

        return bytes(self.bytecode)

    def _parse(self, source: str):
        """Parse assembly source into instructions"""
        for line_num, line in enumerate(source.split('\n'), 1):
            # Remove comments
            if ';' in line:
                line, comment = line.split(';', 1)
                comment = comment.strip()
            else:
                comment = None

            line = line.strip()
            if not line:
                continue

            # Check for label
            label = None
            if ':' in line and not line.startswith('"'):
                label, line = line.split(':', 1)
                label = label.strip()
                line = line.strip()

            # Parse instruction
            if line:
                parts = self._tokenize(line)
                if parts:
                    opcode = parts[0].upper()
                    operands = parts[1:]

                    instr = Instruction(
                        label=label,
                        opcode=opcode,
                        operands=operands,
                        line_num=line_num,
                        comment=comment
                    )
                    self.instructions.append(instr)
            elif label:
                # Label with no instruction on same line
                instr = Instruction(
                    label=label,
                    opcode='NOP',  # Implicit NOP
                    operands=[],
                    line_num=line_num,
                    comment=comment
                )
                self.instructions.append(instr)

    def _tokenize(self, line: str) -> List[str]:
        """Tokenize a line into opcode and operands"""
        tokens = []
        current = []
        in_string = False

        for char in line:
            if char == '"':
                in_string = not in_string
                current.append(char)
            elif char in ' \t' and not in_string:
                if current:
                    tokens.append(''.join(current))
                    current = []
            else:
                current.append(char)

        if current:
            tokens.append(''.join(current))

        return tokens

    def _calculate_label_offsets(self):
        """Calculate byte offsets for all labels"""
        offset = 0

        for instr in self.instructions:
            # Register label at current offset
            if instr.label:
                self.labels[instr.label] = offset

            # Calculate instruction size
            if instr.opcode == 'NOP' and not instr.operands:
                # Implicit NOP from label-only line - no bytes
                continue

            offset += 1  # Opcode byte

            if instr.opcode in self.OPERAND_TYPES:
                operand_types = self.OPERAND_TYPES[instr.opcode]
                for op_type in operand_types:
                    if op_type == 'int':
                        offset += 4  # 32-bit int
                    elif op_type == 'label':
                        offset += 4  # 32-bit address
                    elif op_type == 'string':
                        # String is length (4 bytes) + UTF-8 bytes
                        string_val = self._parse_string(instr.operands[0])
                        offset += 4 + len(string_val)

    def _generate_bytecode(self):
        """Generate bytecode from parsed instructions"""
        for instr in self.instructions:
            # Skip implicit NOPs from label-only lines
            if instr.opcode == 'NOP' and not instr.operands and instr.label:
                continue

            # Emit opcode
            if instr.opcode not in self.OPCODES:
                raise AssemblyError(
                    f"Unknown opcode '{instr.opcode}' at line {instr.line_num}"
                )

            opcode_byte = self.OPCODES[instr.opcode]
            self.bytecode.append(opcode_byte)

            # Emit operands
            if instr.opcode in self.OPERAND_TYPES:
                self._emit_operands(instr)

    def _emit_operands(self, instr: Instruction):
        """Emit operands for an instruction"""
        operand_types = self.OPERAND_TYPES[instr.opcode]

        if len(instr.operands) != len(operand_types):
            raise AssemblyError(
                f"{instr.opcode} expects {len(operand_types)} operand(s), "
                f"got {len(instr.operands)} at line {instr.line_num}"
            )

        for operand, op_type in zip(instr.operands, operand_types):
            if op_type == 'int':
                # Parse as 32-bit signed integer
                try:
                    value = int(operand, 0)  # 0 = auto-detect base (0x, 0b, etc.)
                except ValueError:
                    raise AssemblyError(
                        f"Invalid integer '{operand}' at line {instr.line_num}"
                    )
                self.bytecode.extend(struct.pack('<i', value))

            elif op_type == 'label':
                # Resolve label to offset
                if operand not in self.labels:
                    raise AssemblyError(
                        f"Undefined label '{operand}' at line {instr.line_num}"
                    )
                offset = self.labels[operand]
                self.bytecode.extend(struct.pack('<i', offset))

            elif op_type == 'string':
                # Encode string as length + UTF-8 bytes
                string_val = self._parse_string(operand)
                self.bytecode.extend(struct.pack('<i', len(string_val)))
                self.bytecode.extend(string_val)

    def _parse_string(self, s: str) -> bytes:
        """Parse string literal to bytes"""
        if not (s.startswith('"') and s.endswith('"')):
            raise AssemblyError(f"String must be quoted: {s}")

        # Remove quotes and decode escape sequences
        s = s[1:-1]
        s = s.replace('\\n', '\n')
        s = s.replace('\\t', '\t')
        s = s.replace('\\r', '\r')
        s = s.replace('\\"', '"')
        s = s.replace('\\\\', '\\')

        return s.encode('utf-8')


class PixelDisassembler:
    """Disassembles Pixel IR bytecode to assembly"""

    # Reverse opcode map (byte → name)
    OPCODES = {v: k for k, v in PixelAssembler.OPCODES.items()}

    def __init__(self):
        self.pc = 0
        self.bytecode = b''
        self.output = []

    def disassemble(self, bytecode: bytes) -> str:
        """Disassemble bytecode to assembly source"""
        self.bytecode = bytecode
        self.pc = 0
        self.output = []

        while self.pc < len(self.bytecode):
            offset = self.pc
            instr = self._read_instruction()
            self.output.append(f"{offset:04X}: {instr}")

        return '\n'.join(self.output)

    def _read_instruction(self) -> str:
        """Read and format one instruction"""
        opcode_byte = self.bytecode[self.pc]
        self.pc += 1

        if opcode_byte not in self.OPCODES:
            return f"UNKNOWN({opcode_byte:02X})"

        opcode = self.OPCODES[opcode_byte]

        # Read operands
        if opcode in PixelAssembler.OPERAND_TYPES:
            operand_types = PixelAssembler.OPERAND_TYPES[opcode]
            operands = []

            for op_type in operand_types:
                if op_type == 'int' or op_type == 'label':
                    value = struct.unpack('<i', self.bytecode[self.pc:self.pc+4])[0]
                    self.pc += 4
                    if op_type == 'label':
                        operands.append(f"0x{value:04X}")
                    else:
                        operands.append(str(value))

                elif op_type == 'string':
                    length = struct.unpack('<i', self.bytecode[self.pc:self.pc+4])[0]
                    self.pc += 4
                    string_bytes = self.bytecode[self.pc:self.pc+length]
                    self.pc += length
                    string_val = string_bytes.decode('utf-8')
                    operands.append(f'"{string_val}"')

            return f"{opcode} {', '.join(operands)}"
        else:
            return opcode


class AssemblyError(Exception):
    """Assembly error with line number context"""
    pass


def assemble_file(input_path: Path, output_path: Optional[Path] = None) -> bytes:
    """Assemble a .pxasm file to .pxi bytecode"""
    source = input_path.read_text()

    assembler = PixelAssembler()
    bytecode = assembler.assemble(source)

    if output_path:
        output_path.write_bytes(bytecode)
        print(f"✅ Assembled {len(bytecode)} bytes → {output_path}")

    return bytecode


def disassemble_file(input_path: Path, output_path: Optional[Path] = None) -> str:
    """Disassemble a .pxi file to .pxasm assembly"""
    bytecode = input_path.read_bytes()

    disassembler = PixelDisassembler()
    assembly = disassembler.disassemble(bytecode)

    if output_path:
        output_path.write_text(assembly)
        print(f"✅ Disassembled {len(bytecode)} bytes → {output_path}")

    return assembly


def main():
    """CLI for assembler/disassembler"""
    import argparse

    parser = argparse.ArgumentParser(description="Pixel Assembly Language Tools")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Assemble command
    asm_parser = subparsers.add_parser('asm', help='Assemble .pxasm to .pxi')
    asm_parser.add_argument('input', type=Path, help='Input .pxasm file')
    asm_parser.add_argument('-o', '--output', type=Path, help='Output .pxi file')

    # Disassemble command
    disasm_parser = subparsers.add_parser('disasm', help='Disassemble .pxi to .pxasm')
    disasm_parser.add_argument('input', type=Path, help='Input .pxi file')
    disasm_parser.add_argument('-o', '--output', type=Path, help='Output .pxasm file')

    args = parser.parse_args()

    if args.command == 'asm':
        output = args.output or args.input.with_suffix('.pxi')
        assemble_file(args.input, output)

    elif args.command == 'disasm':
        output = args.output or args.input.with_suffix('.pxasm')
        disassemble_file(args.input, output)


if __name__ == '__main__':
    main()
