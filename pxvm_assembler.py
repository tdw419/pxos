#!/usr/bin/env python3
"""
pxVM Assembler - Convert assembly language to bytecode

Syntax:
    LABEL:              # Define a label
    MNEMONIC rd, rs, rt # Three-operand instruction
    MNEMONIC rd, imm32  # Immediate instruction
    ; comment           # Comment

Example:
    main:
        IMM32 R0, 5
        CALL factorial
        PRINT R0
        HALT

    factorial:
        IMM32 R1, 1
        SUB R2, R0, R1
        JZ R2, base_case
        PUSH R0
        MOV R0, R2
        CALL factorial
        POP R1
        MUL R0, R0, R1
        RET

    base_case:
        IMM32 R0, 1
        RET
"""

import re
import struct
import sys
from typing import List, Dict, Tuple, Optional

class AssemblyError(Exception):
    """Assembly error with line number"""
    def __init__(self, line_num: int, message: str):
        self.line_num = line_num
        self.message = message
        super().__init__(f"Line {line_num}: {message}")

class PxVmAssembler:
    # Opcode definitions (must match spirv_terminal.py)
    OPCODES = {
        'MOV':    0x10,
        'ADD':    0x20,
        'SUB':    0x21,
        'MUL':    0x22,
        'DIV':    0x23,
        'IMM32':  0x30,
        'LOAD':   0x40,
        'STORE':  0x41,
        'PUSH':   0x50,
        'POP':    0x51,
        'CALL':   0x60,
        'RET':    0x61,
        'JMP':    0x70,
        'JZ':     0x71,
        'JNZ':    0x72,
        'SYSCALL': 0x80,
        'PRINT':  0xF0,
        'HALT':   0xFF,
    }

    # Instruction formats
    FORMATS = {
        'MOV':    'RR',      # MOV rd, rs
        'ADD':    'RRR',     # ADD rd, rs, rt
        'SUB':    'RRR',
        'MUL':    'RRR',
        'DIV':    'RRR',
        'IMM32':  'RI',      # IMM32 rd, imm32
        'LOAD':   'RR',      # LOAD rd, [rs]
        'STORE':  'RR',      # STORE [rd], rs
        'PUSH':   'R',       # PUSH rs
        'POP':    'R',       # POP rd
        'CALL':   'L',       # CALL label or CALL addr
        'RET':    '',        # RET (no operands)
        'JMP':    'L',       # JMP label
        'JZ':     'RL',      # JZ rs, label
        'JNZ':    'RL',      # JNZ rs, label
        'SYSCALL': '',       # SYSCALL (no operands, uses registers)
        'PRINT':  'R',       # PRINT rd
        'HALT':   '',        # HALT
    }

    def __init__(self):
        self.labels: Dict[str, int] = {}
        self.bytecode: List[int] = []
        self.current_address = 0

    def parse_register(self, reg_str: str, line_num: int) -> int:
        """Parse register name (R0-R7) to register number"""
        reg_str = reg_str.strip().upper()
        if not reg_str.startswith('R'):
            raise AssemblyError(line_num, f"Invalid register: {reg_str}")
        try:
            reg_num = int(reg_str[1:])
            if reg_num < 0 or reg_num > 7:
                raise AssemblyError(line_num, f"Register out of range: {reg_str}")
            return reg_num
        except ValueError:
            raise AssemblyError(line_num, f"Invalid register: {reg_str}")

    def parse_immediate(self, imm_str: str, line_num: int) -> int:
        """Parse immediate value (decimal, hex, or binary)"""
        imm_str = imm_str.strip()
        try:
            if imm_str.startswith('0x') or imm_str.startswith('0X'):
                return int(imm_str, 16)
            elif imm_str.startswith('0b') or imm_str.startswith('0B'):
                return int(imm_str, 2)
            else:
                return int(imm_str)
        except ValueError:
            raise AssemblyError(line_num, f"Invalid immediate value: {imm_str}")

    def preprocess(self, source: str) -> List[Tuple[int, str, List[str]]]:
        """
        Preprocess source code:
        - Remove comments
        - Extract labels
        - Split into (line_num, mnemonic, operands)
        """
        lines = []
        for line_num, line in enumerate(source.split('\n'), 1):
            # Remove comments
            line = re.sub(r';.*$', '', line)
            line = re.sub(r'#.*$', '', line)
            line = line.strip()

            if not line:
                continue

            # Check for label
            if ':' in line:
                label, rest = line.split(':', 1)
                label = label.strip()
                if label:
                    self.labels[label] = self.current_address
                line = rest.strip()
                if not line:
                    continue

            # Parse instruction
            parts = re.split(r'[,\s]+', line)
            mnemonic = parts[0].upper()
            operands = [op.strip() for op in parts[1:] if op.strip()]

            lines.append((line_num, mnemonic, operands))

            # Calculate address for next instruction
            if mnemonic in self.OPCODES:
                self.current_address += self.instruction_size(mnemonic, operands)

        return lines

    def instruction_size(self, mnemonic: str, operands: List[str]) -> int:
        """Calculate instruction size in bytes"""
        if mnemonic not in self.OPCODES:
            return 0

        fmt = self.FORMATS[mnemonic]
        size = 1  # Opcode

        if fmt == 'RR':
            size += 2  # Two register bytes
        elif fmt == 'RRR':
            size += 3  # Three register bytes
        elif fmt == 'RI':
            size += 1 + 4  # Register + 32-bit immediate
        elif fmt == 'R':
            size += 1  # One register byte
        elif fmt == 'L':
            size += 2  # 16-bit address
        elif fmt == 'RL':
            size += 1 + 2  # Register + 16-bit address

        return size

    def emit_byte(self, byte: int):
        """Emit a single byte to bytecode"""
        self.bytecode.append(byte & 0xFF)

    def emit_word(self, word: int):
        """Emit a 16-bit word (little-endian)"""
        self.bytecode.extend(struct.pack('<H', word & 0xFFFF))

    def emit_dword(self, dword: int):
        """Emit a 32-bit word (little-endian)"""
        self.bytecode.extend(struct.pack('<I', dword & 0xFFFFFFFF))

    def assemble_instruction(self, line_num: int, mnemonic: str, operands: List[str]):
        """Assemble a single instruction"""
        if mnemonic not in self.OPCODES:
            raise AssemblyError(line_num, f"Unknown mnemonic: {mnemonic}")

        opcode = self.OPCODES[mnemonic]
        fmt = self.FORMATS[mnemonic]

        # Emit opcode
        self.emit_byte(opcode)

        # Emit operands based on format
        if fmt == '':
            # No operands (RET, HALT)
            if operands:
                raise AssemblyError(line_num, f"{mnemonic} takes no operands")

        elif fmt == 'R':
            # Single register (PUSH, POP, PRINT)
            if len(operands) != 1:
                raise AssemblyError(line_num, f"{mnemonic} requires 1 operand")
            rd = self.parse_register(operands[0], line_num)
            self.emit_byte(rd)

        elif fmt == 'RR':
            # Two registers (MOV, LOAD, STORE)
            if len(operands) != 2:
                raise AssemblyError(line_num, f"{mnemonic} requires 2 operands")

            # Handle LOAD/STORE special syntax: LOAD R1, [R2]
            op1 = operands[0].strip()
            op2 = operands[1].strip()

            # Remove brackets for LOAD/STORE
            if op2.startswith('[') and op2.endswith(']'):
                op2 = op2[1:-1]

            rd = self.parse_register(op1, line_num)
            rs = self.parse_register(op2, line_num)
            self.emit_byte(rd)
            self.emit_byte(rs)

        elif fmt == 'RRR':
            # Three registers (ADD, SUB, MUL, DIV)
            if len(operands) != 3:
                raise AssemblyError(line_num, f"{mnemonic} requires 3 operands")
            rd = self.parse_register(operands[0], line_num)
            rs = self.parse_register(operands[1], line_num)
            rt = self.parse_register(operands[2], line_num)
            self.emit_byte(rd)
            self.emit_byte(rs)
            self.emit_byte(rt)

        elif fmt == 'RI':
            # Register + immediate (IMM32)
            if len(operands) != 2:
                raise AssemblyError(line_num, f"{mnemonic} requires 2 operands")
            rd = self.parse_register(operands[0], line_num)
            imm = self.parse_immediate(operands[1], line_num)
            self.emit_byte(rd)
            self.emit_dword(imm)

        elif fmt == 'L':
            # Label/address (CALL, JMP)
            if len(operands) != 1:
                raise AssemblyError(line_num, f"{mnemonic} requires 1 operand")

            # Check if it's a label or immediate address
            label = operands[0].strip()
            if label in self.labels:
                addr = self.labels[label]
            else:
                # Try to parse as immediate address
                addr = self.parse_immediate(label, line_num)

            self.emit_word(addr)

        elif fmt == 'RL':
            # Register + label (JZ, JNZ)
            if len(operands) != 2:
                raise AssemblyError(line_num, f"{mnemonic} requires 2 operands")

            rs = self.parse_register(operands[0], line_num)
            label = operands[1].strip()

            if label in self.labels:
                addr = self.labels[label]
            else:
                addr = self.parse_immediate(label, line_num)

            self.emit_byte(rs)
            self.emit_word(addr)

    def assemble(self, source: str) -> bytes:
        """
        Assemble source code to bytecode

        Args:
            source: Assembly source code

        Returns:
            Bytecode as bytes
        """
        # Reset state
        self.labels = {}
        self.bytecode = []
        self.current_address = 0

        # First pass: preprocess and collect labels
        lines = self.preprocess(source)

        # Reset for second pass
        self.bytecode = []
        self.current_address = 0

        # Second pass: generate code
        for line_num, mnemonic, operands in lines:
            self.assemble_instruction(line_num, mnemonic, operands)

        return bytes(self.bytecode)

    def disassemble(self, bytecode: bytes) -> str:
        """
        Disassemble bytecode to assembly (for debugging)

        Args:
            bytecode: Bytecode to disassemble

        Returns:
            Assembly source code
        """
        # Create reverse opcode map
        MNEMONICS = {v: k for k, v in self.OPCODES.items()}

        lines = []
        pc = 0

        while pc < len(bytecode):
            addr = pc
            opcode = bytecode[pc]
            pc += 1

            if opcode not in MNEMONICS:
                lines.append(f"{addr:04X}: .byte 0x{opcode:02X}  ; Unknown opcode")
                continue

            mnemonic = MNEMONICS[opcode]
            fmt = self.FORMATS[mnemonic]

            # Disassemble based on format
            if fmt == '':
                lines.append(f"{addr:04X}: {mnemonic}")

            elif fmt == 'R':
                rd = bytecode[pc]
                pc += 1
                lines.append(f"{addr:04X}: {mnemonic} R{rd}")

            elif fmt == 'RR':
                rd = bytecode[pc]
                rs = bytecode[pc + 1]
                pc += 2
                if mnemonic in ['LOAD', 'STORE']:
                    lines.append(f"{addr:04X}: {mnemonic} R{rd}, [R{rs}]")
                else:
                    lines.append(f"{addr:04X}: {mnemonic} R{rd}, R{rs}")

            elif fmt == 'RRR':
                rd = bytecode[pc]
                rs = bytecode[pc + 1]
                rt = bytecode[pc + 2]
                pc += 3
                lines.append(f"{addr:04X}: {mnemonic} R{rd}, R{rs}, R{rt}")

            elif fmt == 'RI':
                rd = bytecode[pc]
                pc += 1
                imm = struct.unpack('<I', bytecode[pc:pc+4])[0]
                pc += 4
                lines.append(f"{addr:04X}: {mnemonic} R{rd}, {imm}")

            elif fmt == 'L':
                addr_val = struct.unpack('<H', bytecode[pc:pc+2])[0]
                pc += 2
                lines.append(f"{addr:04X}: {mnemonic} 0x{addr_val:04X}")

            elif fmt == 'RL':
                rs = bytecode[pc]
                pc += 1
                addr_val = struct.unpack('<H', bytecode[pc:pc+2])[0]
                pc += 2
                lines.append(f"{addr:04X}: {mnemonic} R{rs}, 0x{addr_val:04X}")

        return '\n'.join(lines)

def main():
    """Command-line assembler"""
    if len(sys.argv) < 2:
        print("Usage: pxvm_assembler.py <input.pxasm> [output.pxvm]")
        print("       pxvm_assembler.py -d <input.pxvm>  (disassemble)")
        sys.exit(1)

    if sys.argv[1] == '-d':
        # Disassemble mode
        if len(sys.argv) < 3:
            print("Error: No input file specified")
            sys.exit(1)

        with open(sys.argv[2], 'rb') as f:
            bytecode = f.read()

        assembler = PxVmAssembler()
        asm = assembler.disassemble(bytecode)
        print(asm)

    else:
        # Assemble mode
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.pxasm', '.pxvm')

        with open(input_file, 'r') as f:
            source = f.read()

        try:
            assembler = PxVmAssembler()
            bytecode = assembler.assemble(source)

            with open(output_file, 'wb') as f:
                f.write(bytecode)

            print(f"Assembled {len(bytecode)} bytes to {output_file}")

            # Show label addresses
            if assembler.labels:
                print("\nLabels:")
                for label, addr in sorted(assembler.labels.items(), key=lambda x: x[1]):
                    print(f"  {label:20s} = 0x{addr:04X}")

        except AssemblyError as e:
            print(f"Error: {e}")
            sys.exit(1)

if __name__ == '__main__':
    main()
