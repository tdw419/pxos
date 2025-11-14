#!/usr/bin/env python3
"""
pxVM Assembler - Assembles pxVM assembly language to bytecode
pxOS Kernel v1.0 - Self-Hosting Edition

Syntax:
    LABEL:              ; Define label
    OPCODE arg1, arg2   ; Instruction
    ; comment           ; Comment

Supported opcodes:
    HALT
    NOP
    IMM8 reg, val
    IMM32 reg, val
    MOV dst, src
    ADD dst, src1, src2
    SUB dst, src1, src2
    JMP addr/label
    JZ reg, addr/label
    CMP dst, reg1, reg2
    LOAD dst, addr
    STORE addr, src
    SYSCALL num

Registers: R0-R7
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import re

# Import opcodes from VM
from pxvm_extended import (
    OP_HALT, OP_NOP, OP_IMM8, OP_IMM32, OP_MOV, OP_ADD, OP_SUB, OP_SYSCALL,
    OP_JMP, OP_JZ, OP_CMP, OP_LOAD, OP_STORE
)


class AssemblyError(Exception):
    """Assembly error with line number"""
    def __init__(self, line_num: int, message: str):
        self.line_num = line_num
        self.message = message
        super().__init__(f"Line {line_num}: {message}")


class Assembler:
    """pxVM assembler"""

    OPCODES = {
        'HALT': OP_HALT,
        'NOP': OP_NOP,
        'IMM8': OP_IMM8,
        'IMM32': OP_IMM32,
        'MOV': OP_MOV,
        'ADD': OP_ADD,
        'SUB': OP_SUB,
        'JMP': OP_JMP,
        'JZ': OP_JZ,
        'CMP': OP_CMP,
        'LOAD': OP_LOAD,
        'STORE': OP_STORE,
        'SYSCALL': OP_SYSCALL,
    }

    REGISTERS = {f'R{i}': i for i in range(8)}

    def __init__(self, imperfect: bool = True):
        self.imperfect = imperfect
        self.labels: Dict[str, int] = {}
        self.fixups: List[Tuple[int, str, int]] = []  # (addr, label, line_num)
        self.output: bytearray = bytearray()
        self.line_num = 0

    def error(self, msg: str):
        """Raise or log error depending on imperfect mode"""
        if self.imperfect:
            print(f"[ASM WARNING] Line {self.line_num}: {msg}")
        else:
            raise AssemblyError(self.line_num, msg)

    def parse_register(self, s: str) -> Optional[int]:
        """Parse register name (R0-R7)"""
        s = s.strip().upper()
        return self.REGISTERS.get(s)

    def parse_immediate(self, s: str) -> Optional[int]:
        """Parse immediate value (decimal or hex)"""
        s = s.strip()
        try:
            if s.startswith('0x') or s.startswith('0X'):
                return int(s, 16)
            else:
                return int(s, 10)
        except ValueError:
            return None

    def parse_operand(self, s: str) -> Tuple[str, int]:
        """Parse operand. Returns (type, value)
        Types: 'reg', 'imm', 'label'
        """
        s = s.strip()

        # Try register
        reg = self.parse_register(s)
        if reg is not None:
            return ('reg', reg)

        # Try immediate
        imm = self.parse_immediate(s)
        if imm is not None:
            return ('imm', imm)

        # Must be label
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', s):
            return ('label', 0)  # value resolved later

        self.error(f"Invalid operand: {s}")
        return ('imm', 0)

    def emit_byte(self, b: int):
        """Emit a single byte"""
        self.output.append(b & 0xFF)

    def emit_int32(self, val: int):
        """Emit 32-bit integer (little-endian, signed)"""
        if val < 0:
            val += 0x100000000
        self.emit_byte(val & 0xFF)
        self.emit_byte((val >> 8) & 0xFF)
        self.emit_byte((val >> 16) & 0xFF)
        self.emit_byte((val >> 24) & 0xFF)

    def current_addr(self) -> int:
        """Get current output address"""
        return len(self.output)

    def assemble_line(self, line: str):
        """Assemble one line"""
        # Remove comments
        if ';' in line:
            line = line[:line.index(';')]
        line = line.strip()

        if not line:
            return  # empty line

        # Check for label
        if ':' in line:
            label, rest = line.split(':', 1)
            label = label.strip()
            if label:
                self.labels[label] = self.current_addr()
            line = rest.strip()
            if not line:
                return

        # Parse instruction
        parts = [p.strip() for p in line.replace(',', ' ').split()]
        if not parts:
            return

        opcode_name = parts[0].upper()
        args = parts[1:]

        if opcode_name not in self.OPCODES:
            self.error(f"Unknown opcode: {opcode_name}")
            return

        opcode = self.OPCODES[opcode_name]
        self.emit_byte(opcode)

        # Encode arguments based on opcode
        if opcode == OP_HALT or opcode == OP_NOP:
            # No arguments
            pass

        elif opcode == OP_IMM8:
            # IMM8 reg, val
            if len(args) < 2:
                self.error("IMM8 requires 2 arguments")
                return
            reg_type, reg = self.parse_operand(args[0])
            val_type, val = self.parse_operand(args[1])
            if reg_type != 'reg':
                self.error("IMM8 arg1 must be register")
                return
            if val_type not in ('imm', 'label'):
                self.error("IMM8 arg2 must be immediate")
                return
            self.emit_byte(reg)
            self.emit_byte(val & 0xFF)

        elif opcode == OP_IMM32:
            # IMM32 reg, val
            if len(args) < 2:
                self.error("IMM32 requires 2 arguments")
                return
            reg_type, reg = self.parse_operand(args[0])
            val_type, val = self.parse_operand(args[1])
            if reg_type != 'reg':
                self.error("IMM32 arg1 must be register")
                return

            self.emit_byte(reg)  # Emit register

            if val_type == 'label':
                # Record fixup
                self.fixups.append((self.current_addr(), args[1], self.line_num))
                self.emit_int32(0)  # placeholder
            else:
                self.emit_int32(val)

        elif opcode == OP_MOV:
            # MOV dst, src
            if len(args) < 2:
                self.error("MOV requires 2 arguments")
                return
            dst_type, dst = self.parse_operand(args[0])
            src_type, src = self.parse_operand(args[1])
            if dst_type != 'reg' or src_type != 'reg':
                self.error("MOV requires register arguments")
                return
            self.emit_byte(dst)
            self.emit_byte(src)

        elif opcode in (OP_ADD, OP_SUB, OP_CMP):
            # ADD/SUB/CMP dst, src1, src2
            if len(args) < 3:
                self.error(f"{opcode_name} requires 3 arguments")
                return
            dst_type, dst = self.parse_operand(args[0])
            s1_type, s1 = self.parse_operand(args[1])
            s2_type, s2 = self.parse_operand(args[2])
            if dst_type != 'reg' or s1_type != 'reg' or s2_type != 'reg':
                self.error(f"{opcode_name} requires register arguments")
                return
            self.emit_byte(dst)
            self.emit_byte(s1)
            self.emit_byte(s2)

        elif opcode == OP_JMP:
            # JMP addr/label
            if len(args) < 1:
                self.error("JMP requires 1 argument")
                return
            addr_type, addr = self.parse_operand(args[0])

            if addr_type == 'label':
                self.fixups.append((self.current_addr(), args[0], self.line_num))
                self.emit_int32(0)  # placeholder
            else:
                self.emit_int32(addr)

        elif opcode == OP_JZ:
            # JZ reg, addr/label
            if len(args) < 2:
                self.error("JZ requires 2 arguments")
                return
            reg_type, reg = self.parse_operand(args[0])
            addr_type, addr = self.parse_operand(args[1])

            if reg_type != 'reg':
                self.error("JZ arg1 must be register")
                return

            self.emit_byte(reg)

            if addr_type == 'label':
                self.fixups.append((self.current_addr(), args[1], self.line_num))
                self.emit_int32(0)  # placeholder
            else:
                self.emit_int32(addr)

        elif opcode == OP_LOAD:
            # LOAD dst, addr
            if len(args) < 2:
                self.error("LOAD requires 2 arguments")
                return
            dst_type, dst = self.parse_operand(args[0])
            addr_type, addr = self.parse_operand(args[1])

            if dst_type != 'reg':
                self.error("LOAD arg1 must be register")
                return

            self.emit_byte(dst)

            if addr_type == 'label':
                self.fixups.append((self.current_addr(), args[1], self.line_num))
                self.emit_int32(0)
            else:
                self.emit_int32(addr)

        elif opcode == OP_STORE:
            # STORE addr, src
            if len(args) < 2:
                self.error("STORE requires 2 arguments")
                return
            addr_type, addr = self.parse_operand(args[0])
            src_type, src = self.parse_operand(args[1])

            if src_type != 'reg':
                self.error("STORE arg2 must be register")
                return

            if addr_type == 'label':
                self.fixups.append((self.current_addr(), args[0], self.line_num))
                self.emit_int32(0)
            else:
                self.emit_int32(addr)

            self.emit_byte(src)

        elif opcode == OP_SYSCALL:
            # SYSCALL num
            if len(args) < 1:
                self.error("SYSCALL requires 1 argument")
                return
            num_type, num = self.parse_operand(args[0])
            if num_type != 'imm':
                self.error("SYSCALL argument must be immediate")
                return
            self.emit_byte(num & 0xFF)

    def resolve_fixups(self):
        """Resolve label references"""
        for addr, label, line_num in self.fixups:
            if label not in self.labels:
                self.line_num = line_num
                self.error(f"Undefined label: {label}")
                continue

            target = self.labels[label]

            # Write 32-bit address at fixup location
            if target < 0:
                target += 0x100000000
            self.output[addr] = target & 0xFF
            self.output[addr + 1] = (target >> 8) & 0xFF
            self.output[addr + 2] = (target >> 16) & 0xFF
            self.output[addr + 3] = (target >> 24) & 0xFF

    def assemble(self, source: str) -> bytes:
        """Assemble source code to bytecode"""
        self.labels.clear()
        self.fixups.clear()
        self.output.clear()

        lines = source.split('\n')
        for i, line in enumerate(lines, 1):
            self.line_num = i
            try:
                self.assemble_line(line)
            except Exception as e:
                if not self.imperfect:
                    raise
                print(f"[ASM ERROR] Line {i}: {e}")

        # Resolve labels
        self.resolve_fixups()

        return bytes(self.output)

    def assemble_file(self, input_path: str, output_path: Optional[str] = None) -> bytes:
        """Assemble file"""
        with open(input_path, 'r') as f:
            source = f.read()

        bytecode = self.assemble(source)

        if output_path:
            with open(output_path, 'wb') as f:
                f.write(bytecode)
            print(f"Assembled {len(bytecode)} bytes → {output_path}")

        return bytecode


def main():
    """Command-line interface"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: pxvm_assembler.py <input.asm> [output.bin]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not output_file:
        # Auto-generate output filename
        if input_file.endswith('.asm'):
            output_file = input_file[:-4] + '.bin'
        else:
            output_file = input_file + '.bin'

    asm = Assembler(imperfect=True)
    try:
        bytecode = asm.assemble_file(input_file, output_file)
        print(f"✓ Assembly successful: {len(bytecode)} bytes")
        print(f"  Labels: {list(asm.labels.keys())}")
    except Exception as e:
        print(f"✗ Assembly failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
