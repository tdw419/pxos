#!/usr/bin/env python3
"""
PXI Assembly IR Compiler v1.0

Compiles PXI Assembly (Intermediate Representation) to pxOS primitives.

This is the ONLY module that needs to know about x86 opcodes. All language
compilers (Python, C, etc.) target PXI Assembly, not raw opcodes.

Architecture:
    Source Language → PXI Assembly IR → This Compiler → pxOS Primitives → Binary

Usage:
    ir_compiler.py program.pxi -o primitives.txt
    ir_compiler.py program.pxi.json --format json -o primitives.txt
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# x86 Opcode Knowledge (ONLY place this exists!)
# ============================================================================

class X86:
    """x86 16-bit real mode opcode generator"""

    @staticmethod
    def MOV_REG_IMM16(reg: str, value: int) -> List[int]:
        """MOV reg16, imm16"""
        reg_codes = {'AX': 0xB8, 'BX': 0xBB, 'CX': 0xB9, 'DX': 0xBA,
                     'SI': 0xBE, 'DI': 0xBF, 'SP': 0xBC, 'BP': 0xBD}
        if reg not in reg_codes:
            raise ValueError(f"Invalid 16-bit register: {reg}")
        return [reg_codes[reg], value & 0xFF, (value >> 8) & 0xFF]

    @staticmethod
    def MOV_REG8_IMM8(reg: str, value: int) -> List[int]:
        """MOV reg8, imm8"""
        reg_codes = {'AL': 0xB0, 'AH': 0xB4, 'BL': 0xB3, 'BH': 0xB7,
                     'CL': 0xB1, 'CH': 0xB5, 'DL': 0xB2, 'DH': 0xB6}
        if reg not in reg_codes:
            raise ValueError(f"Invalid 8-bit register: {reg}")
        return [reg_codes[reg], value & 0xFF]

    @staticmethod
    def MOV_SEG_REG(seg: str, reg: str) -> List[int]:
        """MOV seg, reg"""
        if seg == 'ES' and reg == 'AX':
            return [0x8E, 0xC0]
        elif seg == 'DS' and reg == 'AX':
            return [0x8E, 0xD8]
        elif seg == 'SS' and reg == 'AX':
            return [0x8E, 0xD0]
        else:
            raise ValueError(f"Unsupported MOV {seg}, {reg}")

    @staticmethod
    def XOR_REG_REG(dst: str, src: str) -> List[int]:
        """XOR reg, reg"""
        if dst == 'DI' and src == 'DI':
            return [0x31, 0xFF]
        elif dst == 'SI' and src == 'SI':
            return [0x31, 0xF6]
        elif dst == 'AX' and src == 'AX':
            return [0x31, 0xC0]
        else:
            raise ValueError(f"Unsupported XOR {dst}, {src}")

    @staticmethod
    def OR_REG_REG(dst: str, src: str) -> List[int]:
        """OR reg, reg"""
        if dst == 'AL' and src == 'AL':
            return [0x08, 0xC0]
        else:
            raise ValueError(f"Unsupported OR {dst}, {src}")

    @staticmethod
    def INT(num: int) -> List[int]:
        """INT imm8"""
        return [0xCD, num & 0xFF]

    @staticmethod
    def CALL_REL16(offset: int) -> List[int]:
        """CALL rel16"""
        # offset is relative to NEXT instruction (after the call)
        return [0xE8, offset & 0xFF, (offset >> 8) & 0xFF]

    @staticmethod
    def JMP_REL8(offset: int) -> List[int]:
        """JMP rel8 (short jump)"""
        return [0xEB, offset & 0xFF]

    @staticmethod
    def JZ_REL8(offset: int) -> List[int]:
        """JZ rel8"""
        return [0x74, offset & 0xFF]

    @staticmethod
    def JNZ_REL8(offset: int) -> List[int]:
        """JNZ rel8"""
        return [0x75, offset & 0xFF]

    @staticmethod
    def JE_REL8(offset: int) -> List[int]:
        """JE rel8 (same as JZ)"""
        return [0x74, offset & 0xFF]

    @staticmethod
    def JNE_REL8(offset: int) -> List[int]:
        """JNE rel8 (same as JNZ)"""
        return [0x75, offset & 0xFF]

    @staticmethod
    def RET() -> List[int]:
        """RET"""
        return [0xC3]

    @staticmethod
    def CLI() -> List[int]:
        """CLI"""
        return [0xFA]

    @staticmethod
    def STI() -> List[int]:
        """STI"""
        return [0xFB]

    @staticmethod
    def HLT() -> List[int]:
        """HLT"""
        return [0xF4]

    @staticmethod
    def NOP() -> List[int]:
        """NOP"""
        return [0x90]

    @staticmethod
    def LODSB() -> List[int]:
        """LODSB"""
        return [0xAC]

    @staticmethod
    def STOSB() -> List[int]:
        """STOSB"""
        return [0xAA]

    @staticmethod
    def STOSW() -> List[int]:
        """STOSW"""
        return [0xAB]

    @staticmethod
    def REP() -> List[int]:
        """REP prefix"""
        return [0xF3]

    @staticmethod
    def ADD_REG_IMM8(reg: str, value: int) -> List[int]:
        """ADD reg, imm8"""
        if reg == 'AL':
            return [0x04, value & 0xFF]
        else:
            raise ValueError(f"Unsupported ADD {reg}, imm8")

    @staticmethod
    def CMP_REG_IMM8(reg: str, value: int) -> List[int]:
        """CMP reg, imm8"""
        if reg == 'AL':
            return [0x3C, value & 0xFF]
        else:
            raise ValueError(f"Unsupported CMP {reg}, imm8")


# ============================================================================
# IR Data Structures
# ============================================================================

@dataclass
class IRInstruction:
    """Single IR instruction"""
    op: str
    operands: Dict[str, Union[str, int]]
    line_num: int = 0
    comment: str = ""


@dataclass
class IRLabel:
    """Code label"""
    name: str
    address: Optional[int] = None


@dataclass
class IRData:
    """Data definition"""
    label: str
    data_type: str  # 'byte', 'word', 'string'
    value: Union[List[int], str]
    address: Optional[int] = None


@dataclass
class IRProgram:
    """Complete IR program"""
    instructions: List[IRInstruction] = field(default_factory=list)
    labels: Dict[str, IRLabel] = field(default_factory=dict)
    data: List[IRData] = field(default_factory=list)
    origin: int = 0x7C00


# ============================================================================
# IR Parser
# ============================================================================

class IRParser:
    """Parse PXI Assembly (text or JSON) into IR structures"""

    def __init__(self):
        self.program = IRProgram()

    def parse_json(self, json_data: dict) -> IRProgram:
        """Parse JSON format IR"""
        self.program.origin = json_data.get('origin', 0x7C00)

        # Parse data section
        for data_item in json_data.get('data', []):
            ir_data = IRData(
                label=data_item['label'],
                data_type=data_item['type'],
                value=data_item['value']
            )
            self.program.data.append(ir_data)
            self.program.labels[data_item['label']] = IRLabel(data_item['label'])

        # Parse instructions
        for idx, instr in enumerate(json_data.get('instructions', [])):
            # Handle labels
            if instr['op'] == 'LABEL':
                label_name = instr.get('operands', {}).get('name', instr.get('name', ''))
                if label_name:
                    self.program.labels[label_name] = IRLabel(label_name)
                continue

            # Handle comments (just skip for now)
            if instr['op'] == 'COMMENT':
                continue

            # Handle instructions
            ir_instr = IRInstruction(
                op=instr['op'],
                operands=instr.get('operands', {}),
                line_num=idx,
                comment=instr.get('comment', '')
            )
            self.program.instructions.append(ir_instr)

        return self.program

    def parse_text(self, text: str) -> IRProgram:
        """Parse text format PXI Assembly"""
        lines = text.split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith(';'):
                continue

            # Remove inline comments
            if ';' in line:
                line, comment = line.split(';', 1)
                line = line.strip()
            else:
                comment = ""

            # Parse labels
            if line.endswith(':'):
                label_name = line[:-1]
                self.program.labels[label_name] = IRLabel(label_name)
                continue

            # Parse instructions
            parts = line.split()
            if not parts:
                continue

            op = parts[0].upper()

            # Parse operands
            operands = {}
            if len(parts) > 1:
                ops_str = ' '.join(parts[1:])
                ops_list = [o.strip() for o in ops_str.split(',')]

                # Simple operand parsing
                if len(ops_list) == 1:
                    operands['dst'] = self._parse_operand(ops_list[0])
                elif len(ops_list) == 2:
                    operands['dst'] = self._parse_operand(ops_list[0])
                    operands['src'] = self._parse_operand(ops_list[1])

            ir_instr = IRInstruction(
                op=op,
                operands=operands,
                line_num=line_num,
                comment=comment
            )
            self.program.instructions.append(ir_instr)

        return self.program

    def _parse_operand(self, op_str: str) -> Union[str, int]:
        """Parse operand (register, immediate, or label)"""
        op_str = op_str.strip()

        # Immediate value (hex or decimal)
        if op_str.startswith('0x') or op_str.startswith('0X'):
            return int(op_str, 16)
        elif op_str.isdigit():
            return int(op_str)
        else:
            # Register or label
            return op_str


# ============================================================================
# IR Compiler
# ============================================================================

class IRCompiler:
    """Compile IR to pxOS primitives"""

    def __init__(self):
        self.program: Optional[IRProgram] = None
        self.primitives: List[str] = []
        self.address: int = 0x7C00
        self.label_addresses: Dict[str, int] = {}

    def compile(self, program: IRProgram) -> List[str]:
        """Main compilation pipeline"""
        self.program = program
        self.address = program.origin
        self.primitives = []
        self.label_addresses = {}

        self.emit_comment("=" * 60)
        self.emit_comment("PXI Assembly IR Compiled to pxOS Primitives")
        self.emit_comment("=" * 60)
        self.emit_comment("")

        # Pass 1: Calculate addresses for all labels
        self._calculate_addresses()

        # Pass 2: Generate primitives
        self._generate_primitives()

        # Add boot signature comment
        self.emit_comment("")
        self.emit_comment("=" * 60)
        self.emit_comment("Boot signature (added by build_pxos.py)")
        self.emit_comment("=" * 60)

        return self.primitives

    def _calculate_addresses(self):
        """Pass 1: Calculate addresses for labels and data"""
        addr = self.address

        # Process instructions to determine code size
        for instr in self.program.instructions:
            size = self._get_instruction_size(instr)
            addr += size

        # Data goes after code
        for data in self.program.data:
            label = data.label
            self.label_addresses[label] = addr
            data.address = addr

            if data.data_type == 'string':
                addr += len(data.value) + 1  # +1 for null terminator
            elif data.data_type == 'byte':
                addr += len(data.value) if isinstance(data.value, list) else 1
            elif data.data_type == 'word':
                addr += len(data.value) * 2 if isinstance(data.value, list) else 2

        # Calculate label addresses (second pass for jumps)
        addr = self.address
        for instr in self.program.instructions:
            # Check if there's a label at this position
            for label_name, label in self.program.labels.items():
                if label.address is None and label_name not in self.label_addresses:
                    # This is a code label - we need to track it
                    # For now, we'll handle this in a more sophisticated way
                    pass

            size = self._get_instruction_size(instr)
            addr += size

    def _get_instruction_size(self, instr: IRInstruction) -> int:
        """Calculate size of instruction in bytes"""
        op = instr.op

        # Simple size calculation
        if op in ['CLI', 'STI', 'RET', 'HLT', 'NOP', 'LODSB', 'STOSB', 'STOSW']:
            return 1
        elif op == 'MOV':
            dst = instr.operands.get('dst', '')
            src = instr.operands.get('src', '')
            if isinstance(dst, str) and len(dst) <= 2:  # Register
                if isinstance(src, int):
                    # MOV reg, imm
                    if dst in ['AX', 'BX', 'CX', 'DX', 'SI', 'DI', 'SP', 'BP']:
                        return 3  # 16-bit immediate
                    else:
                        return 2  # 8-bit immediate
                elif isinstance(src, str) and len(src) <= 2:
                    # MOV reg, reg
                    return 2
            return 3  # Conservative estimate
        elif op in ['INT', 'JMP', 'JZ', 'JNZ', 'JE', 'JNE']:
            return 2
        elif op == 'CALL':
            return 3
        elif op == 'REP':
            return 2  # REP + string op
        elif op == 'XOR' or op == 'OR':
            return 2
        else:
            return 3  # Conservative default

    def _generate_primitives(self):
        """Pass 2: Generate primitive commands"""
        addr = self.address

        # Generate code
        for instr in self.program.instructions:
            opcodes = self._compile_instruction(instr)

            comment = instr.comment or self._instr_to_comment(instr)

            for i, opcode in enumerate(opcodes):
                opcode_comment = comment if i == 0 else ""
                self.emit_write(addr, opcode, opcode_comment)
                addr += 1

        # Generate data
        self.emit_comment("")
        for data in self.program.data:
            self.emit_comment(f"Data: {data.label}")
            self.emit_define(data.label, data.address)

            if data.data_type == 'string':
                for i, ch in enumerate(data.value):
                    self.emit_write(data.address + i, ord(ch), f"'{ch}'")
                self.emit_write(data.address + len(data.value), 0, "Null terminator")
            elif data.data_type == 'byte':
                values = data.value if isinstance(data.value, list) else [data.value]
                for i, byte_val in enumerate(values):
                    self.emit_write(data.address + i, byte_val)

    def _compile_instruction(self, instr: IRInstruction) -> List[int]:
        """Compile single instruction to opcodes"""
        op = instr.op
        operands = instr.operands

        try:
            if op == 'MOV':
                return self._compile_mov(operands)
            elif op == 'INT':
                num = operands.get('dst', operands.get('num', 0))
                return X86.INT(num if isinstance(num, int) else int(num, 0))
            elif op == 'CALL':
                target = operands.get('dst', operands.get('target', ''))
                return self._compile_call(target)
            elif op == 'JMP':
                target = operands.get('dst', operands.get('target', ''))
                return self._compile_jmp(target)
            elif op in ['JZ', 'JE']:
                target = operands.get('dst', operands.get('target', ''))
                return X86.JZ_REL8(0)  # TODO: Calculate offset
            elif op in ['JNZ', 'JNE']:
                target = operands.get('dst', operands.get('target', ''))
                return X86.JNZ_REL8(0)  # TODO: Calculate offset
            elif op == 'RET':
                return X86.RET()
            elif op == 'CLI':
                return X86.CLI()
            elif op == 'STI':
                return X86.STI()
            elif op == 'HLT':
                return X86.HLT()
            elif op == 'NOP':
                return X86.NOP()
            elif op == 'LODSB':
                return X86.LODSB()
            elif op == 'STOSB':
                return X86.STOSB()
            elif op == 'STOSW':
                return X86.STOSW()
            elif op == 'REP':
                # REP followed by string op
                return X86.REP()
            elif op == 'XOR':
                dst = operands.get('dst', '')
                src = operands.get('src', '')
                return X86.XOR_REG_REG(dst, src)
            elif op == 'OR':
                dst = operands.get('dst', '')
                src = operands.get('src', '')
                return X86.OR_REG_REG(dst, src)
            else:
                raise ValueError(f"Unsupported instruction: {op}")

        except Exception as e:
            raise CompilationError(f"Error compiling {op}: {e}", instr.line_num)

    def _compile_mov(self, operands: Dict) -> List[int]:
        """Compile MOV instruction"""
        dst = operands.get('dst', '')
        src = operands.get('src', '')

        # MOV reg, imm
        if isinstance(src, int):
            if dst in ['AX', 'BX', 'CX', 'DX', 'SI', 'DI', 'SP', 'BP']:
                return X86.MOV_REG_IMM16(dst, src)
            else:
                return X86.MOV_REG8_IMM8(dst, src)

        # MOV reg, label (resolve label to address)
        elif isinstance(src, str) and src in self.label_addresses:
            label_addr = self.label_addresses[src]
            if dst in ['AX', 'BX', 'CX', 'DX', 'SI', 'DI', 'SP', 'BP']:
                return X86.MOV_REG_IMM16(dst, label_addr)
            else:
                raise ValueError(f"Cannot MOV 8-bit reg from label: {dst}, {src}")

        # MOV seg, reg
        elif dst in ['ES', 'DS', 'SS']:
            return X86.MOV_SEG_REG(dst, src)

        else:
            raise ValueError(f"Unsupported MOV {dst}, {src}")

    def _compile_call(self, target: str) -> List[int]:
        """Compile CALL instruction"""
        # TODO: Calculate relative offset to target label
        # For now, return placeholder
        return X86.CALL_REL16(0x0000)

    def _compile_jmp(self, target: str) -> List[int]:
        """Compile JMP instruction"""
        # Check if it's infinite loop (JMP $)
        if target == '$' or target == 'halt':
            return X86.JMP_REL8(0xFE)  # JMP -2 (self)

        # TODO: Calculate relative offset
        return X86.JMP_REL8(0x00)

    def _instr_to_comment(self, instr: IRInstruction) -> str:
        """Generate comment for instruction"""
        op_str = instr.op
        if instr.operands:
            ops = ', '.join(f"{k}={v}" for k, v in instr.operands.items())
            return f"{op_str} {ops}"
        return op_str

    def emit_comment(self, text: str):
        """Emit comment"""
        self.primitives.append(f"COMMENT {text}")

    def emit_write(self, addr: int, value: int, comment: str = ""):
        """Emit WRITE command"""
        comment_part = f"    COMMENT {comment}" if comment else ""
        self.primitives.append(f"WRITE 0x{addr:04X} 0x{value:02X}{comment_part}")

    def emit_define(self, label: str, addr: int, comment: str = ""):
        """Emit DEFINE command"""
        comment_part = f"    COMMENT {comment}" if comment else ""
        self.primitives.append(f"DEFINE {label} 0x{addr:04X}{comment_part}")


class CompilationError(Exception):
    """IR compilation error"""
    def __init__(self, message: str, line_num: int = 0):
        super().__init__(f"Line {line_num}: {message}")
        self.line_num = line_num


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PXI Assembly IR Compiler - Compile IR to pxOS primitives",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("input", type=Path, help="Input IR file (.pxi or .json)")
    parser.add_argument("-o", "--output", type=Path, help="Output primitives file")
    parser.add_argument("--format", choices=['text', 'json'], default='json',
                       help="Input format (default: json)")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        return 1

    try:
        # Parse IR
        content = args.input.read_text()
        ir_parser = IRParser()

        if args.format == 'json':
            ir_data = json.loads(content)
            program = ir_parser.parse_json(ir_data)
        else:
            program = ir_parser.parse_text(content)

        # Compile to primitives
        compiler = IRCompiler()
        primitives = compiler.compile(program)

        # Output
        output_text = '\n'.join(primitives)

        if args.output:
            args.output.write_text(output_text)
            print(f"Compiled: {args.input} -> {args.output}")
        else:
            print(output_text)

        return 0

    except CompilationError as e:
        print(f"Compilation error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
