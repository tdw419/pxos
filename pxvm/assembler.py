"""
pxvm.assembler - Simple assembler for kernel programs

Supports:
- Labels (for jumps)
- Comments (# or ;)
- Register names (R0-R7)
- Hex literals (0xFF) and decimal
- Basic instructions: MOV, PLOT, ADD, SUB, CMP, JMP, JZ, HALT, NOP
"""

import re
from typing import Dict, List, Tuple


class AssemblerError(Exception):
    """Raised when assembly fails"""
    pass


class Assembler:
    """Assembles human-readable assembly into bytecode"""

    OPCODES = {
        'HALT': 0,
        'MOV': 1,
        'PLOT': 2,
        'ADD': 3,
        'SUB': 4,
        'JMP': 5,
        'JZ': 6,
        'CMP': 7,
        'NOP': 255,
        # Phase 5.1: Pheromone syscalls
        'SYS_EMIT_PHEROMONE': 100,
        'SYS_SENSE_PHEROMONE': 101,
        # Phase 6: Glyph syscalls
        'SYS_WRITE_GLYPH': 102,
        'SYS_READ_GLYPH': 103,
        # Phase 7: Reproduction
        'SYS_SPAWN': 104,
        # Phase 8: Evolution
        'SYS_EAT': 105,
    }

    REGS = {f'R{i}': i for i in range(8)}

    def __init__(self):
        self.labels: Dict[str, int] = {}
        self.code: List[int] = []
        self.pc = 0

    def assemble(self, source: str) -> bytes:
        """Assemble source code into bytecode"""
        # Split into lines and remove comments/empty lines
        lines = []
        for line in source.split('\n'):
            # Remove comments
            line = re.sub(r'[#;].*$', '', line).strip()
            if line:
                lines.append(line)

        # Two-pass assembly
        # Pass 1: Find all labels and their addresses
        self.pc = 0
        self.labels.clear()
        for line in lines:
            if ':' in line:
                label_part, instr_part = line.split(':', 1)
                label = label_part.strip()
                self.labels[label] = self.pc
                line = instr_part.strip()
            if line:  # If there's an instruction after the label
                self.pc += self._estimate_size(line)

        # Pass 2: Generate bytecode
        self.code.clear()
        self.pc = 0
        for line in lines:
            if ':' in line:
                _, instr_part = line.split(':', 1)
                line = instr_part.strip()
            if line:
                self._emit_instruction(line)

        return bytes(self.code)

    def _estimate_size(self, line: str) -> int:
        """Estimate instruction size in bytes"""
        parts = self._parse_line(line)
        if not parts:
            return 0

        op = parts[0].upper()
        if op == 'MOV':
            return 6  # opcode(1) + reg(1) + imm32(4)
        elif op in ('ADD', 'SUB', 'CMP'):
            return 3  # opcode(1) + dst(1) + src(1)
        elif op in ('JMP', 'JZ'):
            return 2  # opcode(1) + offset(1)
        elif op in ('PLOT', 'HALT', 'NOP', 'SYS_EMIT_PHEROMONE', 'SYS_SENSE_PHEROMONE',
                    'SYS_WRITE_GLYPH', 'SYS_READ_GLYPH', 'SYS_SPAWN', 'SYS_EAT'):
            return 1  # opcode only (uses registers)
        else:
            raise AssemblerError(f"Unknown instruction: {op}")

    def _parse_line(self, line: str) -> List[str]:
        """Parse instruction line into parts"""
        # Split by whitespace and commas
        parts = re.split(r'[\s,]+', line)
        return [p for p in parts if p]

    def _emit_instruction(self, line: str):
        """Emit bytecode for one instruction"""
        parts = self._parse_line(line)
        if not parts:
            return

        op = parts[0].upper()
        old_pc = self.pc

        if op == 'HALT':
            self.code.append(self.OPCODES['HALT'])
            self.pc += 1

        elif op == 'NOP':
            self.code.append(self.OPCODES['NOP'])
            self.pc += 1

        elif op == 'PLOT':
            self.code.append(self.OPCODES['PLOT'])
            self.pc += 1

        elif op == 'MOV':
            if len(parts) < 3:
                raise AssemblerError(f"MOV requires 2 operands: {line}")
            dst = parts[1].upper()
            val = parts[2]

            if dst not in self.REGS:
                raise AssemblerError(f"Invalid register: {dst}")

            # Parse immediate value
            if val.startswith('0x') or val.startswith('0X'):
                imm = int(val, 16)
            else:
                imm = int(val)

            self.code.append(self.OPCODES['MOV'])
            self.code.append(self.REGS[dst])
            # Emit 32-bit little-endian immediate
            for i in range(4):
                self.code.append((imm >> (8 * i)) & 0xFF)
            self.pc += 6

        elif op in ('ADD', 'SUB', 'CMP'):
            if len(parts) < 3:
                raise AssemblerError(f"{op} requires 2 operands: {line}")
            dst = parts[1].upper()
            src = parts[2].upper()

            if dst not in self.REGS or src not in self.REGS:
                raise AssemblerError(f"Invalid registers: {dst}, {src}")

            self.code.append(self.OPCODES[op])
            self.code.append(self.REGS[dst])
            self.code.append(self.REGS[src])
            self.pc += 3

        elif op in ('JMP', 'JZ'):
            if len(parts) < 2:
                raise AssemblerError(f"{op} requires a label: {line}")
            label = parts[1]

            if label not in self.labels:
                raise AssemblerError(f"Undefined label: {label}")

            target = self.labels[label]
            # Calculate offset from next instruction
            next_pc = self.pc + 2
            offset = target - next_pc

            # Must fit in signed byte (-128 to 127)
            if offset < -128 or offset > 127:
                raise AssemblerError(f"Jump offset too large: {offset} (must be -128 to 127)")

            # Convert to unsigned byte representation
            if offset < 0:
                offset += 256

            self.code.append(self.OPCODES[op])
            self.code.append(offset)
            self.pc += 2

        elif op in ('SYS_EMIT_PHEROMONE', 'SYS_SENSE_PHEROMONE',
                    'SYS_WRITE_GLYPH', 'SYS_READ_GLYPH', 'SYS_SPAWN', 'SYS_EAT'):
            # Simple syscalls - just emit opcode (uses registers)
            self.code.append(self.OPCODES[op])
            self.pc += 1

        else:
            raise AssemblerError(f"Unknown instruction: {op}")


def assemble(source: str) -> bytes:
    """Convenience function to assemble source code"""
    asm = Assembler()
    return asm.assemble(source)
