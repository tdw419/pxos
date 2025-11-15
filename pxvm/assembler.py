# pxvm/assembler.py
# Real assembler for the multi-kernel VM
# Supports labels, comments, and clean syntax

from typing import Dict, List
import re

OPCODES = {
    'HALT': 0,
    'MOV':  1,
    'PLOT': 2,
    'SYS_PLOT': 2,
    'ADD':  3,
    'ADDI': 4,
    'SYS_EMIT_PHEROMONE': 100,
    'SYS_SENSE_PHEROMONE': 101,
    'SYS_WRITE_GLYPH': 102,
    'SYS_READ_GLYPH': 103,
    'SYS_SPAWN': 104,
    'NOP':  255,
}

REGS = {f'R{i}': i for i in range(8)}

class Assembler:
    def __init__(self):
        self.labels: Dict[str, int] = {}
        self.code: List[int] = []
        self.pc = 0

    def assemble(self, source: str) -> bytes:
        processed_lines = []
        for line in source.split('\n'):
            # Strip comments and whitespace
            line = line.split(';')[0]
            line = line.split('#')[0]
            line = line.strip()
            if line:
                processed_lines.append(line)
        lines = processed_lines

        self.pc = 0
        self.labels.clear()
        self.code.clear()

        # Pass 1: find labels
        for line in lines:
            if ':' in line:
                label = line.split(':')[0].strip()
                self.labels[label] = self.pc
            else:
                self.pc += self._estimate_size(line)

        # Pass 2: assemble
        self.pc = 0
        for line in lines:
            if ':' in line:
                instr = line.split(':', 1)[1].strip()
            else:
                instr = line

            if instr:
                self._emit(instr)

        return bytes(self.code)

    def _estimate_size(self, instr: str) -> int:
        parts = instr.replace(',', ' ').split()
        op = parts[0].upper() if parts else ''
        if op == 'MOV':
            if len(parts) > 2 and parts[2].upper() in REGS:
                return 6 + 3 # Emulated MOV R,R
            else:
                return 6 # MOV R, imm
        elif op == 'ADD':
            if len(parts) > 2 and parts[2].upper() not in REGS:
                return 6 # This is an ADDI R, imm
            else:
                return 3 # ADD R, R
        else: # HALT, PLOT, NOP, SYSCALLS
            return 1

    def _emit(self, instr: str):
        # A more robust way to parse: remove commas, then split by whitespace
        parts = instr.replace(',', ' ').split()
        op = parts[0].upper()
        if op not in OPCODES:
            # Handle JMP, JZ etc. later
            if op.startswith('J') or op == 'CMP':
                self.code.append(OPCODES['NOP'])
                return
            raise ValueError(f"Unknown opcode: {op}")

        if op == 'MOV':
            self.code.append(OPCODES['MOV'])
            dst_reg_str = parts[1].upper()
            src_str = parts[2]
            dst_reg = REGS[dst_reg_str]

            if src_str.upper() in REGS:
                # Emulate MOV R, R with MOV R, 0 and ADD R, R
                src_reg = REGS[src_str.upper()]
                self.code.append(dst_reg)
                self.code.extend((0).to_bytes(4, 'little', signed=True))
                self.code.append(OPCODES['ADD'])
                self.code.extend([dst_reg, src_reg])
            else:
                val = int(src_str, 0) if src_str.startswith('0x') else int(src_str)
                self.code.append(dst_reg)
                self.code.extend(val.to_bytes(4, 'little', signed=True))

        elif op == 'ADD':
            dst_reg = REGS[parts[1].upper()]
            src_str = parts[2]
            if src_str.upper() in REGS:
                self.code.append(OPCODES['ADD'])
                self.code.extend([dst_reg, REGS[src_str.upper()]])
            else:
                self.code.append(OPCODES['ADDI'])
                val = int(src_str, 0) if src_str.startswith('0x') else int(src_str)
                self.code.append(dst_reg)
                self.code.extend(val.to_bytes(4, 'little', signed=True))
        else:
            self.code.append(OPCODES[op])

        self.pc += len(self.code) - self.pc
