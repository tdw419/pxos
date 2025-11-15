#!/usr/bin/env python3
"""
pxos_assembler.py - A simple assembler for the pxVM.
"""

import struct
from pxvm import OP_HALT, OP_IMM32, OP_ADD, OP_CMP, OP_JMP, OP_JZ, OP_JNZ, OP_LOAD, OP_STORE, OP_SYSCALL

class PxOSAssembler:
    def assemble(self, pseudo_assembly):
        bytecode = bytearray()
        labels = {}
        fixups = []

        # First pass: collect labels
        offset = 0
        for line in pseudo_assembly.strip().split('\n'):
            line = line.split(';')[0].strip()
            if not line:
                continue

            if line.endswith(':'):
                labels[line[:-1]] = offset
                continue

            parts = line.split()
            op = parts[0]
            if op in ("IMM32", "LOAD", "STORE"):
                offset += 6
            elif op in ("ADD", "CMP"):
                offset += 3
            elif op in ("JMP", "JZ", "JNZ"):
                offset += 5
            elif op == "SYSCALL":
                offset += 2
            elif op == "HALT":
                offset += 1
            elif op == "PRINT":
                offset += 7
            elif op == "FORK":
                offset += 2

        # Second pass: assemble
        for line in pseudo_assembly.strip().split('\n'):
            line = line.split(';')[0].strip()
            if not line or line.endswith(':'):
                continue

            parts = line.split(None, 1)
            op = parts[0]
            args = [arg.strip() for arg in parts[1].split(',')] if len(parts) > 1 else []

            if op == "IMM32":
                reg = int(args[0][1:])
                val = int(args[1], 0)
                bytecode.extend([OP_IMM32, reg, *struct.pack("<I", val)])
            elif op == "ADD":
                reg1 = int(args[0][1:])
                reg2 = int(args[1][1:])
                bytecode.extend([OP_ADD, reg1, reg2])
            elif op == "CMP":
                reg1 = int(args[0][1:])
                reg2 = int(args[1][1:])
                bytecode.extend([OP_CMP, reg1, reg2])
            elif op in ("JMP", "JZ", "JNZ"):
                target = args[0]
                fixups.append((len(bytecode), target))
                bytecode.extend([{'JMP': OP_JMP, 'JZ': OP_JZ, 'JNZ': OP_JNZ}[op], 0, 0, 0, 0])
            elif op == "LOAD":
                reg = int(args[0][1:])
                addr = int(args[1][1:-1])
                bytecode.extend([OP_LOAD, reg, *struct.pack("<I", addr)])
            elif op == "STORE":
                addr = int(args[0][1:-1])
                reg = int(args[1][1:])
                bytecode.extend([OP_STORE, *struct.pack("<I", addr), reg])
            elif op == "SYSCALL":
                num = int(args[0])
                bytecode.extend([OP_SYSCALL, num])
            elif op == "HALT":
                bytecode.append(OP_HALT)
            elif op == "PRINT": # This is a pseudo-op for a syscall
                msg_id = int(args[0])
                bytecode.extend([OP_IMM32, 1, *struct.pack("<I", msg_id)])
                bytecode.extend([OP_SYSCALL, 1])
            elif op == "FORK":
                 bytecode.extend([OP_SYSCALL, 11])

        for addr, target in fixups:
            if target in labels:
                target_addr = labels[target]
                bytecode[addr+1:addr+5] = struct.pack("<I", target_addr)

        padding = (4 - len(bytecode) % 4) % 4
        bytecode.extend([OP_HALT] * padding)

        program_words = [word for word in struct.iter_unpack('<I', bytecode)]
        entry_point = labels.get("CLOCK", 0)
        return program_words, entry_point
