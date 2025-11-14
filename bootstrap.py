#!/usr/bin/env python3
"""
bootstrap.py - The pxOS SPIR-V Interpreter
This file loads a .vram file and interprets it as a SPIR-V module.
"""
import sys
import argparse
import numpy as np
import struct
from PIL import Image

VRAM_WIDTH, VRAM_HEIGHT = 800, 600

# Opcodes from our spec
OP_EXT_INST_IMPORT = 12
OP_EXT_INST = 13
OP_RETURN = 253
OP_FUNCTION_END = 56

# Our custom syscall IDs
SYS_PRINT = 1

class pxOS_SPIRV_CPU:
    def __init__(self, vram_file=None):
        self.vram_bytes = np.zeros(VRAM_WIDTH * VRAM_HEIGHT * 4, dtype=np.uint8)
        self.pc = 0  # Program Counter, now a word index into the VRAM bytes
        self.running = True
        self.ext_inst_sets = {}

        if vram_file:
            self.load_vram_from_file(vram_file)

    def load_vram_from_file(self, filename):
        try:
            with open(filename, 'rb') as f:
                self.vram_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            print(f"Loaded VRAM from {filename}")
        except Exception as e:
            print(f"Failed to load VRAM from {filename}: {e}")
            self.running = False

    def read_word(self):
        """Reads a 32-bit word from the current PC."""
        if self.pc * 4 + 4 > len(self.vram_bytes):
            return None
        word_bytes = self.vram_bytes[self.pc*4 : self.pc*4 + 4]
        word = struct.unpack('<I', word_bytes)[0]
        self.pc += 1
        return word

    def read_string(self, start_pc):
        """Reads a null-terminated string from VRAM."""
        byte_index = start_pc * 4
        s = ""
        while byte_index < len(self.vram_bytes):
            char_byte = self.vram_bytes[byte_index]
            if char_byte == 0:
                break
            s += chr(char_byte)
            byte_index += 1
        return s

    def fetch_decode_execute(self):
        if not self.running:
            return

        start_pc = self.pc
        header = self.read_word()
        if header is None:
            self.running = False
            return

        word_count = header >> 16
        opcode = header & 0xFFFF

        operands = [self.read_word() for _ in range(word_count - 1)]

        if opcode == OP_EXT_INST_IMPORT:
            result_id = operands[0]
            name_start_word = self.pc - word_count + 2
            name = self.read_string(name_start_word)
            if name == "PXOS.syscalls":
                self.ext_inst_sets[result_id] = name
                print(f"PC={start_pc}: Imported extended instruction set \"{name}\" as ID %{result_id}")

        elif opcode == OP_EXT_INST:
            set_id = operands[2]
            inst_id = operands[3]

            if self.ext_inst_sets.get(set_id) == "PXOS.syscalls":
                if inst_id == SYS_PRINT:
                    # The string for SYS_PRINT starts at the word *after* the syscall ID
                    str_start_word = start_pc + 5 # header + 4 operands before string
                    message = self.read_string(str_start_word)
                    print(f"SYSCALL: SYS_PRINT -> \"{message}\"")
                else:
                    print(f"SYSCALL: Unknown syscall ID {inst_id}")

        elif opcode == OP_RETURN or opcode == OP_FUNCTION_END:
            print(f"PC={start_pc}: OpReturn/OpFunctionEnd. Halting for now.")
            self.running = False

        # We ignore all other opcodes for now
        else:
            print(f"PC={start_pc}: Skipping Opcode {opcode}")

    def run(self, max_cycles=100):
        print("="*50)
        print("pxOS SPIR-V CPU Execution Start")
        print("="*50)

        cycle_count = 0
        while self.running and cycle_count < max_cycles:
            self.fetch_decode_execute()
            cycle_count += 1

        print("pxOS CPU Halted.")

def main():
    parser = argparse.ArgumentParser(description="pxOS SPIR-V Interpreter")
    parser.add_argument('vram_file', help="Path to a .vram file to execute.")
    args = parser.parse_args()

    cpu = pxOS_SPIRV_CPU(vram_file=args.vram_file)
    cpu.run()

if __name__ == "__main__":
    main()
