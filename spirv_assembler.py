#!/usr/bin/env python3
"""
spirv_assembler.py - Assembles a .spvasm text file into a .vram binary.
"""
import sys
import struct
import argparse
import numpy as np

VRAM_WIDTH, VRAM_HEIGHT = 800, 600

# A very simple mapping of opcode names to numbers.
# This is not a real SPIR-V assembler, just a tool for our v0.1 spec.
OPCODE_MAP = {
    "OpCapability": 17,
    "OpEntryPoint": 19,
    "OpTypeVoid": 43,
    "OpTypeFunction": 44,
    "OpTypePointer": 48,
    "OpConstant": 49,
    "OpFunction": 54,
    "OpFunctionEnd": 56,
    "OpLoad": 61,
    "OpStore": 62,
    "OpExtInstImport": 12,
    "OpExtInst": 13,
    "OpReturn": 253,
}

# Mapping for our custom syscalls
SYSCALL_MAP = {
    "SYS_PRINT": 1,
}

def pack_string(s):
    """Packs a string into 32-bit words, null-terminated."""
    s_bytes = s.encode('utf-8') + b'\x00'
    padding = -len(s_bytes) % 4
    s_bytes += b'\x00' * padding
    return [struct.unpack('<I', s_bytes[i:i+4])[0] for i in range(0, len(s_bytes), 4)]

def assemble_line(line):
    """Assembles a single line of .spvasm text into a list of 32-bit words."""
    line = line.split(';')[0].strip()
    parts = line.split()
    if not parts:
        return []

    op_name = parts[0]
    if op_name not in OPCODE_MAP:
        raise ValueError(f"Unknown opcode: {op_name}")

    opcode = OPCODE_MAP[op_name]
    words = []

    # Process operands
    operands = []
    is_sys_print = 'SYS_PRINT' in parts

    for i, part in enumerate(parts[1:]):
        if is_sys_print and part.startswith('"'):
            # For SYS_PRINT, the string follows the syscall ID
            string_literal = line.split('"')[1]
            operands.append(SYSCALL_MAP['SYS_PRINT'])
            operands.extend(pack_string(string_literal))
            break # String is always last
        elif part.startswith('"'):
            string_literal = line.split('"')[1]
            operands.extend(pack_string(string_literal))
            break
        elif part in SYSCALL_MAP:
            operands.append(SYSCALL_MAP[part])
        elif part.isdigit() or (part.startswith('-') and part[1:].isdigit()):
            operands.append(int(part))
        else: # Assume it's a result ID like %1
             operands.append(int(part.replace('%', '')))

    word_count = len(operands) + 1
    instruction_header = (word_count << 16) | opcode
    words.append(instruction_header)
    words.extend(operands)

    return words

def main():
    parser = argparse.ArgumentParser(description="pxOS SPIR-V Assembler v0.1")
    parser.add_argument('input_file', help="The input .spvasm file.")
    parser.add_argument('output_file', help="The output .vram file.")
    args = parser.parse_args()

    all_words = []
    with open(args.input_file, 'r') as f:
        for line in f:
            try:
                words = assemble_line(line)
                all_words.extend(words)
            except ValueError as e:
                print(f"Error assembling line: {line.strip()}\n{e}", file=sys.stderr)
                sys.exit(1)

    # Convert words to a numpy array of uint8 to simulate pixels
    spirv_bytes = b''.join(struct.pack('<I', word) for word in all_words)

    # Pad with zeros to fill the VRAM
    if len(spirv_bytes) > VRAM_WIDTH * VRAM_HEIGHT * 4:
        raise ValueError("Program is too large to fit in VRAM.")

    padding = VRAM_WIDTH * VRAM_HEIGHT * 4 - len(spirv_bytes)
    spirv_bytes += b'\x00' * padding

    vram = np.frombuffer(spirv_bytes, dtype=np.uint8).reshape((VRAM_HEIGHT, VRAM_WIDTH, 4))

    with open(args.output_file, 'wb') as f:
        f.write(vram.tobytes())

    print(f"Assembled {args.input_file} to {args.output_file} ({len(all_words)} words).")

if __name__ == "__main__":
    main()
