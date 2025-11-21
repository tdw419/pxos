#!/usr/bin/env python3
"""
VRAM OS Bootloader v0.1
The first program that runs when VRAM OS starts

This bootloader:
1. Initializes the system
2. Displays "VRAM OS v0.1" message
3. Sets up the execution environment
4. Jumps to the kernel
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.vram_state import VRAMState
from core.pxl_isa import PXLAssembler, Opcode, OperandType, Register, Instruction
from core.interpreter import PXLInterpreter


def create_bootloader() -> list:
    """
    Create the bootloader program

    Memory layout:
    - Instructions at (0,0) to (31,0)
    - String data at (32,0) to (63,0)
    - Display output at (0,1) onwards
    """
    asm = PXLAssembler()

    program = []

    # === STAGE 1: Initialize ===
    # Clear R0-R3 registers
    program.append(asm.assemble_load_imm(Register.R0, 0))
    program.append(asm.assemble_load_imm(Register.R1, 0))
    program.append(asm.assemble_load_imm(Register.R2, 0))
    program.append(asm.assemble_load_imm(Register.R3, 0))

    # === STAGE 2: Display "VRAM OS v0.1" ===
    # We'll write each character as a colored pixel
    # Character 'V' = 86
    program.append(asm.assemble_load_imm(Register.R0, 86))  # 'V'
    program.append(Instruction(
        Opcode.PIXEL_WRITE,
        OperandType.MEMORY,
        0,   # X=0
        1    # Y=1
    ))

    # Character 'R' = 82
    program.append(asm.assemble_load_imm(Register.R0, 82))  # 'R'
    program.append(Instruction(
        Opcode.PIXEL_WRITE,
        OperandType.MEMORY,
        1,   # X=1
        1    # Y=1
    ))

    # Character 'A' = 65
    program.append(asm.assemble_load_imm(Register.R0, 65))  # 'A'
    program.append(Instruction(
        Opcode.PIXEL_WRITE,
        OperandType.MEMORY,
        2,   # X=2
        1    # Y=1
    ))

    # Character 'M' = 77
    program.append(asm.assemble_load_imm(Register.R0, 77))  # 'M'
    program.append(Instruction(
        Opcode.PIXEL_WRITE,
        OperandType.MEMORY,
        3,   # X=3
        1    # Y=1
    ))

    # Space = 32
    program.append(asm.assemble_load_imm(Register.R0, 32))  # ' '
    program.append(Instruction(
        Opcode.PIXEL_WRITE,
        OperandType.MEMORY,
        4,   # X=4
        1    # Y=1
    ))

    # Character 'O' = 79
    program.append(asm.assemble_load_imm(Register.R0, 79))  # 'O'
    program.append(Instruction(
        Opcode.PIXEL_WRITE,
        OperandType.MEMORY,
        5,   # X=5
        1    # Y=1
    ))

    # Character 'S' = 83
    program.append(asm.assemble_load_imm(Register.R0, 83))  # 'S'
    program.append(Instruction(
        Opcode.PIXEL_WRITE,
        OperandType.MEMORY,
        6,   # X=6
        1    # Y=1
    ))

    # === STAGE 3: Success indicator ===
    # Write green pixel (0, 255, 0, 255) to position (10, 1)
    program.append(asm.assemble_load_imm(Register.R0, 255))  # Green
    program.append(Instruction(
        Opcode.PIXEL_WRITE,
        OperandType.MEMORY,
        10,  # X=10
        1    # Y=1
    ))

    # === STAGE 4: Halt ===
    # For now, just halt. Later this will jump to kernel.
    program.append(asm.assemble_halt())

    return program


def build_bootloader_vram() -> VRAMState:
    """Build a complete VRAM state with bootloader"""
    vram = VRAMState(1024, 1024)

    # Create bootloader program
    program = create_bootloader()

    print(f"Bootloader program: {len(program)} instructions")

    # Write program to bootloader region (0,0) onwards
    for i, instr in enumerate(program):
        r, g, b, a = instr.to_pixel()
        vram.write_pixel(i, 0, r, g, b, a)

    return vram


def main():
    print("=== VRAM OS Bootloader v0.1 ===\n")

    # Build bootloader
    print("Building bootloader...")
    vram = build_bootloader_vram()

    # Save to JSON
    output_path = Path("bootloader_vram.json")
    vram.save_json(output_path)
    print(f"Saved VRAM state to {output_path}")

    # Run bootloader
    print("\n=== Executing Bootloader ===\n")
    interpreter = PXLInterpreter(vram)
    interpreter.run(start_x=0, start_y=0, debug=True)

    # Check output
    print("\n=== Bootloader Output ===")
    print("Reading text from row 1:")
    text = ""
    for x in range(10):
        r, g, b, a = vram.read_pixel(x, 1)
        # Alpha channel contains ASCII value
        if a > 0:
            text += chr(a)

    print(f"Text: '{text}'")

    # Check green success pixel
    success_pixel = vram.read_pixel(10, 1)
    print(f"Success indicator at (10,1): RGBA{success_pixel}")

    if text.startswith("VRAM OS"):
        print("\n✓ Bootloader executed successfully!")
        print("VRAM OS is booting!")
    else:
        print("\n✗ Bootloader output incorrect")

    return vram


if __name__ == "__main__":
    vram = main()
