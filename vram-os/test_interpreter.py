#!/usr/bin/env python3
"""Test the PXL-ISA interpreter with a simple program"""

from core.vram_state import VRAMState
from core.pxl_isa import PXLAssembler, Register
from core.interpreter import PXLInterpreter

print("=== PXL-ISA Interpreter Test ===\n")

# Create VRAM state (1024x1024)
print("Creating VRAM state...")
vram = VRAMState(1024, 1024)

# Write a test program: Add two numbers and write result to pixel
asm = PXLAssembler()
program = [
    asm.assemble_load_imm(Register.R0, 10),      # R0 = 10
    asm.assemble_load_imm(Register.R1, 32),      # R1 = 32
    asm.assemble_add(Register.R0, Register.R1),  # R0 = R0 + R1 (42)
    asm.assemble_pixel_write(500, 500),          # Write R0 to pixel (500,500)
    asm.assemble_halt()
]

print(f"Writing {len(program)} instructions to bootloader region (0,0)...")
for i, instr in enumerate(program):
    r, g, b, a = instr.to_pixel()
    vram.write_pixel(i, 0, r, g, b, a)

# Create interpreter and run
print("\nStarting PXL-ISA interpreter...")
interpreter = PXLInterpreter(vram)
interpreter.run(start_x=0, start_y=0, debug=True)

# Verify result
print(f"\n=== Verification ===")
print(f"Expected: R0 = 42")
print(f"Actual:   R0 = {interpreter.cpu.get_register(Register.R0)}")

# Check if pixel was written
pixel = vram.read_pixel(500, 500)
print(f"\nPixel (500,500) RGBA: {pixel}")
print(f"Expected: (0, 0, 0, 42) [42 in Alpha channel]")

if interpreter.cpu.get_register(Register.R0) == 42:
    print("\n✓ Test PASSED! VRAM OS interpreter working correctly!")
else:
    print("\n✗ Test FAILED!")
