#!/usr/bin/env python3
"""
PXL-ISA v0.1 - Pixel Instruction Set Architecture
Encoder/Decoder for VRAM OS instructions stored as RGBA pixels
"""

from enum import IntEnum
from typing import Tuple, List, Optional
from dataclasses import dataclass


class Opcode(IntEnum):
    """PXL-ISA Opcodes (stored in R channel)"""
    # Control Flow
    NOP = 0x00          # No operation
    JMP = 0x01          # Unconditional jump
    JMP_IF = 0x02       # Conditional jump
    CALL = 0x03         # Call subroutine
    RET = 0x04          # Return from subroutine

    # Data Movement
    LOAD = 0x10         # Load from memory
    STORE = 0x11        # Store to memory
    MOV = 0x12          # Move between registers

    # Arithmetic
    ADD = 0x20          # Addition
    SUB = 0x21          # Subtraction
    MUL = 0x22          # Multiplication
    DIV = 0x23          # Division
    INC = 0x24          # Increment
    DEC = 0x25          # Decrement

    # Logic
    AND = 0x30          # Bitwise AND
    OR = 0x31           # Bitwise OR
    XOR = 0x32          # Bitwise XOR
    NOT = 0x33          # Bitwise NOT

    # Comparison
    CMP = 0x40          # Compare
    TEST = 0x41         # Test (AND without storing)

    # Pixel Operations (VRAM-specific)
    PIXEL_READ = 0x50   # Read pixel at (X,Y)
    PIXEL_WRITE = 0x51  # Write pixel at (X,Y)
    PIXEL_COPY = 0x52   # Copy pixel region

    # System Calls
    SYSCALL = 0xF0      # System call
    HALT = 0xFF         # Halt execution


class OperandType(IntEnum):
    """Operand types (stored in G channel)"""
    IMMEDIATE = 0x00    # Immediate value
    REGISTER = 0x01     # Register reference
    MEMORY = 0x02       # Memory address
    RELATIVE = 0x03     # Relative offset


class Register(IntEnum):
    """Virtual registers"""
    R0 = 0x00
    R1 = 0x01
    R2 = 0x02
    R3 = 0x03
    R4 = 0x04
    R5 = 0x05
    R6 = 0x06
    R7 = 0x07
    R8 = 0x08
    R9 = 0x09
    R10 = 0x0A
    R11 = 0x0B
    R12 = 0x0C
    R13 = 0x0D
    R14 = 0x0E
    R15 = 0x0F
    PC = 0x10           # Program Counter
    SP = 0x11           # Stack Pointer
    FLAGS = 0x12        # Flags register


@dataclass
class Instruction:
    """Represents a PXL-ISA instruction"""
    opcode: Opcode
    operand_type: OperandType
    operand1: int       # B channel (0-255)
    operand2: int       # A channel (0-255)

    def to_pixel(self) -> Tuple[int, int, int, int]:
        """
        Encode instruction as RGBA pixel
        R = opcode
        G = operand type
        B = operand1
        A = operand2
        """
        return (
            int(self.opcode),
            int(self.operand_type),
            self.operand1 & 0xFF,
            self.operand2 & 0xFF
        )

    @classmethod
    def from_pixel(cls, r: int, g: int, b: int, a: int) -> 'Instruction':
        """Decode RGBA pixel into instruction"""
        return cls(
            opcode=Opcode(r),
            operand_type=OperandType(g),
            operand1=b,
            operand2=a
        )

    def __str__(self) -> str:
        """Human-readable instruction format"""
        return f"{self.opcode.name} {self.operand_type.name} {self.operand1:02X} {self.operand2:02X}"


class PXLAssembler:
    """Assembles text instructions into PXL-ISA pixel format"""

    @staticmethod
    def assemble_nop() -> Instruction:
        """NOP - No operation"""
        return Instruction(Opcode.NOP, OperandType.IMMEDIATE, 0, 0)

    @staticmethod
    def assemble_load_imm(dest_reg: Register, value: int) -> Instruction:
        """LOAD Rdest, #value - Load immediate value into register"""
        return Instruction(
            Opcode.LOAD,
            OperandType.IMMEDIATE,
            int(dest_reg),
            value & 0xFF
        )

    @staticmethod
    def assemble_mov(dest_reg: Register, src_reg: Register) -> Instruction:
        """MOV Rdest, Rsrc - Move value from src to dest register"""
        return Instruction(
            Opcode.MOV,
            OperandType.REGISTER,
            int(dest_reg),
            int(src_reg)
        )

    @staticmethod
    def assemble_add(dest_reg: Register, src_reg: Register) -> Instruction:
        """ADD Rdest, Rsrc - Add src to dest"""
        return Instruction(
            Opcode.ADD,
            OperandType.REGISTER,
            int(dest_reg),
            int(src_reg)
        )

    @staticmethod
    def assemble_jmp(offset_y: int) -> Instruction:
        """JMP +offset - Relative jump"""
        return Instruction(
            Opcode.JMP,
            OperandType.RELATIVE,
            0,  # X offset (0 for same column)
            offset_y & 0xFF
        )

    @staticmethod
    def assemble_pixel_write(x: int, y: int) -> Instruction:
        """PIXEL_WRITE X, Y - Write to pixel (X,Y)"""
        return Instruction(
            Opcode.PIXEL_WRITE,
            OperandType.MEMORY,
            x & 0xFF,
            y & 0xFF
        )

    @staticmethod
    def assemble_halt() -> Instruction:
        """HALT - Stop execution"""
        return Instruction(Opcode.HALT, OperandType.IMMEDIATE, 0, 0)


class PXLDisassembler:
    """Disassembles PXL-ISA pixels back into text instructions"""

    @staticmethod
    def disassemble(instruction: Instruction) -> str:
        """Convert instruction to assembly-like text"""
        opcode = instruction.opcode
        op_type = instruction.operand_type
        op1 = instruction.operand1
        op2 = instruction.operand2

        if opcode == Opcode.NOP:
            return "NOP"

        elif opcode == Opcode.LOAD:
            if op_type == OperandType.IMMEDIATE:
                return f"LOAD R{op1}, #{op2}"
            else:
                return f"LOAD R{op1}, [{op2}]"

        elif opcode == Opcode.MOV:
            return f"MOV R{op1}, R{op2}"

        elif opcode == Opcode.ADD:
            return f"ADD R{op1}, R{op2}"

        elif opcode == Opcode.JMP:
            if op_type == OperandType.RELATIVE:
                return f"JMP +{op2}"
            else:
                return f"JMP ({op1}, {op2})"

        elif opcode == Opcode.PIXEL_WRITE:
            return f"PIXEL_WRITE ({op1}, {op2})"

        elif opcode == Opcode.HALT:
            return "HALT"

        else:
            return f"{opcode.name} {op_type.name} {op1:02X} {op2:02X}"


if __name__ == "__main__":
    print("PXL-ISA v0.1 Encoder/Decoder Test\n")

    # Create some test instructions
    asm = PXLAssembler()

    instructions = [
        asm.assemble_load_imm(Register.R0, 42),
        asm.assemble_load_imm(Register.R1, 10),
        asm.assemble_add(Register.R0, Register.R1),
        asm.assemble_pixel_write(100, 100),
        asm.assemble_halt()
    ]

    print("Assembly:")
    disasm = PXLDisassembler()
    for i, instr in enumerate(instructions):
        pixel = instr.to_pixel()
        text = disasm.disassemble(instr)
        print(f"{i:03d}: {text:30s} | RGBA: {pixel}")

    print("\nDecoding test:")
    # Test round-trip encoding/decoding
    for instr in instructions:
        pixel = instr.to_pixel()
        decoded = Instruction.from_pixel(*pixel)
        assert decoded.opcode == instr.opcode
        assert decoded.operand_type == instr.operand_type
        print(f"✓ {disasm.disassemble(decoded)}")

    print("\nPXL-ISA encoder/decoder working! ✓")
