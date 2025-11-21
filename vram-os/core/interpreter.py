#!/usr/bin/env python3
"""
PXL-ISA Interpreter - Executes pixel-encoded programs
This is a CPU-based simulator for testing before GPU implementation
"""

from typing import Dict, List, Optional
from .vram_state import VRAMState
from .pxl_isa import Instruction, Opcode, OperandType, Register


class CPUState:
    """Virtual CPU state for executing PXL-ISA programs"""

    def __init__(self):
        # 16 general-purpose registers + special registers
        self.registers: Dict[Register, int] = {
            reg: 0 for reg in Register
        }

        # Program counter (X, Y coordinates)
        self.pc_x = 0
        self.pc_y = 0

        # Flags
        self.zero_flag = False
        self.carry_flag = False
        self.halted = False

        # Execution stats
        self.instructions_executed = 0
        self.max_instructions = 10000  # Safety limit

    def get_register(self, reg: Register) -> int:
        """Read register value"""
        return self.registers.get(reg, 0)

    def set_register(self, reg: Register, value: int) -> None:
        """Write register value (with overflow handling)"""
        self.registers[reg] = value & 0xFFFFFFFF  # 32-bit values


class PXLInterpreter:
    """Interpreter for PXL-ISA programs stored in VRAMState"""

    def __init__(self, vram: VRAMState):
        self.vram = vram
        self.cpu = CPUState()
        self.debug = False
        self.trace: List[str] = []

    def fetch(self) -> Instruction:
        """Fetch instruction from current PC location"""
        r, g, b, a = self.vram.read_pixel(self.cpu.pc_x, self.cpu.pc_y)
        return Instruction.from_pixel(r, g, b, a)

    def increment_pc(self) -> None:
        """Move PC to next instruction pixel (sequential)"""
        self.cpu.pc_x += 1
        if self.cpu.pc_x >= self.vram.width:
            self.cpu.pc_x = 0
            self.cpu.pc_y += 1

    def execute(self, instruction: Instruction) -> None:
        """Execute a single PXL-ISA instruction"""
        opcode = instruction.opcode
        op_type = instruction.operand_type
        op1 = instruction.operand1
        op2 = instruction.operand2

        if self.debug:
            from .pxl_isa import PXLDisassembler
            disasm = PXLDisassembler.disassemble(instruction)
            self.trace.append(f"[{self.cpu.pc_x:04d},{self.cpu.pc_y:04d}] {disasm}")

        # Execute based on opcode
        if opcode == Opcode.NOP:
            pass  # No operation

        elif opcode == Opcode.HALT:
            self.cpu.halted = True

        elif opcode == Opcode.LOAD:
            if op_type == OperandType.IMMEDIATE:
                # LOAD Rdest, #value
                reg = Register(op1)
                value = op2
                self.cpu.set_register(reg, value)
            elif op_type == OperandType.MEMORY:
                # LOAD Rdest, [addr] - Load from VRAM pixel
                reg = Register(op1)
                # For simplicity, op2 is Y coordinate, X is 0
                r, g, b, a = self.vram.read_pixel(0, op2)
                # Interpret pixel as 32-bit value
                value = (r << 24) | (g << 16) | (b << 8) | a
                self.cpu.set_register(reg, value)

        elif opcode == Opcode.STORE:
            if op_type == OperandType.MEMORY:
                # STORE Rsrc, [addr] - Store to VRAM pixel
                reg = Register(op1)
                value = self.cpu.get_register(reg)
                # Break 32-bit value into RGBA
                r = (value >> 24) & 0xFF
                g = (value >> 16) & 0xFF
                b = (value >> 8) & 0xFF
                a = value & 0xFF
                self.vram.write_pixel(0, op2, r, g, b, a)

        elif opcode == Opcode.MOV:
            # MOV Rdest, Rsrc
            dest_reg = Register(op1)
            src_reg = Register(op2)
            value = self.cpu.get_register(src_reg)
            self.cpu.set_register(dest_reg, value)

        elif opcode == Opcode.ADD:
            if op_type == OperandType.REGISTER:
                # ADD Rdest, Rsrc
                dest_reg = Register(op1)
                src_reg = Register(op2)
                result = self.cpu.get_register(dest_reg) + self.cpu.get_register(src_reg)
                self.cpu.set_register(dest_reg, result)
                self.cpu.zero_flag = (result == 0)
            elif op_type == OperandType.IMMEDIATE:
                # ADD Rdest, #value
                dest_reg = Register(op1)
                result = self.cpu.get_register(dest_reg) + op2
                self.cpu.set_register(dest_reg, result)

        elif opcode == Opcode.SUB:
            if op_type == OperandType.REGISTER:
                dest_reg = Register(op1)
                src_reg = Register(op2)
                result = self.cpu.get_register(dest_reg) - self.cpu.get_register(src_reg)
                self.cpu.set_register(dest_reg, result)
                self.cpu.zero_flag = (result == 0)

        elif opcode == Opcode.INC:
            reg = Register(op1)
            result = self.cpu.get_register(reg) + 1
            self.cpu.set_register(reg, result)

        elif opcode == Opcode.DEC:
            reg = Register(op1)
            result = self.cpu.get_register(reg) - 1
            self.cpu.set_register(reg, result)

        elif opcode == Opcode.JMP:
            if op_type == OperandType.RELATIVE:
                # Relative jump: offset in Y direction
                self.cpu.pc_y += op2
                return  # Don't increment PC again

        elif opcode == Opcode.JMP_IF:
            # Conditional jump (if zero flag set)
            if self.cpu.zero_flag:
                self.cpu.pc_y += op2
                return

        elif opcode == Opcode.PIXEL_WRITE:
            # PIXEL_WRITE (X, Y) - Write R0 value to pixel
            x, y = op1, op2
            value = self.cpu.get_register(Register.R0)
            r = (value >> 24) & 0xFF
            g = (value >> 16) & 0xFF
            b = (value >> 8) & 0xFF
            a = value & 0xFF
            self.vram.write_pixel(x, y, r, g, b, a)

        elif opcode == Opcode.PIXEL_READ:
            # PIXEL_READ (X, Y) - Read pixel into R0
            x, y = op1, op2
            r, g, b, a = self.vram.read_pixel(x, y)
            value = (r << 24) | (g << 16) | (b << 8) | a
            self.cpu.set_register(Register.R0, value)

        elif opcode == Opcode.SYSCALL:
            # System call - op1 is syscall number
            self.handle_syscall(op1, op2)

        else:
            raise NotImplementedError(f"Opcode {opcode.name} not implemented")

        # Increment instruction counter
        self.cpu.instructions_executed += 1

    def handle_syscall(self, syscall_id: int, arg: int) -> None:
        """Handle system calls"""
        if syscall_id == 0:  # SYSCALL 0: Print R0 as number
            value = self.cpu.get_register(Register.R0)
            print(f"[SYSCALL] Print: {value}")
        elif syscall_id == 1:  # SYSCALL 1: Print character from R0
            value = self.cpu.get_register(Register.R0)
            print(f"[SYSCALL] Print char: {chr(value & 0xFF)}", end='')
        else:
            print(f"[SYSCALL] Unknown syscall {syscall_id}")

    def run(self, start_x: int = 0, start_y: int = 0, debug: bool = False) -> None:
        """Run program starting at (start_x, start_y)"""
        self.cpu.pc_x = start_x
        self.cpu.pc_y = start_y
        self.debug = debug
        self.trace = []

        print(f"Starting execution at ({start_x}, {start_y})...")

        while not self.cpu.halted and self.cpu.instructions_executed < self.cpu.max_instructions:
            # Fetch instruction
            instruction = self.fetch()

            # Execute instruction
            self.execute(instruction)

            # Move to next instruction (if not a jump)
            if instruction.opcode not in [Opcode.JMP, Opcode.JMP_IF, Opcode.HALT]:
                self.increment_pc()

        if self.cpu.halted:
            print(f"\nProgram halted after {self.cpu.instructions_executed} instructions.")
        else:
            print(f"\nExecution limit reached ({self.cpu.max_instructions} instructions).")

        if debug:
            print("\nExecution trace:")
            for line in self.trace[-20:]:  # Show last 20 instructions
                print(line)

        print(f"\nFinal register state:")
        for reg in [Register.R0, Register.R1, Register.R2, Register.R3]:
            print(f"  {reg.name} = {self.cpu.get_register(reg)}")


if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from core.vram_state import VRAMState
    from core.pxl_isa import PXLAssembler, Register

    print("PXL-ISA Interpreter Test\n")

    # Create VRAM state
    vram = VRAMState(1024, 1024)

    # Write a simple test program
    asm = PXLAssembler()
    program = [
        asm.assemble_load_imm(Register.R0, 10),   # R0 = 10
        asm.assemble_load_imm(Register.R1, 32),   # R1 = 32
        asm.assemble_add(Register.R0, Register.R1),  # R0 = R0 + R1
        asm.assemble_halt()
    ]

    # Write program to bootloader region (0,0)
    for i, instr in enumerate(program):
        r, g, b, a = instr.to_pixel()
        vram.write_pixel(i, 0, r, g, b, a)

    # Run interpreter
    interpreter = PXLInterpreter(vram)
    interpreter.run(start_x=0, start_y=0, debug=True)

    print(f"\nExpected R0 = 42, got R0 = {interpreter.cpu.get_register(Register.R0)}")
    print("Interpreter test complete! ✓")
