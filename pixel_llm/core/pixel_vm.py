#!/usr/bin/env python3
"""
Pixel VM - Pixel-Native Bytecode Interpreter

A simple stack-based virtual machine that executes bytecode from pixel images.

THIS IS THE REAL EVOLUTION:
  Not Python source in pixels → executed by CPython
  But BYTECODE in pixels → executed by this VM

Architecture:
  - Stack-based (like Forth, JVM, Python bytecode)
  - Simple opcodes (1 byte opcode + operands)
  - Programs stored as .pxi pixel images
  - Python is just the interpreter runtime

Opcodes:
  0x01  PUSH <value>     - Push 32-bit value onto stack
  0x02  POP              - Pop value from stack
  0x03  ADD              - Pop two values, push sum
  0x04  SUB              - Pop two values, push difference
  0x05  MUL              - Pop two values, push product
  0x06  DIV              - Pop two values, push quotient
  0x10  PRINT            - Pop value and print it
  0x11  PRINT_STR <len>  - Print string of length <len>
  0x20  JMP <offset>     - Jump to offset
  0x21  JZ <offset>      - Jump if top of stack is zero
  0x30  LOAD <addr>      - Load value from memory
  0x31  STORE <addr>     - Store value to memory
  0xFF  HALT             - Stop execution

Philosophy:
"This is where pixels become executable.
 Not text pretending to be code.
 Not Python masquerading as pixels.
 Raw bytecode. Pure substrate logic."
"""

import sys
import struct
from pathlib import Path
from typing import List, Dict, Any

# Bootstrap
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pixel_llm.core.pixelfs import PixelFS


class PixelVM:
    """
    Simple stack-based VM for executing pixel bytecode programs.

    Memory model:
      - Stack: operand stack for computation
      - Memory: 1024 32-bit integers (4KB address space)
      - Program: bytecode loaded from pixels
      - PC: program counter

    This VM proves pixels can be EXECUTABLE, not just storage.
    """

    # Opcodes
    OP_PUSH = 0x01
    OP_POP = 0x02
    OP_ADD = 0x03
    OP_SUB = 0x04
    OP_MUL = 0x05
    OP_DIV = 0x06
    OP_PRINT = 0x10
    OP_PRINT_STR = 0x11
    OP_JMP = 0x20
    OP_JZ = 0x21
    OP_LOAD = 0x30
    OP_STORE = 0x31
    OP_HALT = 0xFF

    def __init__(self, debug: bool = False):
        self.stack: List[int] = []
        self.memory: List[int] = [0] * 1024  # 1024 words of memory
        self.program: bytes = b''
        self.pc: int = 0  # Program counter
        self.debug = debug
        self.halted = False

    def load_program(self, program_bytes: bytes):
        """Load bytecode program into VM"""
        self.program = program_bytes
        self.pc = 0
        self.halted = False
        if self.debug:
            print(f"[VM] Loaded {len(program_bytes)} bytes of bytecode")

    def load_from_pixel(self, pixel_path: str):
        """Load program from a pixel image"""
        fs = PixelFS()
        program_bytes = fs.read(pixel_path)
        self.load_program(program_bytes)
        if self.debug:
            print(f"[VM] Loaded from pixel: {pixel_path}")

    def read_byte(self) -> int:
        """Read one byte from program and advance PC"""
        if self.pc >= len(self.program):
            raise RuntimeError(f"PC out of bounds: {self.pc} >= {len(self.program)}")
        byte = self.program[self.pc]
        self.pc += 1
        return byte

    def read_int32(self) -> int:
        """Read 32-bit signed integer from program (little-endian)"""
        if self.pc + 4 > len(self.program):
            raise RuntimeError(f"Not enough bytes for int32 at PC={self.pc}")
        value = struct.unpack('<i', self.program[self.pc:self.pc+4])[0]
        self.pc += 4
        return value

    def read_uint16(self) -> int:
        """Read 16-bit unsigned integer from program"""
        if self.pc + 2 > len(self.program):
            raise RuntimeError(f"Not enough bytes for uint16 at PC={self.pc}")
        value = struct.unpack('<H', self.program[self.pc:self.pc+2])[0]
        self.pc += 2
        return value

    def execute_instruction(self):
        """Execute one instruction"""
        opcode = self.read_byte()

        if self.debug:
            print(f"[VM] PC={self.pc-1:04x} OP={opcode:02x} STACK={self.stack}")

        if opcode == self.OP_PUSH:
            value = self.read_int32()
            self.stack.append(value)

        elif opcode == self.OP_POP:
            if not self.stack:
                raise RuntimeError("Stack underflow on POP")
            self.stack.pop()

        elif opcode == self.OP_ADD:
            if len(self.stack) < 2:
                raise RuntimeError("Stack underflow on ADD")
            b = self.stack.pop()
            a = self.stack.pop()
            self.stack.append(a + b)

        elif opcode == self.OP_SUB:
            if len(self.stack) < 2:
                raise RuntimeError("Stack underflow on SUB")
            b = self.stack.pop()
            a = self.stack.pop()
            self.stack.append(a - b)

        elif opcode == self.OP_MUL:
            if len(self.stack) < 2:
                raise RuntimeError("Stack underflow on MUL")
            b = self.stack.pop()
            a = self.stack.pop()
            self.stack.append(a * b)

        elif opcode == self.OP_DIV:
            if len(self.stack) < 2:
                raise RuntimeError("Stack underflow on DIV")
            b = self.stack.pop()
            a = self.stack.pop()
            if b == 0:
                raise RuntimeError("Division by zero")
            self.stack.append(a // b)

        elif opcode == self.OP_PRINT:
            if not self.stack:
                raise RuntimeError("Stack underflow on PRINT")
            value = self.stack.pop()
            print(value)

        elif opcode == self.OP_PRINT_STR:
            length = self.read_uint16()
            if self.pc + length > len(self.program):
                raise RuntimeError(f"String out of bounds at PC={self.pc}")
            string_bytes = self.program[self.pc:self.pc+length]
            self.pc += length
            text = string_bytes.decode('utf-8')
            print(text)

        elif opcode == self.OP_JMP:
            offset = self.read_int32()
            self.pc = offset

        elif opcode == self.OP_JZ:
            offset = self.read_int32()
            if not self.stack:
                raise RuntimeError("Stack underflow on JZ")
            value = self.stack.pop()
            if value == 0:
                self.pc = offset

        elif opcode == self.OP_LOAD:
            addr = self.read_uint16()
            if addr >= len(self.memory):
                raise RuntimeError(f"Memory address out of bounds: {addr}")
            self.stack.append(self.memory[addr])

        elif opcode == self.OP_STORE:
            addr = self.read_uint16()
            if addr >= len(self.memory):
                raise RuntimeError(f"Memory address out of bounds: {addr}")
            if not self.stack:
                raise RuntimeError("Stack underflow on STORE")
            value = self.stack.pop()
            self.memory[addr] = value

        elif opcode == self.OP_HALT:
            self.halted = True

        else:
            raise RuntimeError(f"Unknown opcode: 0x{opcode:02x} at PC={self.pc-1}")

    def run(self, max_instructions: int = 10000):
        """Run the program until HALT or max instructions"""
        instruction_count = 0

        while not self.halted and instruction_count < max_instructions:
            try:
                self.execute_instruction()
                instruction_count += 1
            except Exception as e:
                print(f"❌ Runtime error at PC={self.pc}: {e}")
                print(f"   Stack: {self.stack}")
                raise

        if not self.halted:
            print(f"⚠️  Program did not halt after {max_instructions} instructions")

        if self.debug:
            print(f"[VM] Executed {instruction_count} instructions")
            print(f"[VM] Final stack: {self.stack}")

        return instruction_count


def assemble_program(asm_code: List[tuple]) -> bytes:
    """
    Assemble a program from tuples of (opcode, *args).

    Example:
        [
            (PixelVM.OP_PUSH, 42),
            (PixelVM.OP_PUSH, 8),
            (PixelVM.OP_ADD,),
            (PixelVM.OP_PRINT,),
            (PixelVM.OP_HALT,),
        ]
    """
    bytecode = bytearray()

    for instruction in asm_code:
        opcode = instruction[0]
        bytecode.append(opcode)

        if opcode == PixelVM.OP_PUSH:
            # PUSH requires int32 operand
            value = instruction[1]
            bytecode.extend(struct.pack('<i', value))

        elif opcode == PixelVM.OP_PRINT_STR:
            # PRINT_STR requires length (uint16) + string bytes
            string = instruction[1].encode('utf-8')
            length = len(string)
            bytecode.extend(struct.pack('<H', length))
            bytecode.extend(string)

        elif opcode in (PixelVM.OP_JMP, PixelVM.OP_JZ):
            # Jump instructions require int32 offset
            offset = instruction[1]
            bytecode.extend(struct.pack('<i', offset))

        elif opcode in (PixelVM.OP_LOAD, PixelVM.OP_STORE):
            # Memory instructions require uint16 address
            addr = instruction[1]
            bytecode.extend(struct.pack('<H', addr))

    return bytes(bytecode)


def main():
    """Main entry point for pixel VM"""
    if len(sys.argv) < 2:
        print("Pixel VM - Bytecode Interpreter")
        print()
        print("Usage:")
        print("  python3 pixel_llm/core/pixel_vm.py <program.pxi>")
        print("  python3 pixel_llm/core/pixel_vm.py --demo")
        print()
        sys.exit(1)

    if sys.argv[1] == '--demo':
        # Run a demo program
        print("=" * 70)
        print("PIXEL VM DEMO: Computing 42 + 8")
        print("=" * 70)
        print()

        # Assemble a simple program
        program = assemble_program([
            (PixelVM.OP_PUSH, 42),
            (PixelVM.OP_PUSH, 8),
            (PixelVM.OP_ADD,),
            (PixelVM.OP_PRINT,),
            (PixelVM.OP_HALT,),
        ])

        print(f"Program bytecode: {program.hex()}")
        print(f"Program size: {len(program)} bytes")
        print()
        print("Executing...")
        print()

        vm = PixelVM(debug=True)
        vm.load_program(program)
        vm.run()

        print()
        print("=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)

    else:
        # Load and run a pixel program
        program_path = sys.argv[1]
        print(f"Loading pixel program: {program_path}")

        vm = PixelVM(debug=True)
        vm.load_from_pixel(program_path)
        vm.run()


if __name__ == "__main__":
    main()
