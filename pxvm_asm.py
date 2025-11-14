#!/usr/bin/env python3
"""
pxVM Assembler - Build bytecode programs programmatically

Example usage:
    from pxvm_asm import ProgramBuilder

    prog = ProgramBuilder()
    prog.imm32(1, 3)      # R1 = 3 (layer_id for vm)
    prog.syscall(4)       # SYS_LAYER_USE_ID
    prog.imm32(1, 150)    # R1 = x
    prog.imm32(2, 150)    # R2 = y
    prog.imm32(3, 300)    # R3 = w
    prog.imm32(4, 80)     # R4 = h
    prog.imm32(5, 1)      # R5 = color_id
    prog.syscall(2)       # SYS_RECT_ID
    prog.halt()

    bytecode = prog.build()
"""
from __future__ import annotations
from typing import List
from pxvm import (
    OP_HALT, OP_NOP, OP_IMM8, OP_IMM32, OP_MOV, OP_ADD, OP_SUB, OP_SYSCALL,
    SYS_PRINT_ID, SYS_RECT_ID, SYS_TEXT_ID, SYS_LAYER_USE_ID
)


class ProgramBuilder:
    """Helper for building pxVM programs"""

    def __init__(self):
        self.code: List[int] = []

    def emit_byte(self, byte: int):
        """Emit a single byte"""
        self.code.append(byte & 0xFF)

    def emit_int32(self, val: int):
        """Emit 32-bit integer (little-endian, signed)"""
        if val < 0:
            val = val + 0x100000000
        self.emit_byte(val & 0xFF)
        self.emit_byte((val >> 8) & 0xFF)
        self.emit_byte((val >> 16) & 0xFF)
        self.emit_byte((val >> 24) & 0xFF)

    def halt(self):
        """HALT - stop execution"""
        self.emit_byte(OP_HALT)
        return self

    def nop(self):
        """NOP - no operation"""
        self.emit_byte(OP_NOP)
        return self

    def imm8(self, reg: int, val: int):
        """IMM8 reg, val - load 8-bit immediate"""
        self.emit_byte(OP_IMM8)
        self.emit_byte(reg)
        self.emit_byte(val)
        return self

    def imm32(self, reg: int, val: int):
        """IMM32 reg, val - load 32-bit immediate"""
        self.emit_byte(OP_IMM32)
        self.emit_byte(reg)
        self.emit_int32(val)
        return self

    def mov(self, dst: int, src: int):
        """MOV dst, src - move register to register"""
        self.emit_byte(OP_MOV)
        self.emit_byte(dst)
        self.emit_byte(src)
        return self

    def add(self, dst: int, src1: int, src2: int):
        """ADD dst, src1, src2 - add two registers"""
        self.emit_byte(OP_ADD)
        self.emit_byte(dst)
        self.emit_byte(src1)
        self.emit_byte(src2)
        return self

    def sub(self, dst: int, src1: int, src2: int):
        """SUB dst, src1, src2 - subtract two registers"""
        self.emit_byte(OP_SUB)
        self.emit_byte(dst)
        self.emit_byte(src1)
        self.emit_byte(src2)
        return self

    def syscall(self, num: int):
        """SYSCALL num - invoke system call"""
        self.emit_byte(OP_SYSCALL)
        self.emit_byte(num)
        return self

    # Syscall helpers
    def sys_print(self, message_id: int):
        """SYS_PRINT_ID - print predefined message"""
        self.imm32(1, message_id)
        self.syscall(SYS_PRINT_ID)
        return self

    def sys_rect(self, x: int, y: int, w: int, h: int, color_id: int):
        """SYS_RECT_ID - draw rectangle"""
        self.imm32(1, x)
        self.imm32(2, y)
        self.imm32(3, w)
        self.imm32(4, h)
        self.imm32(5, color_id)
        self.syscall(SYS_RECT_ID)
        return self

    def sys_text(self, x: int, y: int, color_id: int, message_id: int):
        """SYS_TEXT_ID - draw text"""
        self.imm32(1, x)
        self.imm32(2, y)
        self.imm32(3, color_id)
        self.imm32(4, message_id)
        self.syscall(SYS_TEXT_ID)
        return self

    def sys_layer(self, layer_id: int):
        """SYS_LAYER_USE_ID - switch layer"""
        self.imm32(1, layer_id)
        self.syscall(SYS_LAYER_USE_ID)
        return self

    def build(self) -> bytes:
        """Build final bytecode"""
        return bytes(self.code)

    def save(self, filename: str):
        """Save bytecode to file"""
        with open(filename, 'wb') as f:
            f.write(self.build())
        print(f"Wrote {len(self.code)} bytes to {filename}")


# Example programs
def build_hello_program() -> bytes:
    """Simple hello world using VM syscalls"""
    prog = ProgramBuilder()

    # Print boot message
    prog.sys_print(1)  # "PXVM booting..."

    # Switch to VM layer
    prog.sys_layer(3)  # layer_id = 3 ("vm")

    # Draw a simple window frame
    prog.sys_rect(150, 150, 500, 300, 1)  # window background

    # Draw title bar
    prog.sys_rect(150, 150, 500, 40, 2)  # title bar

    # Draw title text
    prog.sys_text(170, 160, 4, 2)  # "PXVM ready."

    # Print ready message
    prog.sys_print(2)  # "PXVM ready."

    prog.halt()
    return prog.build()


def build_window_program() -> bytes:
    """More complex window with multiple elements"""
    prog = ProgramBuilder()

    # Console: boot message
    prog.sys_print(1)  # "PXVM booting..."

    # Background layer
    prog.sys_layer(1)  # background
    prog.sys_rect(0, 0, 800, 600, 3)  # dark blue background

    # VM layer
    prog.sys_layer(3)  # vm layer

    # Main window
    prog.sys_rect(100, 80, 600, 400, 1)  # window frame

    # Title bar
    prog.sys_rect(100, 80, 600, 40, 2)  # title bar

    # Title text
    prog.sys_text(120, 90, 4, 2)  # "PXVM ready."

    # Button 1
    prog.sys_rect(150, 200, 200, 50, 1)  # button background
    prog.sys_text(170, 213, 5, 5)  # "Process started."

    # Button 2
    prog.sys_rect(150, 270, 200, 50, 1)  # button background
    prog.sys_text(170, 283, 5, 6)  # "Process terminated."

    # Console: ready message
    prog.sys_print(2)  # "PXVM ready."
    prog.sys_print(4)  # "Kernel init done."

    prog.halt()
    return prog.build()


def main():
    """Generate example programs"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pxvm_asm.py <output.pxvm> [program_type]")
        print("\nProgram types:")
        print("  hello  - Simple hello world (default)")
        print("  window - Complex window with buttons")
        print("\nExample:")
        print("  python pxvm_asm.py examples/hello.pxvm hello")
        sys.exit(1)

    output_file = sys.argv[1]
    program_type = sys.argv[2] if len(sys.argv) > 2 else "hello"

    if program_type == "hello":
        bytecode = build_hello_program()
    elif program_type == "window":
        bytecode = build_window_program()
    else:
        print(f"Unknown program type: {program_type}")
        sys.exit(1)

    with open(output_file, 'wb') as f:
        f.write(bytecode)

    print(f"Generated {len(bytecode)} bytes → {output_file}")
    print(f"Run with: python pxvm_run.py {output_file}")


if __name__ == "__main__":
    main()
