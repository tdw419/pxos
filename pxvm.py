#!/usr/bin/env python3
"""
pxVM - Minimal bytecode VM for pxOS
Supports syscalls to PXTERM for graphics output

IMPERFECT COMPUTING MODE:
- Unknown syscalls log warnings, don't crash
- Invalid syscall args use safe defaults
- VM execution errors are caught and logged
"""
from __future__ import annotations
from typing import List, Dict, Tuple
from dataclasses import dataclass


# Opcodes
OP_HALT = 0x00
OP_NOP = 0x01
OP_IMM8 = 0x10     # IMM8 reg, val (load 8-bit immediate)
OP_IMM32 = 0x11    # IMM32 reg, val (load 32-bit immediate)
OP_MOV = 0x20      # MOV dst, src (register to register)
OP_ADD = 0x30      # ADD dst, src1, src2
OP_SUB = 0x31      # SUB dst, src1, src2
OP_SYSCALL = 0xF0  # SYSCALL num

# Syscall numbers
SYS_PRINT_ID = 1
SYS_RECT_ID = 2
SYS_TEXT_ID = 3
SYS_LAYER_USE_ID = 4


@dataclass
class VMState:
    """VM execution state"""
    pc: int = 0              # Program counter
    registers: List[int] = None  # R0-R7
    halted: bool = False

    def __post_init__(self):
        if self.registers is None:
            self.registers = [0] * 8


class PxVM:
    """
    pxVM Bytecode Interpreter

    Features:
    - 8 general-purpose registers (R0-R7)
    - 32-bit integer operations
    - Syscalls that emit PXTERM instructions
    - Imperfect mode: never crashes on bad syscalls
    """

    def __init__(self, imperfect: bool = True):
        self.imperfect = imperfect
        self.state = VMState()
        self.memory = bytearray(65536)  # 64KB memory
        self.sysout: List[str] = []  # Collected PXTERM lines

        # Syscall lookup tables
        self.sys_messages = {
            1: "PXVM booting...",
            2: "PXVM ready.",
            3: "Task complete.",
            4: "Kernel init done.",
            5: "Process started.",
            6: "Process terminated.",
        }

        self.sys_colors = {
            1: (40, 40, 100, 255),   # window frame
            2: (20, 20, 60, 255),    # title bar
            3: (0, 0, 40, 255),      # background
            4: (255, 255, 255, 255), # white text
            5: (200, 200, 255, 255), # light blue text
        }

        self.sys_layers = {
            1: "background",
            2: "ui",
            3: "vm",
            4: "overlay",
        }

    def load(self, bytecode: bytes):
        """Load bytecode into memory"""
        self.memory[:len(bytecode)] = bytecode
        self.state.pc = 0
        self.state.halted = False

    def fetch_byte(self) -> int:
        """Fetch next byte from memory"""
        if self.state.pc >= len(self.memory):
            raise RuntimeError(f"PC out of bounds: {self.state.pc}")
        byte = self.memory[self.state.pc]
        self.state.pc += 1
        return byte

    def fetch_int32(self) -> int:
        """Fetch 32-bit integer (little-endian)"""
        b0 = self.fetch_byte()
        b1 = self.fetch_byte()
        b2 = self.fetch_byte()
        b3 = self.fetch_byte()
        # Interpret as signed 32-bit
        val = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        if val >= 0x80000000:
            val -= 0x100000000
        return val

    def handle_syscall(self, num: int):
        """Handle syscall - emits PXTERM instructions (imperfect mode: never crashes)"""
        r = self.state.registers

        try:
            if num == SYS_PRINT_ID:
                # SYS_PRINT_ID: R1 = message_id
                msg_id = r[1]
                msg = self.sys_messages.get(msg_id)
                if msg is None:
                    self.sysout.append(f"PRINT [vm warn] unknown message_id {msg_id}")
                else:
                    self.sysout.append(f"PRINT PXVM: {msg}")

            elif num == SYS_RECT_ID:
                # SYS_RECT_ID: R1=x, R2=y, R3=w, R4=h, R5=color_id
                x = r[1]
                y = r[2]
                w = r[3]
                h = r[4]
                color_id = r[5]
                rgba = self.sys_colors.get(color_id, (255, 0, 255, 255))
                if color_id not in self.sys_colors:
                    self.sysout.append(f"# WARNING: unknown color_id {color_id}, using fallback")
                r_, g_, b_, a_ = rgba
                self.sysout.append(f"RECT {x} {y} {w} {h} {r_} {g_} {b_} {a_}")

            elif num == SYS_TEXT_ID:
                # SYS_TEXT_ID: R1=x, R2=y, R3=color_id, R4=message_id
                x = r[1]
                y = r[2]
                color_id = r[3]
                msg_id = r[4]
                rgba = self.sys_colors.get(color_id, (255, 255, 255, 255))
                msg = self.sys_messages.get(msg_id, f"[vm warn] unknown message_id {msg_id}")
                r_, g_, b_, a_ = rgba
                self.sysout.append(f"TEXT {x} {y} {r_} {g_} {b_} {a_} {msg}")

            elif num == SYS_LAYER_USE_ID:
                # SYS_LAYER_USE_ID: R1=layer_id
                layer_id = r[1]
                name = self.sys_layers.get(layer_id)
                if name is None:
                    self.sysout.append(f"PRINT [vm warn] unknown layer_id {layer_id}")
                else:
                    self.sysout.append(f"SELECT {name}")

            else:
                # Unknown syscall - imperfect mode: log and continue
                self.sysout.append(f"# WARNING: unknown syscall {num} with args R1-R7={r[1:8]}")

        except Exception as e:
            # Imperfect mode: catch all syscall errors
            if self.imperfect:
                self.sysout.append(f"# ERROR: syscall {num} raised {type(e).__name__}: {e}")
            else:
                raise

    def step(self) -> bool:
        """Execute one instruction. Returns True if should continue."""
        if self.state.halted:
            return False

        try:
            opcode = self.fetch_byte()

            if opcode == OP_HALT:
                self.state.halted = True
                return False

            elif opcode == OP_NOP:
                pass

            elif opcode == OP_IMM8:
                # IMM8 reg, val
                reg = self.fetch_byte()
                val = self.fetch_byte()
                if 0 <= reg < 8:
                    self.state.registers[reg] = val

            elif opcode == OP_IMM32:
                # IMM32 reg, val
                reg = self.fetch_byte()
                val = self.fetch_int32()
                if 0 <= reg < 8:
                    self.state.registers[reg] = val

            elif opcode == OP_MOV:
                # MOV dst, src
                dst = self.fetch_byte()
                src = self.fetch_byte()
                if 0 <= dst < 8 and 0 <= src < 8:
                    self.state.registers[dst] = self.state.registers[src]

            elif opcode == OP_ADD:
                # ADD dst, src1, src2
                dst = self.fetch_byte()
                src1 = self.fetch_byte()
                src2 = self.fetch_byte()
                if 0 <= dst < 8 and 0 <= src1 < 8 and 0 <= src2 < 8:
                    result = self.state.registers[src1] + self.state.registers[src2]
                    # Wrap to 32-bit signed
                    if result >= 0x80000000:
                        result -= 0x100000000
                    elif result < -0x80000000:
                        result += 0x100000000
                    self.state.registers[dst] = result

            elif opcode == OP_SUB:
                # SUB dst, src1, src2
                dst = self.fetch_byte()
                src1 = self.fetch_byte()
                src2 = self.fetch_byte()
                if 0 <= dst < 8 and 0 <= src1 < 8 and 0 <= src2 < 8:
                    result = self.state.registers[src1] - self.state.registers[src2]
                    # Wrap to 32-bit signed
                    if result >= 0x80000000:
                        result -= 0x100000000
                    elif result < -0x80000000:
                        result += 0x100000000
                    self.state.registers[dst] = result

            elif opcode == OP_SYSCALL:
                # SYSCALL num
                num = self.fetch_byte()
                self.state.registers[0] = num  # Store in R0 for consistency
                self.handle_syscall(num)

            else:
                # Unknown opcode - imperfect mode: log and continue
                if self.imperfect:
                    self.sysout.append(f"# WARNING: unknown opcode 0x{opcode:02X} at PC={self.state.pc-1}")
                else:
                    raise RuntimeError(f"Unknown opcode: 0x{opcode:02X}")

            return True

        except Exception as e:
            # Imperfect mode: log VM execution errors
            if self.imperfect:
                self.sysout.append(f"# ERROR: VM execution failed at PC={self.state.pc}: {e}")
                self.state.halted = True
                return False
            else:
                raise

    def run(self, max_steps: int = 10000):
        """Run VM until HALT or max_steps"""
        steps = 0
        while steps < max_steps and self.step():
            steps += 1

        if steps >= max_steps and not self.state.halted:
            if self.imperfect:
                self.sysout.append(f"# WARNING: VM hit max_steps ({max_steps}), halting")
            else:
                raise RuntimeError(f"VM exceeded max_steps: {max_steps}")

    def get_sysout(self) -> List[str]:
        """Get collected PXTERM output"""
        return self.sysout

    def reset(self):
        """Reset VM state"""
        self.state = VMState()
        self.sysout = []


# Helper functions for assembling bytecode
def assemble_imm8(reg: int, val: int) -> bytes:
    """Assemble IMM8 instruction"""
    return bytes([OP_IMM8, reg & 0xFF, val & 0xFF])


def assemble_imm32(reg: int, val: int) -> bytes:
    """Assemble IMM32 instruction"""
    # Convert to signed 32-bit
    if val < 0:
        val = val + 0x100000000
    return bytes([
        OP_IMM32,
        reg & 0xFF,
        val & 0xFF,
        (val >> 8) & 0xFF,
        (val >> 16) & 0xFF,
        (val >> 24) & 0xFF,
    ])


def assemble_syscall(num: int) -> bytes:
    """Assemble SYSCALL instruction"""
    return bytes([OP_SYSCALL, num & 0xFF])


def assemble_halt() -> bytes:
    """Assemble HALT instruction"""
    return bytes([OP_HALT])
