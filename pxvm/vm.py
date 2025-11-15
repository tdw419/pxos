"""
pxvm.vm - Multi-Kernel Virtual Machine Core

Implements a simple VM where multiple kernels (processes) run in parallel,
sharing a common framebuffer for visual communication.

Each kernel has:
- 64KB isolated memory
- 8 general-purpose registers (R0-R7)
- Program counter (PC)
- Zero flag for conditional jumps

All kernels share:
- 1024x1024 RGB framebuffer (visible to all)
"""

from typing import List, Optional
import numpy as np


class Kernel:
    """A single kernel (process) running in the VM"""

    def __init__(self, pid: int, code: bytes, color: int = 0x00FF00):
        self.pid = pid
        self.color = color  # Default color for this kernel
        self.pc = 0  # Program counter
        self.regs = [0] * 8  # R0-R7
        self.zero_flag = False
        self.halted = False
        self.cycles = 0

        # 64KB memory
        self.memory = bytearray(64 * 1024)
        if len(code) > len(self.memory):
            raise ValueError(f"Code too large: {len(code)} bytes (max 64KB)")
        self.memory[:len(code)] = code

    def read_byte(self, addr: int) -> int:
        """Read a single byte from memory"""
        if 0 <= addr < len(self.memory):
            return self.memory[addr]
        return 0

    def read_word(self, addr: int) -> int:
        """Read a 32-bit little-endian word"""
        val = 0
        for i in range(4):
            val |= self.read_byte(addr + i) << (8 * i)
        return val

    def write_byte(self, addr: int, val: int):
        """Write a single byte to memory"""
        if 0 <= addr < len(self.memory):
            self.memory[addr] = val & 0xFF

    def write_word(self, addr: int, val: int):
        """Write a 32-bit little-endian word"""
        for i in range(4):
            self.write_byte(addr + i, (val >> (8 * i)) & 0xFF)


class PxVM:
    """Multi-kernel virtual machine with shared framebuffer"""

    # Opcodes
    OP_HALT = 0    # Stop execution
    OP_MOV = 1     # MOV Rd, imm32 - Load immediate into register
    OP_PLOT = 2    # PLOT - Draw pixel at (R0, R1) with color R2
    OP_ADD = 3     # ADD Rd, Rs - Rd += Rs
    OP_SUB = 4     # SUB Rd, Rs - Rd -= Rs
    OP_JMP = 5     # JMP offset8 - Unconditional jump (signed byte offset)
    OP_JZ = 6      # JZ offset8 - Jump if zero flag set
    OP_CMP = 7     # CMP Rd, Rs - Set zero flag if Rd == Rs
    OP_NOP = 255   # No operation

    def __init__(self, width: int = 1024, height: int = 1024):
        self.width = width
        self.height = height
        self.framebuffer = np.zeros((height, width, 3), dtype=np.uint8)
        self.kernels: List[Kernel] = []
        self.next_pid = 1
        self.cycle = 0

    def spawn_kernel(self, code: bytes, color: int = 0x00FF00) -> int:
        """Create a new kernel and return its PID"""
        kernel = Kernel(self.next_pid, code, color)
        self.kernels.append(kernel)
        pid = self.next_pid
        self.next_pid += 1
        return pid

    def step(self):
        """Execute one instruction for each running kernel (round-robin)"""
        if not self.kernels:
            return

        for kernel in self.kernels[:]:  # Copy list in case kernel dies
            if kernel.halted:
                continue

            if kernel.pc >= len(kernel.memory):
                kernel.halted = True
                continue

            # Fetch opcode
            opcode = kernel.read_byte(kernel.pc)
            kernel.pc += 1
            kernel.cycles += 1

            # Execute
            if opcode == self.OP_HALT:
                kernel.halted = True

            elif opcode == self.OP_MOV:  # MOV Rd, imm32
                if kernel.pc + 4 >= len(kernel.memory):
                    kernel.halted = True
                    continue
                reg = kernel.read_byte(kernel.pc)
                kernel.pc += 1
                val = kernel.read_word(kernel.pc)
                kernel.pc += 4
                if 0 <= reg < 8:
                    kernel.regs[reg] = val & 0xFFFFFFFF

            elif opcode == self.OP_PLOT:  # PLOT x=R0, y=R1, color=R2
                x = kernel.regs[0] % self.width
                y = kernel.regs[1] % self.height
                color = kernel.regs[2] & 0xFFFFFF
                r = (color >> 16) & 0xFF
                g = (color >> 8) & 0xFF
                b = color & 0xFF
                self.framebuffer[y, x] = [r, g, b]

            elif opcode == self.OP_ADD:  # ADD Rd, Rs
                dst = kernel.read_byte(kernel.pc)
                src = kernel.read_byte(kernel.pc + 1)
                kernel.pc += 2
                if 0 <= dst < 8 and 0 <= src < 8:
                    kernel.regs[dst] = (kernel.regs[dst] + kernel.regs[src]) & 0xFFFFFFFF

            elif opcode == self.OP_SUB:  # SUB Rd, Rs
                dst = kernel.read_byte(kernel.pc)
                src = kernel.read_byte(kernel.pc + 1)
                kernel.pc += 2
                if 0 <= dst < 8 and 0 <= src < 8:
                    result = kernel.regs[dst] - kernel.regs[src]
                    kernel.regs[dst] = result & 0xFFFFFFFF
                    kernel.zero_flag = (result == 0)

            elif opcode == self.OP_CMP:  # CMP Rd, Rs
                dst = kernel.read_byte(kernel.pc)
                src = kernel.read_byte(kernel.pc + 1)
                kernel.pc += 2
                if 0 <= dst < 8 and 0 <= src < 8:
                    kernel.zero_flag = (kernel.regs[dst] == kernel.regs[src])

            elif opcode == self.OP_JMP:  # JMP offset
                offset = kernel.read_byte(kernel.pc)
                kernel.pc += 1
                # Interpret as signed byte
                if offset >= 128:
                    offset = offset - 256
                kernel.pc = (kernel.pc + offset) & 0xFFFF

            elif opcode == self.OP_JZ:  # JZ offset
                offset = kernel.read_byte(kernel.pc)
                kernel.pc += 1
                if kernel.zero_flag:
                    if offset >= 128:
                        offset = offset - 256
                    kernel.pc = (kernel.pc + offset) & 0xFFFF

            elif opcode == self.OP_NOP:
                pass  # Do nothing

            else:
                # Unknown opcode - treat as NOP
                pass

        self.cycle += 1

    def run(self, max_cycles: int = 1000):
        """Run VM for specified number of cycles"""
        for _ in range(max_cycles):
            self.step()
            if all(k.halted for k in self.kernels):
                break

    def alive_count(self) -> int:
        """Return number of non-halted kernels"""
        return sum(1 for k in self.kernels if not k.halted)
