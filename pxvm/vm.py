# pxvm/vm.py
# Multi-Kernel Virtual Machine — Phase 5 Foundation
# Real, runnable, testable. Starting from your actual repo.

from typing import List, Dict, Optional
import numpy as np
from scipy.ndimage import uniform_filter

class Kernel:
    def __init__(self, pid: int, code: bytes, color: int):
        self.pid = pid
        self.color = color
        self.pc = 0
        self.regs = [0] * 8
        self.memory = bytearray(64 * 1024)  # 64KB per kernel
        self.memory[:len(code)] = code
        self.halted = False
        self.cycles = 0

    def read_mem(self, addr: int, size: int = 1) -> int:
        if addr + size > len(self.memory):
            return 0
        val = 0
        for i in range(size):
            val |= self.memory[addr + i] << (8 * i)
        return val

    def write_mem(self, addr: int, val: int, size: int = 1):
        for i in range(size):
            if addr + i < len(self.memory):
                self.memory[addr + i] = val & 0xFF
            val >>= 8

class PxVM:
    def __init__(self, width=1024, height=1024):
        self.width = width
        self.height = height
        self.framebuffer = np.zeros((height, width, 3), dtype=np.uint8)
        self.pheromone = np.zeros((height, width), dtype=np.uint8)
        self.pheromone_decay = 1
        self.kernels: List[Kernel] = []
        self.next_pid = 1
        self.cycle = 0

    def spawn_kernel(self, code: bytes, color: int = 0x00FF00) -> int:
        kernel = Kernel(self.next_pid, code, color)
        self.kernels.append(kernel)
        pid = self.next_pid
        self.next_pid += 1
        return pid

    def step(self):
        if not self.kernels:
            return

        # Pheromone physics
        self.pheromone[self.pheromone > 0] -= self.pheromone_decay
        self.pheromone = uniform_filter(self.pheromone, size=3, mode='constant', cval=0)

        for kernel in self.kernels[:]:
            if kernel.halted:
                continue

            if kernel.pc >= len(kernel.memory):
                kernel.halted = True
                continue

            opcode = kernel.memory[kernel.pc]
            kernel.pc += 1
            kernel.cycles += 1

            if opcode == 0:   # HALT
                kernel.halted = True
            elif opcode == 1:  # MOV R, imm32
                reg = kernel.memory[kernel.pc]
                val_bytes = kernel.memory[kernel.pc + 1 : kernel.pc + 5]
                val = int.from_bytes(val_bytes, byteorder='little', signed=True)
                kernel.regs[reg] = val
                kernel.pc += 5
            elif opcode == 2:  # PLOT x=R0, y=R1, color=R2
                x = kernel.regs[0] % self.width
                y = kernel.regs[1] % self.height
                color = kernel.regs[2] & 0xFFFFFF
                r, g, b = (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF
                if 0 <= y < self.height and 0 <= x < self.width:
                    self.framebuffer[y, x] = [r, g, b]
            elif opcode == 3:  # ADD Rdst, Rsrc
                dst = kernel.memory[kernel.pc]
                src = kernel.memory[kernel.pc + 1]
                kernel.regs[dst] += kernel.regs[src]
                kernel.pc += 2
            elif opcode == 4: # ADDI Rdst, imm32
                dst_reg = kernel.memory[kernel.pc]
                val_bytes = kernel.memory[kernel.pc + 1 : kernel.pc + 5]
                val = int.from_bytes(val_bytes, byteorder='little', signed=True)
                kernel.regs[dst_reg] += val
                kernel.pc += 5
            elif opcode == 100: # SYS_EMIT_PHEROMONE
                x, y, strength = kernel.regs[0], kernel.regs[1], kernel.regs[2]
                if 0 <= y < self.height and 0 <= x < self.width:
                    self.pheromone[y, x] = max(self.pheromone[y, x], strength & 0xFF)
            elif opcode == 101: # SYS_SENSE_PHEROMONE
                x, y = kernel.regs[0], kernel.regs[1]
                if 0 <= y < self.height and 0 <= x < self.width:
                    kernel.regs[0] = self.pheromone[y, x]
                else:
                    kernel.regs[0] = 0
            elif opcode == 255:  # NOP
                pass
            else:
                pass # Imperfect mode
        self.cycle += 1
