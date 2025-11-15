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

    # Syscalls - Chemical Communication (Phase 5.1)
    OP_SYS_EMIT_PHEROMONE = 100   # R0=x, R1=y, R2=strength (0-255)
    OP_SYS_SENSE_PHEROMONE = 101  # R0=x, R1=y -> R0=strength

    # Syscalls - Symbolic Communication (Phase 6)
    OP_SYS_WRITE_GLYPH = 102  # R0=x, R1=y, R2=glyph_id (0-15)
    OP_SYS_READ_GLYPH = 103   # R0=x, R1=y -> R0=glyph_id

    # Syscalls - Reproduction (Phase 7)
    OP_SYS_SPAWN = 104  # R1=child_x, R2=child_y -> R0=child_pid

    def __init__(self, width: int = 1024, height: int = 1024):
        self.width = width
        self.height = height
        self.framebuffer = np.zeros((height, width, 3), dtype=np.uint8)

        # Phase 5.1: Pheromone layer (chemical communication)
        self.pheromone = np.zeros((height, width), dtype=np.float32)
        self.pheromone_decay = 0.95  # Multiply by this each cycle
        self.pheromone_diffusion = 0.1  # Diffusion rate

        # Phase 6: Glyph layer (symbolic communication)
        self.glyphs = np.zeros((height, width), dtype=np.uint8)  # 0-15 glyph IDs

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

    def spawn_child(self, parent: Kernel, child_x: int, child_y: int) -> int:
        """
        Create a child kernel by cloning parent's memory.

        Args:
            parent: Parent kernel to clone from
            child_x: X position for child
            child_y: Y position for child

        Returns:
            Child PID (or 0 on failure)
        """
        MAX_KERNELS = 64
        if len(self.kernels) >= MAX_KERNELS:
            return 0  # Failure - too many kernels

        # Create child with full memory copy
        child = Kernel(self.next_pid, bytes(parent.memory), parent.color)

        # Reset execution state
        child.pc = 0
        child.halted = False
        child.cycles = 0
        child.zero_flag = False

        # Copy parent's registers, then set position
        child.regs = parent.regs.copy()
        child.regs[0] = child_x % self.width
        child.regs[1] = child_y % self.height

        # Add to scheduler
        self.kernels.append(child)
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

            # Pheromone syscalls
            elif opcode == self.OP_SYS_EMIT_PHEROMONE:
                x = kernel.regs[0] % self.width
                y = kernel.regs[1] % self.height
                strength = min(255, kernel.regs[2])  # Clamp to 0-255
                # Add to existing pheromone (saturate at 255)
                self.pheromone[y, x] = min(255, self.pheromone[y, x] + strength)

            elif opcode == self.OP_SYS_SENSE_PHEROMONE:
                x = kernel.regs[0] % self.width
                y = kernel.regs[1] % self.height
                # Return pheromone strength at location
                kernel.regs[0] = int(self.pheromone[y, x])

            # Glyph syscalls
            elif opcode == self.OP_SYS_WRITE_GLYPH:
                x = kernel.regs[0] % self.width
                y = kernel.regs[1] % self.height
                glyph_id = kernel.regs[2] & 0x0F  # Only 4 bits (0-15)
                self.glyphs[y, x] = glyph_id

            elif opcode == self.OP_SYS_READ_GLYPH:
                x = kernel.regs[0] % self.width
                y = kernel.regs[1] % self.height
                kernel.regs[0] = self.glyphs[y, x]

            # Reproduction syscalls
            elif opcode == self.OP_SYS_SPAWN:
                # R1 = child x position
                # R2 = child y position
                # Returns child PID in R0 (0 if failed)
                child_x = kernel.regs[1]
                child_y = kernel.regs[2]
                child_pid = self.spawn_child(kernel, child_x, child_y)
                kernel.regs[0] = child_pid

            else:
                # Unknown opcode - treat as NOP
                pass

        # Update pheromone field (decay and diffusion)
        self._update_pheromones()

        self.cycle += 1

    def run(self, max_cycles: int = 1000):
        """Run VM for specified number of cycles"""
        for _ in range(max_cycles):
            self.step()
            if all(k.halted for k in self.kernels):
                break

    def _update_pheromones(self):
        """Update pheromone field: decay and diffusion"""
        # Decay
        self.pheromone *= self.pheromone_decay

        # Simple diffusion: average with neighbors
        if self.pheromone_diffusion > 0:
            from scipy import ndimage
            # Use a 3x3 kernel for diffusion
            kernel = np.ones((3, 3)) / 9.0
            diffused = ndimage.convolve(self.pheromone, kernel, mode='constant')
            # Blend original with diffused
            self.pheromone = (1 - self.pheromone_diffusion) * self.pheromone + \
                           self.pheromone_diffusion * diffused

        # Clamp to valid range
        self.pheromone = np.clip(self.pheromone, 0, 255)

    def alive_count(self) -> int:
        """Return number of non-halted kernels"""
        return sum(1 for k in self.kernels if not k.halted)
