"""
Pixel Process Control Block (PPCB)
Processes as visible sprites with RGB registers
"""

import numpy as np
from typing import Tuple, Dict
import random


class PixelProcess:
    """
    A Pixel Process - the fundamental unit of execution.

    A process is a visible rectangular region of memory with:
    - Row 0: Header (PID, status, IP, SP)
    - Row 1: RGB registers (R0, R1, R2)
    - Rows 2+: Stack and program memory

    Process State Color Codes:
    - Green (#00FF00): Running
    - Red (#FF0000): Terminated
    - Yellow (#FFFF00): Blocked/Waiting
    - Blue (#0000FF): Kernel Mode
    """

    # State color codes
    STATE_RUNNING = (0, 255, 0)
    STATE_TERMINATED = (255, 0, 0)
    STATE_BLOCKED = (255, 255, 0)
    STATE_KERNEL = (0, 0, 255)
    STATE_READY = (128, 128, 128)

    def __init__(self, pid: int, memory_manager):
        """
        Create a new pixel process.

        Args:
            pid: Process ID
            memory_manager: Reference to the PixelMemoryManager
        """
        self.pid = pid
        self.state = "READY"
        self.memory_manager = memory_manager

        # Allocate 16x16 pixel block for this process
        try:
            self.base_x, self.base_y = memory_manager.allocate_16x16_block(pid)
        except MemoryError:
            raise MemoryError(f"Cannot allocate memory for PID {pid}")

        # RGB Registers
        # These are visible in Row 1 of the process block
        self.registers: Dict[str, Tuple[int, int, int]] = {
            'R0': (0, 0, 0),  # Accumulator
            'R1': (0, 0, 0),  # Counter
            'R2': (0, 0, 0),  # Index
        }

        # Program Counter (relative to base)
        self.program_counter: Tuple[int, int] = (0, 2)  # Start at row 2

        # Stack Pointer (starts at row 2, grows down)
        self.stack_pointer: int = 2

        # Process metadata
        self.cycles_executed = 0
        self.parent_pid: int = 0

        # Initialize the process header
        self._update_header()

    def _update_header(self):
        """
        Update the visual process header (Row 0).

        Header Layout:
        - Pixel (0,0): PID (encoded as RGB)
        - Pixel (1,0): Status (color-coded)
        - Pixel (2,0): IP_X
        - Pixel (3,0): IP_Y
        - Pixel (4,0): Stack Pointer
        """
        memory = self.memory_manager.memory

        # PID at (0,0)
        memory[self.base_x, self.base_y] = (
            self.pid % 256,
            (self.pid // 256) % 256,
            0
        )

        # Status at (1,0)
        state_colors = {
            "READY": self.STATE_READY,
            "RUNNING": self.STATE_RUNNING,
            "BLOCKED": self.STATE_BLOCKED,
            "TERMINATED": self.STATE_TERMINATED,
            "KERNEL": self.STATE_KERNEL
        }
        memory[self.base_x + 1, self.base_y] = state_colors.get(self.state, (0, 0, 0))

        # IP at (2,0) and (3,0)
        memory[self.base_x + 2, self.base_y] = (self.program_counter[0], 0, 0)
        memory[self.base_x + 3, self.base_y] = (self.program_counter[1], 0, 0)

        # SP at (4,0)
        memory[self.base_x + 4, self.base_y] = (self.stack_pointer, 0, 0)

        # Registers in Row 1
        memory[self.base_x, self.base_y + 1] = self.registers['R0']
        memory[self.base_x + 1, self.base_y + 1] = self.registers['R1']
        memory[self.base_x + 2, self.base_y + 1] = self.registers['R2']

    def execute_step(self, memory: np.ndarray, heatmap: np.ndarray):
        """
        Execute one instruction cycle.

        For now, this runs a simple Game of Life simulation.
        In the future, this will be the Piet interpreter.

        Args:
            memory: Reference to the main memory array
            heatmap: Reference to the heatmap overlay
        """
        self.state = "RUNNING"
        self._update_header()

        # Execute one step of Game of Life
        self._game_of_life_step(memory, heatmap)

        self.cycles_executed += 1

        # Update instruction pointer (for demo purposes, just increment)
        ip_x, ip_y = self.program_counter
        ip_x = (ip_x + 1) % 16
        if ip_x == 0:
            ip_y = (ip_y + 1) % 16
        self.program_counter = (ip_x, ip_y)

        self.state = "READY"
        self._update_header()

    def initialize_game_of_life(self):
        """
        Initialize the process memory with a random Game of Life pattern.

        This is a demo - the process will run Conway's Game of Life
        in its allocated memory region.
        """
        memory = self.memory_manager.memory

        # Skip the header (rows 0-1) and initialize rows 2-15 with random noise
        for y in range(2, 16):
            for x in range(16):
                if random.random() > 0.7:  # 30% alive
                    memory[self.base_x + x, self.base_y + y] = (255, 255, 255)
                else:
                    memory[self.base_x + x, self.base_y + y] = (0, 0, 0)

    def _game_of_life_step(self, memory: np.ndarray, heatmap: np.ndarray):
        """
        Execute one step of Conway's Game of Life.

        Rules:
        - Any live cell with 2-3 neighbors survives
        - Any dead cell with exactly 3 neighbors becomes alive
        - All other cells die or stay dead

        Args:
            memory: Reference to the main memory array
            heatmap: Reference to the heatmap overlay
        """
        # Create a temporary buffer for the new state
        new_state = np.zeros((16, 16), dtype=np.uint8)

        # Only process rows 2-15 (skip header and registers)
        for y in range(2, 16):
            for x in range(16):
                # Count living neighbors
                neighbors = self._count_neighbors(memory, x, y)

                # Get current state
                current_pixel = memory[self.base_x + x, self.base_y + y]
                is_alive = np.sum(current_pixel) > 128  # Bright = alive

                # Apply Game of Life rules
                if is_alive:
                    new_state[x, y] = 1 if neighbors in [2, 3] else 0
                else:
                    new_state[x, y] = 1 if neighbors == 3 else 0

                # Update heatmap (mark accessed pixels as hot)
                heatmap[self.base_x + x, self.base_y + y] += 0.1

        # Apply the new state
        for y in range(2, 16):
            for x in range(16):
                if new_state[x, y] == 1:
                    # Alive: bright white
                    memory[self.base_x + x, self.base_y + y] = (255, 255, 255)
                else:
                    # Dead: black
                    memory[self.base_x + x, self.base_y + y] = (0, 0, 0)

    def _count_neighbors(self, memory: np.ndarray, x: int, y: int) -> int:
        """
        Count the number of living neighbors for a cell.

        Args:
            memory: Reference to the main memory array
            x, y: Cell coordinates (relative to process base)

        Returns:
            Number of living neighbors (0-8)
        """
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip the cell itself

                # Wrap around within the 16x16 block
                nx = (x + dx) % 16
                ny = max(2, min(15, y + dy))  # Clamp to valid rows

                neighbor_pixel = memory[self.base_x + nx, self.base_y + ny]
                if np.sum(neighbor_pixel) > 128:  # Bright = alive
                    count += 1

        return count

    def push_stack(self, value: Tuple[int, int, int]):
        """
        Push a value onto the process stack.

        Args:
            value: RGB tuple to push
        """
        memory = self.memory_manager.memory

        # Check stack overflow
        if self.stack_pointer >= 16:
            raise RuntimeError(f"Stack overflow in PID {self.pid}")

        # Write to stack
        sp_x = self.stack_pointer % 16
        sp_y = self.stack_pointer // 16
        memory[self.base_x + sp_x, self.base_y + sp_y] = value

        self.stack_pointer += 1
        self._update_header()

    def pop_stack(self) -> Tuple[int, int, int]:
        """
        Pop a value from the process stack.

        Returns:
            RGB tuple from the top of the stack

        Raises:
            RuntimeError: If stack is empty
        """
        memory = self.memory_manager.memory

        # Check stack underflow
        if self.stack_pointer <= 2:
            raise RuntimeError(f"Stack underflow in PID {self.pid}")

        self.stack_pointer -= 1

        # Read from stack
        sp_x = self.stack_pointer % 16
        sp_y = self.stack_pointer // 16
        value = tuple(memory[self.base_x + sp_x, self.base_y + sp_y])

        # Erase the popped value
        memory[self.base_x + sp_x, self.base_y + sp_y] = (0, 0, 0)

        self._update_header()
        return value

    def syscall(self, syscall_number: int, *args):
        """
        Execute a system call.

        This will be implemented to handle color-coded syscalls.

        Args:
            syscall_number: The syscall to execute
            *args: Syscall arguments
        """
        # TODO: Implement syscall handling
        pass

    def terminate(self):
        """Terminate the process and free its memory."""
        self.state = "TERMINATED"
        self._update_header()
        self.memory_manager.deallocate_block(self.pid)
