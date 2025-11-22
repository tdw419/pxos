"""
Pure Pixel Kernel - Main Kernel Implementation
State == Color: The operating system where RAM is the display
"""

import pygame
import numpy as np
from typing import Optional, Tuple
import sys

from pixel_memory_manager import PixelMemoryManager
from pixel_scheduler import PixelScheduler
from pixel_process import PixelProcess


class PixelKernel:
    """
    The core Pure Pixel Kernel.

    This is the main OS class that manages:
    - Physical memory (pixel grid)
    - Process scheduling
    - System calls
    - Display rendering
    """

    def __init__(self, width: int = 800, height: int = 600):
        """
        Initialize the Pure Pixel Kernel.

        Args:
            width: Screen width in pixels
            height: Screen height in pixels
        """
        self.width = width
        self.height = height

        # Physical Memory: A 3D NumPy array (W, H, 3) representing RGB pixels
        # This IS the RAM. If it's not visible, it doesn't exist.
        self.memory = np.zeros((width, height, 3), dtype=np.uint8)

        # Subsystems
        self.pmm = PixelMemoryManager(self.memory, width, height)
        self.scheduler = PixelScheduler()

        # Kernel state
        self.running = False
        self.clock = None
        self.screen = None

        # Performance tracking
        self.frame_count = 0
        self.total_instructions = 0

        # Heatmap overlay for visualization
        self.heatmap = np.zeros((width, height), dtype=np.float32)
        self.heatmap_decay = 0.95  # Heat decay per frame

    def boot(self):
        """
        Boot the Pixel OS.

        This initializes Pygame, creates the display surface, and starts the main loop.
        """
        print("=" * 60)
        print("PURE PIXEL KERNEL v1.0")
        print("State == Color | RAM == Display")
        print("=" * 60)
        print(f"Initializing {self.width}x{self.height} pixel memory...")

        # Initialize Pygame
        pygame.init()

        # Create display with hardware acceleration and double buffering
        self.screen = pygame.display.set_mode(
            (self.width, self.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Pure Pixel Kernel v1.0")

        self.clock = pygame.time.Clock()

        print(f"Available memory: {self.pmm.get_free_memory()} pixels")
        print(f"Page size: 16x16 = 256 pixels = 768 bytes")

        # Create initial processes
        self._create_init_processes()

        print("Kernel booted successfully. Starting main loop...")
        print("=" * 60)

        # Start the main kernel loop
        self.running = True
        self.run_kernel_loop()

    def _create_init_processes(self):
        """Create the initial process ecosystem."""
        # Process 1: The "init" process - a simple Game of Life simulation
        try:
            proc1 = PixelProcess(1, self.pmm)
            proc1.initialize_game_of_life()
            self.scheduler.add_process(proc1)
            print(f"[BOOT] Created PID 1 (init) at ({proc1.base_x}, {proc1.base_y})")
        except MemoryError as e:
            print(f"[ERROR] Failed to create init process: {e}")

        # Process 2: A second GoL instance
        try:
            proc2 = PixelProcess(2, self.pmm)
            proc2.initialize_game_of_life()
            self.scheduler.add_process(proc2)
            print(f"[BOOT] Created PID 2 at ({proc2.base_x}, {proc2.base_y})")
        except MemoryError as e:
            print(f"[ERROR] Failed to create second process: {e}")

    def run_kernel_loop(self):
        """
        Main kernel loop.

        This is the heart of the OS. It:
        1. Schedules processes (round-robin)
        2. Executes instructions
        3. Renders memory to screen
        4. Handles input events
        """
        while self.running:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.shutdown()
                elif event.type == pygame.KEYDOWN:
                    self._handle_keypress(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_click(event.pos)

            # 1. Scheduler: Pick the next process to execute
            current_proc = self.scheduler.get_next_process()

            # 2. Execution: Run one instruction cycle
            if current_proc:
                self._execute_process(current_proc)

            # 3. Decay heatmap
            self.heatmap *= self.heatmap_decay

            # 4. Render: Blit the NumPy memory array to the screen
            # This is the fastest way to update the display
            pygame.surfarray.blit_array(self.screen, self.memory)

            # 5. Visual overlays (heatmap, process borders)
            self._render_overlays()

            # 6. Update display
            pygame.display.flip()

            # 7. Maintain 60 FPS
            self.clock.tick(60)
            self.frame_count += 1

            # Print stats every 60 frames (1 second)
            if self.frame_count % 60 == 0:
                self._print_stats()

    def _execute_process(self, process: PixelProcess):
        """
        Execute one instruction cycle for the given process.

        Args:
            process: The process to execute
        """
        # For now, we execute the Game of Life rules
        # In the future, this will be the Piet interpreter
        process.execute_step(self.memory, self.heatmap)
        self.total_instructions += 1

    def _render_overlays(self):
        """Render visual overlays: heatmap, process borders, scheduler cursor."""
        # Draw process borders
        for process in self.scheduler.processes:
            color = (0, 255, 0) if process.state == "RUNNING" else (100, 100, 100)
            rect = pygame.Rect(process.base_x, process.base_y, 16, 16)
            pygame.draw.rect(self.screen, color, rect, 1)

        # Draw heatmap overlay (optional, can be toggled)
        # This shows "hot" pixels that are frequently accessed
        # heatmap_surface = self._create_heatmap_surface()
        # self.screen.blit(heatmap_surface, (0, 0))

    def _handle_keypress(self, key):
        """Handle keyboard input."""
        if key == pygame.K_ESCAPE:
            self.shutdown()
        elif key == pygame.K_SPACE:
            # Pause/unpause
            print("[KERNEL] Execution paused (press SPACE to resume)")
            paused = True
            while paused:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.shutdown()
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        paused = False
                        print("[KERNEL] Execution resumed")
        elif key == pygame.K_n:
            # Spawn a new process
            try:
                new_pid = len(self.scheduler.processes) + 1
                new_proc = PixelProcess(new_pid, self.pmm)
                new_proc.initialize_game_of_life()
                self.scheduler.add_process(new_proc)
                print(f"[KERNEL] Spawned PID {new_pid} at ({new_proc.base_x}, {new_proc.base_y})")
            except MemoryError as e:
                print(f"[ERROR] Cannot spawn process: {e}")
        elif key == pygame.K_d:
            # Print debug info
            self._print_debug_info()

    def _handle_mouse_click(self, pos: Tuple[int, int]):
        """
        Handle mouse click - spawn a new process at the clicked location.

        Args:
            pos: (x, y) pixel coordinates of the click
        """
        try:
            new_pid = len(self.scheduler.processes) + 1
            new_proc = PixelProcess(new_pid, self.pmm)
            new_proc.initialize_game_of_life()
            self.scheduler.add_process(new_proc)
            print(f"[KERNEL] Spawned PID {new_pid} at click ({pos[0]}, {pos[1]}) -> ({new_proc.base_x}, {new_proc.base_y})")
        except MemoryError as e:
            print(f"[ERROR] Cannot spawn process: {e}")

    def _print_stats(self):
        """Print kernel statistics."""
        fps = self.clock.get_fps()
        free_mem = self.pmm.get_free_memory()
        total_mem = self.width * self.height
        used_mem = total_mem - free_mem
        mem_percent = (used_mem / total_mem) * 100

        print(f"[STATS] FPS: {fps:.1f} | "
              f"Processes: {len(self.scheduler.processes)} | "
              f"Instructions: {self.total_instructions} | "
              f"Memory: {used_mem}/{total_mem} ({mem_percent:.1f}%)")

    def _print_debug_info(self):
        """Print detailed debug information."""
        print("\n" + "=" * 60)
        print("DEBUG INFO")
        print("=" * 60)
        print(f"Frame: {self.frame_count}")
        print(f"FPS: {self.clock.get_fps():.1f}")
        print(f"Total Instructions: {self.total_instructions}")
        print(f"\nMemory:")
        print(f"  Total: {self.width * self.height} pixels")
        print(f"  Free: {self.pmm.get_free_memory()} pixels")
        print(f"  Used: {self.width * self.height - self.pmm.get_free_memory()} pixels")
        print(f"\nProcesses:")
        for proc in self.scheduler.processes:
            print(f"  PID {proc.pid}: State={proc.state}, "
                  f"Location=({proc.base_x}, {proc.base_y}), "
                  f"IP=({proc.program_counter[0]}, {proc.program_counter[1]})")
        print(f"\nScheduler:")
        print(f"  Algorithm: Round-Robin")
        print(f"  Time Slice: {self.scheduler.time_slice} frames")
        print(f"  Current Process: PID {self.scheduler.current_process_index + 1 if self.scheduler.processes else 'None'}")
        print("=" * 60 + "\n")

    def shutdown(self):
        """Gracefully shutdown the kernel."""
        print("\n" + "=" * 60)
        print("KERNEL SHUTDOWN")
        print("=" * 60)
        print(f"Total uptime: {self.frame_count} frames ({self.frame_count / 60:.1f} seconds)")
        print(f"Total instructions executed: {self.total_instructions}")
        print(f"Final process count: {len(self.scheduler.processes)}")
        print("=" * 60)

        self.running = False
        pygame.quit()
        sys.exit(0)


def main():
    """Entry point for the Pixel OS."""
    kernel = PixelKernel(width=800, height=600)
    kernel.boot()


if __name__ == "__main__":
    main()
