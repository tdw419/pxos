"""
Pixel Scheduler
Round-Robin scheduler with visual scanline metaphor
"""

from typing import List, Optional
from pixel_process import PixelProcess


class PixelScheduler:
    """
    The Pixel Scheduler - determines which process gets CPU time.

    Implements Round-Robin scheduling with a visual "scanline" cursor
    that moves across the screen visiting process blocks.

    Key Features:
    - Round-robin fairness
    - Configurable time slice (in frames)
    - Visual scanline cursor
    - Process state tracking
    """

    def __init__(self, time_slice: int = 100):
        """
        Initialize the scheduler.

        Args:
            time_slice: Number of frames each process gets before switching
        """
        self.processes: List[PixelProcess] = []
        self.current_process_index: int = 0
        self.time_slice: int = time_slice
        self.cycles_executed: int = 0

        # Scanline position (for visualization)
        self.scanline_y: int = 0

    def add_process(self, process: PixelProcess):
        """
        Add a new process to the scheduler.

        Args:
            process: The process to add
        """
        self.processes.append(process)
        print(f"[SCHEDULER] Added PID {process.pid} to scheduler (total: {len(self.processes)})")

    def remove_process(self, pid: int):
        """
        Remove a process from the scheduler.

        Args:
            pid: Process ID to remove

        Raises:
            ValueError: If PID not found
        """
        for i, proc in enumerate(self.processes):
            if proc.pid == pid:
                del self.processes[i]
                print(f"[SCHEDULER] Removed PID {pid} from scheduler (remaining: {len(self.processes)})")

                # Adjust current process index if needed
                if self.current_process_index >= len(self.processes):
                    self.current_process_index = 0
                return

        raise ValueError(f"Process PID {pid} not found in scheduler")

    def get_next_process(self) -> Optional[PixelProcess]:
        """
        Get the next process to execute (Round-Robin).

        Returns:
            The next process, or None if no processes exist

        Algorithm:
        1. If time slice expired, move to next process
        2. Skip any BLOCKED or TERMINATED processes
        3. Return the current process
        """
        if not self.processes:
            return None

        # Check if time slice has expired
        if self.cycles_executed >= self.time_slice:
            self._context_switch()

        # Skip blocked/terminated processes
        attempts = 0
        max_attempts = len(self.processes)

        while attempts < max_attempts:
            current = self.processes[self.current_process_index]

            if current.state not in ["BLOCKED", "TERMINATED"]:
                self.cycles_executed += 1
                return current

            # Move to next process
            self.current_process_index = (self.current_process_index + 1) % len(self.processes)
            attempts += 1

        # No runnable processes
        return None

    def _context_switch(self):
        """
        Perform a context switch to the next process.

        This saves the state of the current process and loads the next one.
        """
        if not self.processes:
            return

        # Save context of current process (implicit - registers are pixels)
        current = self.processes[self.current_process_index]
        current.state = "READY"
        current._update_header()

        # Move to next process (round-robin)
        self.current_process_index = (self.current_process_index + 1) % len(self.processes)

        # Reset time slice counter
        self.cycles_executed = 0

        # Load context of new process (implicit - just read the pixels)
        new_process = self.processes[self.current_process_index]
        print(f"[SCHEDULER] Context switch: PID {current.pid} -> PID {new_process.pid}")

    def get_scanline_position(self) -> int:
        """
        Get the current scanline Y position.

        The scanline is a visual metaphor - it moves down the screen,
        visiting processes in the order they appear in memory.

        Returns:
            Y coordinate of the scanline
        """
        return self.scanline_y

    def update_scanline(self, screen_height: int):
        """
        Update the scanline position.

        The scanline moves down the screen at a constant rate.

        Args:
            screen_height: Total screen height
        """
        self.scanline_y = (self.scanline_y + 1) % screen_height

    def print_schedule(self):
        """Print the current process schedule."""
        print("\n[SCHEDULER] Current Schedule:")
        print(f"  Time Slice: {self.time_slice} frames")
        print(f"  Current Process: PID {self.processes[self.current_process_index].pid if self.processes else 'None'}")
        print(f"  Cycles Executed: {self.cycles_executed}/{self.time_slice}")
        print(f"  Total Processes: {len(self.processes)}")

        for i, proc in enumerate(self.processes):
            marker = ">>>" if i == self.current_process_index else "   "
            print(f"  {marker} PID {proc.pid}: {proc.state} (cycles: {proc.cycles_executed})")
