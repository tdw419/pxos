from collections import deque
from src.process import PixelProcess

class PixelQueue:
    """A simple FIFO queue for managing processes, implemented using a deque."""
    def __init__(self):
        self._items = deque()

    def enqueue(self, item):
        """Adds a process to the end of the queue."""
        self._items.append(item)

    def dequeue(self):
        """Removes and returns a process from the front of the queue."""
        if self.is_empty():
            return None
        return self._items.popleft()

    def is_empty(self):
        """Checks if the queue is empty."""
        return len(self._items) == 0

    def __len__(self):
        return len(self._items)

class PixelScheduler:
    """Manages process scheduling using a round-robin algorithm."""
    def __init__(self, memory_manager):
        self.ready_queue = PixelQueue()
        self.current_process = None
        self.memory_manager = memory_manager
        self.time_slice = 1000  # Number of pixels to execute before a context switch

    def add_process(self, process):
        """Adds a new process to the ready queue."""
        if isinstance(process, PixelProcess):
            process.state = "READY"
            self.ready_queue.enqueue(process)
        else:
            raise TypeError("Only PixelProcess objects can be added to the scheduler.")

    def schedule(self):
        """Selects the next process to run."""
        old_process = self.current_process

        # If the current process has exceeded its time slice, preempt it.
        if old_process and old_process.state == "RUNNING" and old_process.pixels_executed >= self.time_slice:
            old_process.state = "READY"
            old_process.pixels_executed = 0  # Reset counter
            self.ready_queue.enqueue(old_process)
            self.current_process = None

        # If there is no current process, select the next one from the ready queue.
        if self.current_process is None and not self.ready_queue.is_empty():
            new_process = self.ready_queue.dequeue()

            # Perform the context switch
            self.context_switch(old_process, new_process)

            self.current_process = new_process
            self.current_process.state = "RUNNING"

        return self.current_process

    def context_switch(self, old_process, new_process):
        """
        Performs a context switch from the old process to the new process.
        This is a conceptual implementation; the actual pixel manipulation will be
        handled by the main execution loop.
        """
        print(f"Context switching...")

        # Save old process context (if there was one)
        if old_process:
            print(f"Saving context for PID {old_process.pid}")
            old_context_pixels = old_process.save_context()
            # In a real scenario, we would write these pixels to a specific memory area.
            # For now, we just demonstrate the mechanism.
            # Example: self.memory_manager.write_context(old_process.pid, old_context_pixels)

        # Load new process context
        if new_process:
            print(f"Loading context for PID {new_process.pid}")
            # In a real scenario, we would read pixels from memory to load the context.
            # Example: new_context_pixels = self.memory_manager.read_context(new_process.pid)
            # new_process.load_context(new_context_pixels)

        print("Context switch complete.")

    def get_running_process(self):
        """Returns the currently running process."""
        return self.current_process
