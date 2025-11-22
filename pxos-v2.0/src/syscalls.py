from src.process import PixelProcess
from src.memory_manager import PixelMemoryManager
from src.scheduler import PixelScheduler

class PixelSystemCalls:
    """Handles system calls triggered by specific pixel color codes."""
    # System call numbers are encoded as specific colors
    SYS_FORK = (255, 0, 0)       # Red = fork()
    SYS_READ = (0, 255, 0)       # Green = read()
    SYS_WRITE = (0, 0, 255)      # Blue = write()
    SYS_EXIT = (255, 255, 0)     # Yellow = exit()
    SYS_SCHED_YIELD = (255, 0, 255) # Magenta = sched_yield()

    def __init__(self, memory_manager: PixelMemoryManager, scheduler: PixelScheduler):
        self.memory_manager = memory_manager
        self.scheduler = scheduler
        self._next_pid = 2  # PID 1 is reserved for init

    def _allocate_pid(self):
        """Allocates a new, unique Process ID."""
        pid = self._next_pid
        self._next_pid += 1
        return pid

    def handle_syscall(self, syscall_pixel, process: PixelProcess):
        """Dispatches a system call based on the pixel color."""
        if syscall_pixel == self.SYS_FORK:
            return self.sys_fork(process)
        elif syscall_pixel == self.SYS_READ:
            return self.sys_read(process)
        elif syscall_pixel == self.SYS_WRITE:
            return self.sys_write(process)
        elif syscall_pixel == self.SYS_EXIT:
            return self.sys_exit(process)
        elif syscall_pixel == self.SYS_SCHED_YIELD:
            return self.sys_sched_yield(process)
        else:
            # Not a syscall, treat as a normal instruction
            return None

    def sys_fork(self, parent_process: PixelProcess):
        """Creates a child process by copying the parent's memory and state."""
        print(f"--- SYSCALL: FORK initiated by PID {parent_process.pid} ---")

        # 1. Allocate a new PID for the child
        child_pid = self._allocate_pid()
        child_process = PixelProcess(child_pid, parent_pid=parent_process.pid)

        # 2. Allocate a new memory region for the child
        parent_mem_info = self.memory_manager.page_table[parent_process.pid]
        child_mem_base = self.memory_manager.allocate_process_space(
            child_pid,
            width=parent_mem_info['width'],
            height=parent_mem_info['height']
        )
        if child_mem_base is None:
            # Fork fails if no memory is available
            parent_process.registers['R'] = (255, 0, 0) # Error code
            return [parent_process]

        # 3. Copy parent's memory region to the child's region
        parent_base = parent_mem_info['base']
        width, height = parent_mem_info['width'], parent_mem_info['height']

        source_region = self.memory_manager.physical_memory.get_region(parent_base[0], parent_base[1], width, height)
        self.memory_manager.physical_memory.set_region(child_mem_base[0], child_mem_base[1], source_region)

        # 4. Copy parent's state (registers) to the child
        child_process.registers = parent_process.registers.copy()

        # 5. Set return values for fork()
        # Child gets return value 0 (encoded as a black pixel)
        child_process.registers['R'] = (0, 0, 0)
        # Parent gets the child's PID (encoded in the Red channel)
        parent_process.registers['R'] = (child_pid, 0, 0)

        print(f"--- FORK successful: Parent {parent_process.pid} created Child {child_pid} ---")

        # 6. Add the child process to the scheduler's ready queue
        self.scheduler.add_process(child_process)

        # Both processes are now ready to run
        return [parent_process, child_process]

    def sys_read(self, process):
        """Stub for the read system call."""
        print(f"--- SYSCALL: READ (not implemented) by PID {process.pid} ---")
        # In a real implementation, this would block the process and wait for input.
        return [process]

    def sys_write(self, process):
        """Stub for the write system call."""
        print(f"--- SYSCALL: WRITE (not implemented) by PID {process.pid} ---")
        # In a real implementation, this would write data from the process's memory to an output device.
        return [process]

    def sys_exit(self, process):
        """Terminates the current process."""
        print(f"--- SYSCALL: EXIT initiated by PID {process.pid} ---")
        process.state = "TERMINATED"
        self.memory_manager.free_process_space(process.pid)
        return []  # Process is removed from the system

    def sys_sched_yield(self, process):
        """Voluntarily yields the CPU to another process."""
        print(f"--- SYSCALL: SCHED_YIELD by PID {process.pid} ---")
        process.state = "READY"
        self.scheduler.add_process(process)
        # The main loop will call the scheduler to get the next process
        return [process]
