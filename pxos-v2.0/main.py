from src.memory_manager import PixelMemoryManager
from src.scheduler import PixelScheduler
from src.process import PixelProcess
from src.syscalls import PixelSystemCalls
from src.file_system import PixelFileSystem

class PixelOS:
    """The main class for the Pixel OS simulation."""
    def __init__(self):
        print("Booting Pixel OS...")
        self.memory_manager = PixelMemoryManager(width=1024, height=768)
        self.scheduler = PixelScheduler(self.memory_manager)
        self.syscall_handler = PixelSystemCalls(self.memory_manager, self.scheduler)
        self.file_system = PixelFileSystem()
        self.running = True
        print("--- Core components initialized ---")

    def boot_loader(self):
        """Simulates the boot process, creates the init process, and loads a program."""
        print("--- Starting Boot Loader ---")

        # 1. Create the 'init' process (PID 1)
        init_process = PixelProcess(pid=1)

        # 2. Allocate memory for the init process
        init_mem_base = self.memory_manager.allocate_process_space(pid=1, width=256, height=256)
        if init_mem_base is None:
            raise MemoryError("Failed to allocate memory for the init process.")

        init_process.memory_region_base = init_mem_base
        print(f"Allocated memory for PID 1 at {init_mem_base}")

        # 3. Create a simple program with a FORK and EXIT syscall
        # Instruction at (0,0) is FORK
        # Instruction at (1,0) is EXIT
        program_start_phys = self.memory_manager.pixel_to_physical((0, 0), pid=1)
        self.memory_manager.write_pixel(program_start_phys, PixelSystemCalls.SYS_FORK)

        program_exit_phys = self.memory_manager.pixel_to_physical((1, 0), pid=1)
        self.memory_manager.write_pixel(program_exit_phys, PixelSystemCalls.SYS_EXIT)

        print("Loaded simple program into PID 1's memory.")

        # 4. Set the program counter and add the process to the scheduler
        init_process.registers['PC'] = (0, 0)
        self.scheduler.add_process(init_process)

        print("--- Boot Loader finished. Handing control to scheduler ---")

    def run(self):
        """The main execution loop of the OS."""
        self.boot_loader()

        loop_count = 0
        max_loops = 10  # To prevent infinite loops during simulation

        while self.running and loop_count < max_loops:
            print(f"\n--- OS Loop {loop_count} ---")

            current_process = self.scheduler.schedule()

            if current_process is None:
                print("No active processes. Halting system.")
                self.running = False
                continue

            print(f"Running PID: {current_process.pid}, PC: {current_process.registers['PC']}")

            # Fetch the instruction (pixel) from memory
            try:
                pc_coords = current_process.registers['PC']
                physical_pc = self.memory_manager.pixel_to_physical(pc_coords, current_process.pid)
                instruction_pixel = self.memory_manager.read_pixel(physical_pc)
            except (IndexError, ValueError) as e:
                print(f"Error fetching instruction for PID {current_process.pid}: {e}. Terminating process.")
                current_process.state = "TERMINATED"
                continue

            print(f"  Instruction Pixel: {instruction_pixel}")

            # Check for system call
            result = self.syscall_handler.handle_syscall(instruction_pixel, current_process)

            if result is not None:
                # The process might have been terminated by the syscall
                if not any(p.pid == current_process.pid for p in result if hasattr(p, 'pid')):
                     self.scheduler.current_process = None # Force scheduler to pick a new process
                pass # Syscall handled
            else:
                # This is a regular instruction (not implemented yet)
                print(f"  (Regular instruction - not implemented)")
                pass

            # Advance Program Counter for the currently running process
            if self.scheduler.current_process and self.scheduler.current_process.state == "RUNNING":
                current_pc_x, current_pc_y = self.scheduler.current_process.registers['PC']
                self.scheduler.current_process.registers['PC'] = (current_pc_x + 1, current_pc_y)
                self.scheduler.current_process.pixels_executed += 1

            loop_count += 1

        print("\n--- Pixel OS Simulation Finished ---")


if __name__ == "__main__":
    os_instance = PixelOS()
    os_instance.run()
