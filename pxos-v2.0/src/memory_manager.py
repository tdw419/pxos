from src.pixel_grid import PixelGrid

class PixelMemoryManager:
    """Manages the pixel-based physical memory of the OS."""
    def __init__(self, width=1920, height=1080):
        """Initializes the memory manager with a physical memory grid."""
        self.physical_memory = PixelGrid(width, height)
        self.page_size = (16, 16)  # A standard 16x16 pixel page
        self.page_table = {}
        self._next_free_region_x = 0
        self._next_free_region_y = 0

    def find_free_region(self, width=256, height=256):
        """
        Finds a contiguous free region of a given size.
        NOTE: This is a simple allocator that just finds the next available slot.
        """
        # Simple sequential allocation for now
        if self._next_free_region_x + width > self.physical_memory.width:
            self._next_free_region_x = 0
            self._next_free_region_y += height

        if self._next_free_region_y + height > self.physical_memory.height:
            raise MemoryError("Out of physical memory")

        base_coords = (self._next_free_region_x, self._next_free_region_y)
        self._next_free_region_x += width

        return base_coords

    def allocate_process_space(self, pid, width=256, height=256):
        """Allocates a contiguous pixel region for a new process."""
        try:
            base_region_coords = self.find_free_region(width, height)
            self.page_table[pid] = {
                'base': base_region_coords,
                'width': width,
                'height': height,
                'heap_ptr': (0, 0),  # Relative to base
                'stack_ptr': (width - 1, height - 1)  # Bottom-right of region
            }
            return base_region_coords
        except MemoryError as e:
            print(f"Failed to allocate space for PID {pid}: {e}")
            return None

    def free_process_space(self, pid):
        """Frees the memory region associated with a process."""
        if pid in self.page_table:
            # In a more complex system, we would mark these regions as free.
            # For this simple sequential allocator, we can't easily reuse space.
            del self.page_table[pid]
            return True
        return False

    def pixel_to_physical(self, virtual_pixel, pid):
        """Converts a process's virtual pixel coordinates to physical coordinates."""
        if pid not in self.page_table:
            raise ValueError(f"Invalid PID: {pid}")

        base_x, base_y = self.page_table[pid]['base']
        virtual_x, virtual_y = virtual_pixel

        # Boundary checks
        if not (0 <= virtual_x < self.page_table[pid]['width'] and 0 <= virtual_y < self.page_table[pid]['height']):
            raise IndexError("Virtual pixel coordinates are out of bounds for the process memory region.")

        return (base_x + virtual_x, base_y + virtual_y)

    def read_pixel(self, physical_coords):
        """Reads a pixel directly from physical memory."""
        return self.physical_memory[physical_coords]

    def write_pixel(self, physical_coords, value):
        """Writes a pixel directly to physical memory."""
        self.physical_memory[physical_coords] = value
