class PixelProcess:
    """Represents a single process in the Pixel OS."""
    def __init__(self, pid, parent_pid=0):
        self.pid = pid
        self.parent_pid = parent_pid
        self.state = "READY"  # Can be READY, RUNNING, BLOCKED, TERMINATED

        # Registers
        self.registers = {
            'R': (0, 0, 0),      # Red accumulator
            'G': (0, 0, 0),      # Green data register
            'B': (0, 0, 0),      # Blue address register
            'PC': (0, 0),        # Program Counter (x, y)
            'SP': (255, 255),    # Stack Pointer (defaults to bottom-right of a 256x256 region)
        }

        self.memory_region_base = None # Base coordinates of its allocated memory
        self.pixels_executed = 0

    def encode_position(self, pos):
        """Encodes a 2D position (like PC or SP) into two pixels."""
        x, y = pos
        # Simple encoding: x and y are stored in the Red channel of two separate pixels.
        return [(x, 0, 0), (y, 0, 0)]

    def decode_position(self, pixels):
        """Decodes two pixels back into a 2D position."""
        return (pixels[0][0], pixels[1][0])

    def save_context(self):
        """Saves the process's current register state into a list of pixels."""
        context_pixels = []
        # The order here must match load_context
        register_order = ['R', 'G', 'B', 'PC', 'SP']

        for reg_name in register_order:
            value = self.registers[reg_name]
            if reg_name in ['PC', 'SP']:
                # Position registers are encoded as two pixels
                context_pixels.extend(self.encode_position(value))
            else:
                # Color registers are stored as a single RGB pixel
                context_pixels.append(value)

        return context_pixels

    def load_context(self, context_pixels):
        """Loads the process's state from a list of pixels."""
        ptr = 0
        register_order = ['R', 'G', 'B', 'PC', 'SP']

        for reg_name in register_order:
            if reg_name in ['PC', 'SP']:
                self.registers[reg_name] = self.decode_position(context_pixels[ptr:ptr+2])
                ptr += 2
            else:
                self.registers[reg_name] = context_pixels[ptr]
                ptr += 1
