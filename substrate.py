import numpy as np

class Substrate:
    """A base class representing the VRAM substrate."""
    def __init__(self, width=4096, height=4096):
        self.width = width
        self.height = height
        self.vram = np.zeros((height, width, 4), dtype=np.float32)
        print(f"Substrate initialized with size {width}x{height}")

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Reads a region of pixels from the VRAM."""
        return self.vram[y:y+height, x:x+width]

    def inject(self, pattern: np.ndarray, x: int, y: int):
        """Injects a pattern of pixels into the VRAM."""
        h, w, _ = pattern.shape
        self.vram[y:y+h, x:x+w] = pattern
