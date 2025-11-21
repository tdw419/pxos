"""
pxos/vram_sim.py

Simulated VRAM substrate - the primary development surface for pxOS.

This is a Python-backed "virtual texture" that agents can read/write to build the OS.
Real GPU VRAM is just an I/O backend that can be swapped in later.

Benefits:
- Safety: crash/corrupt/reset without bricking hardware
- Observability: print, diff, visualize pixel patterns
- Determinism: no driver/GPU scheduling weirdness
- Instrumentation: log every read/write, step through changes
- Portability: runs on any machine (even without GPU)
- Bridging: PNG save/load matches real hardware pipeline
"""

from dataclasses import dataclass
import numpy as np
from PIL import Image
from typing import Tuple

RGBA = Tuple[int, int, int, int]


@dataclass
class SimulatedVRAM:
    """
    A simulated VRAM surface backed by numpy/PIL.

    This is the substrate the agent works on to build pxOS.
    PNGs and real GPU VRAM are just I/O backends.
    """
    width: int
    height: int

    def __post_init__(self):
        # uint8 RGBA texture
        self.buffer = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        # Initialize with full alpha
        self.buffer[:, :, 3] = 255

    # --- Core pixel operations ---

    def read_pixel(self, x: int, y: int) -> RGBA:
        """Read a single pixel at (x, y)."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"Pixel ({x}, {y}) out of bounds for {self.width}x{self.height} VRAM")
        r, g, b, a = self.buffer[y, x]
        return int(r), int(g), int(b), int(a)

    def write_pixel(self, x: int, y: int, rgba: RGBA) -> None:
        """Write a single pixel at (x, y)."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"Pixel ({x}, {y}) out of bounds for {self.width}x{self.height} VRAM")
        r, g, b, a = rgba
        self.buffer[y, x] = [r, g, b, a]

    def fill_rect(self, x0: int, y0: int, w: int, h: int, rgba: RGBA) -> None:
        """Fill a rectangle with a solid color."""
        # Clamp to VRAM bounds
        x1 = max(0, x0)
        y1 = max(0, y0)
        x2 = min(self.width, x0 + w)
        y2 = min(self.height, y0 + h)

        if x2 <= x1 or y2 <= y1:
            return  # Nothing to fill

        r, g, b, a = rgba
        self.buffer[y1:y2, x1:x2] = [r, g, b, a]

    def copy_rect(self, src_x: int, src_y: int, w: int, h: int,
                  dst_x: int, dst_y: int) -> None:
        """Copy a rectangle from one location to another within VRAM."""
        # Clamp source
        sx1 = max(0, src_x)
        sy1 = max(0, src_y)
        sx2 = min(self.width, src_x + w)
        sy2 = min(self.height, src_y + h)

        # Clamp destination
        dx1 = max(0, dst_x)
        dy1 = max(0, dst_y)
        dx2 = min(self.width, dst_x + w)
        dy2 = min(self.height, dst_y + h)

        # Compute actual overlap
        actual_w = min(sx2 - sx1, dx2 - dx1)
        actual_h = min(sy2 - sy1, dy2 - dy1)

        if actual_w <= 0 or actual_h <= 0:
            return

        # Copy pixels
        self.buffer[dy1:dy1+actual_h, dx1:dx1+actual_w] = \
            self.buffer[sy1:sy1+actual_h, sx1:sx1+actual_w].copy()

    # --- Persistence ---

    def save_png(self, path: str) -> None:
        """Save VRAM to a PNG file."""
        img = Image.fromarray(self.buffer, mode="RGBA")
        img.save(path)

    @classmethod
    def load_png(cls, path: str) -> "SimulatedVRAM":
        """Load VRAM from a PNG file."""
        img = Image.open(path).convert("RGBA")
        arr = np.array(img, dtype=np.uint8)
        h, w, _ = arr.shape
        vram = cls(w, h)
        vram.buffer[:] = arr
        return vram

    # --- Inspection ---

    def get_region_slice(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Get a numpy view of a region (useful for analysis)."""
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.width, x + w)
        y2 = min(self.height, y + h)
        return self.buffer[y1:y2, x1:x2]

    def count_unique_colors(self, x: int, y: int, w: int, h: int) -> int:
        """Count unique RGBA colors in a region."""
        region = self.get_region_slice(x, y, w, h)
        # Reshape to list of colors
        colors = region.reshape(-1, 4)
        # Convert to tuples for set uniqueness
        unique = set(tuple(c) for c in colors)
        return len(unique)

    def __repr__(self):
        return f"SimulatedVRAM({self.width}x{self.height})"
