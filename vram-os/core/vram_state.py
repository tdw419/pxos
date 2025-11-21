#!/usr/bin/env python3
"""
VRAM OS - VRAMState Storage System
Manages the unified VRAM memory space as a pixel grid
"""

import json
import base64
import hashlib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict
import struct


@dataclass
class MemoryRegion:
    """Defines a region in the VRAM grid"""
    x: int
    y: int
    width: int
    height: int
    name: str


class VRAMState:
    """
    Unified VRAM state - the entire OS lives here
    This is the single source of truth for all OS state
    """

    def __init__(self, width: int = 1024, height: int = 1024):
        """Initialize empty VRAM state"""
        self.width = width
        self.height = height

        # RGBA pixel array (4 bytes per pixel)
        self.pixels = bytearray(width * height * 4)

        # Memory map - where things live in the 2D space
        self.memory_map = {
            'bootloader': MemoryRegion(0, 0, 64, 1, 'bootloader'),
            'kernel': MemoryRegion(0, 1, 1024, 256, 'kernel'),
            'registers': MemoryRegion(0, 257, 16, 1, 'registers'),
            'stack': MemoryRegion(0, 258, 256, 256, 'stack'),
            'heap': MemoryRegion(256, 258, 768, 768, 'heap'),
            'program_counter': MemoryRegion(16, 257, 1, 1, 'program_counter')
        }

        self.version = "0.1.0"

    def write_pixel(self, x: int, y: int, r: int, g: int, b: int, a: int) -> None:
        """Write RGBA value to pixel at (x, y)"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            raise ValueError(f"Pixel coordinates ({x}, {y}) out of bounds")

        if any(v < 0 or v > 255 for v in [r, g, b, a]):
            raise ValueError(f"RGBA values must be 0-255")

        idx = (y * self.width + x) * 4
        self.pixels[idx:idx+4] = bytes([r, g, b, a])

    def read_pixel(self, x: int, y: int) -> Tuple[int, int, int, int]:
        """Read RGBA value from pixel at (x, y)"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            raise ValueError(f"Pixel coordinates ({x}, {y}) out of bounds")

        idx = (y * self.width + x) * 4
        return tuple(self.pixels[idx:idx+4])

    def clear_region(self, region: MemoryRegion, r: int = 0, g: int = 0, b: int = 0, a: int = 0) -> None:
        """Clear a memory region to a specific color"""
        for y in range(region.y, region.y + region.height):
            for x in range(region.x, region.x + region.width):
                self.write_pixel(x, y, r, g, b, a)

    def get_checksum(self) -> str:
        """Calculate SHA-256 checksum of pixel data"""
        return hashlib.sha256(self.pixels).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for JSON storage)"""
        return {
            'width': self.width,
            'height': self.height,
            'pixels': base64.b64encode(self.pixels).decode('ascii'),
            'memory_map': {
                name: {
                    'x': region.x,
                    'y': region.y,
                    'width': region.width,
                    'height': region.height,
                    'name': region.name
                }
                for name, region in self.memory_map.items()
            },
            'version': self.version,
            'checksum': self.get_checksum()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VRAMState':
        """Deserialize from dictionary"""
        state = cls(width=data['width'], height=data['height'])
        state.pixels = bytearray(base64.b64decode(data['pixels']))
        state.version = data.get('version', '0.1.0')

        # Restore memory map
        if 'memory_map' in data:
            state.memory_map = {
                name: MemoryRegion(**region_data)
                for name, region_data in data['memory_map'].items()
            }

        return state

    def save_json(self, filepath: Path) -> None:
        """Save VRAMState to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"VRAMState saved to {filepath}")

    @classmethod
    def load_json(cls, filepath: Path) -> 'VRAMState':
        """Load VRAMState from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def export_png(self, filepath: Path) -> None:
        """Export VRAMState as PNG image"""
        try:
            from PIL import Image
            import numpy as np

            # Reshape pixels to (height, width, 4) RGBA array
            pixels_array = np.frombuffer(self.pixels, dtype=np.uint8)
            pixels_array = pixels_array.reshape((self.height, self.width, 4))

            # Create PIL Image
            img = Image.fromarray(pixels_array, mode='RGBA')
            img.save(filepath)
            print(f"VRAMState exported as PNG to {filepath}")

        except ImportError:
            print("Warning: PIL (Pillow) not available. Cannot export PNG.")
            print("Install with: pip install Pillow")

    @classmethod
    def from_png(cls, filepath: Path) -> 'VRAMState':
        """Load VRAMState from PNG image"""
        try:
            from PIL import Image
            import numpy as np

            img = Image.open(filepath).convert('RGBA')
            width, height = img.size

            state = cls(width=width, height=height)
            pixels_array = np.array(img)
            state.pixels = bytearray(pixels_array.flatten())

            print(f"VRAMState loaded from PNG: {filepath}")
            return state

        except ImportError:
            raise ImportError("PIL (Pillow) required to load PNG. Install with: pip install Pillow")


if __name__ == "__main__":
    # Test VRAMState
    print("Creating VRAMState (1024x1024)...")
    state = VRAMState(1024, 1024)

    # Write some test pixels
    state.write_pixel(0, 0, 255, 0, 0, 255)  # Red pixel at (0,0)
    state.write_pixel(1, 0, 0, 255, 0, 255)  # Green pixel at (1,0)
    state.write_pixel(2, 0, 0, 0, 255, 255)  # Blue pixel at (2,0)

    # Read back
    print(f"Pixel (0,0): {state.read_pixel(0, 0)}")
    print(f"Pixel (1,0): {state.read_pixel(1, 0)}")
    print(f"Pixel (2,0): {state.read_pixel(2, 0)}")

    # Save to JSON
    state.save_json(Path("test_vram_state.json"))

    # Load back
    state2 = VRAMState.load_json(Path("test_vram_state.json"))
    print(f"Loaded state checksum: {state2.get_checksum()}")

    print("\nVRAMState system working! ✓")
