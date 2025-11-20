"""
VRAM-NATIVE DISPLAY ENGINE
Direct framebuffer access via Python Buffer Protocol
Implements cellular automata metrics for system health monitoring
"""

import numpy as np
import mmap
import os
import struct
from typing import Tuple, Optional
import zlib
import hashlib

class SubstrateDisplay:
    def __init__(self, width: int = 1024, height: int = 768):
        self.width = width
        self.height = height
        self.bpp = 4  # 32-bit RGBA

        # Initialize framebuffer
        self.fb_device = self._init_framebuffer()
        self.vram = self._map_framebuffer()

        # Font system for plain text rendering
        self.font_atlas = self._create_font_atlas()

        # Cellular automata metrics
        self.complexity_history = []
        self.stability_threshold = 0.1

        print("🎯 SUBSTRATE DISPLAY ENGINE INITIALIZED")
        print(f"   VRAM: {width}x{height} ({self.bpp}-channel)")
        print("   Direct framebuffer access: ENABLED")

    def _init_framebuffer(self) -> int:
        """Initialize Linux framebuffer device"""
        try:
            # Try to open framebuffer
            fb = os.open('/dev/fb0', os.O_RDWR)

            # Get screen info (fixed screen info)
            finfo = os.read(fb, 64)  # Read fixed_info struct
            self.actual_width = struct.unpack('I', finfo[0:4])[0]
            self.actual_height = struct.unpack('I', finfo[4:8])[0]

            print(f"   Detected display: {self.actual_width}x{self.actual_height}")
            return fb

        except Exception as e:
            print(f"   Framebuffer unavailable: {e}")
            print("   Falling back to memory-mapped simulation")
            return -1

    def _map_framebuffer(self) -> np.ndarray:
        """Memory-map the framebuffer into NumPy array"""
        if self.fb_device != -1:
            # Real framebuffer mapping
            buffer_size = self.actual_width * self.actual_height * self.bpp
            mapped = mmap.mmap(self.fb_device, buffer_size, mmap.MAP_SHARED, mmap.PROT_WRITE)

            # Create NumPy array view of the mapped memory
            vram = np.frombuffer(mapped, dtype=np.uint8).reshape(
                self.actual_height, self.actual_width, self.bpp
            )
        else:
            # Simulated VRAM for development
            vram = np.zeros((self.height, self.width, self.bpp), dtype=np.uint8)

        return vram

    def _create_font_atlas(self) -> dict:
        """Create bitmap font atlas for plain text rendering"""
        # Simple 8x8 bitmap font (ASCII 32-126)
        font = {}

        # Basic alphanumeric characters as 8x8 patterns
        for char_code in range(32, 127):
            char = chr(char_code)
            # Create simple pattern (in real implementation, load from font file)
            pattern = np.zeros((8, 8), dtype=np.uint8)

            # Simple representation - real implementation would use proper bitmaps
            if char.isalnum():
                pattern[2:6, 2:6] = 255  # Simple block for alphanumeric
            elif char == ' ':
                pattern[:, :] = 0  # Space is empty
            else:
                pattern[3:5, 3:5] = 255  # Dot for punctuation

            font[char] = pattern

        return font

    def render_text(self, text: str, x: int, y: int,
                   color: Tuple[int, int, int] = (255, 255, 255)):
        """Render plain text directly to VRAM"""
        current_x = x

        for char in text:
            if char in self.font_atlas:
                char_pattern = self.font_atlas[char]

                # Blit character to VRAM
                for row in range(8):
                    for col in range(8):
                        if char_pattern[row, col] > 0:
                            px_x, px_y = current_x + col, y + row
                            if (0 <= px_x < self.width and
                                0 <= px_y < self.height):
                                self.vram[px_y, px_x, 0:3] = color
                                self.vram[px_y, px_x, 3] = 255  # Alpha

                current_x += 8  # Move to next character position

    def calculate_complexity(self) -> float:
        """Calculate Lempel-Ziv complexity of current VRAM state"""
        # Convert VRAM to binary pattern for complexity analysis
        binary_pattern = (self.vram[:, :, 0] > 128).astype(np.uint8)

        # Flatten and convert to string for LZ complexity
        flat_pattern = binary_pattern.flatten()
        pattern_string = ''.join(str(bit) for bit in flat_pattern[:1000])  # Sample

        # Simple Lempel-Ziv complexity approximation
        complexity = self._lempel_ziv_complexity(pattern_string)
        normalized_complexity = complexity / len(pattern_string)

        # Track complexity history for stability analysis
        self.complexity_history.append(normalized_complexity)
        if len(self.complexity_history) > 100:
            self.complexity_history.pop(0)

        return normalized_complexity

    def _lempel_ziv_complexity(self, sequence: str) -> int:
        """Calculate Lempel-Ziv complexity of a binary sequence"""
        i, n = 0, 1
        sub_strings = set()

        while i + n <= len(sequence):
            sub_str = sequence[i:i + n]
            if sub_str not in sub_strings:
                sub_strings.add(sub_str)
                i += n
                n = 1
            else:
                n += 1

        return len(sub_strings)

    def detect_instability(self) -> bool:
        """Detect system instability using cellular automata metrics"""
        if len(self.complexity_history) < 10:
            return False

        # Check for homogeneity (Class 1 behavior)
        recent_complexity = np.mean(self.complexity_history[-10:])
        if recent_complexity < self.stability_threshold:
            return True  # System has entered low-complexity state

        # Check for oscillation
        if len(self.complexity_history) >= 20:
            variance = np.var(self.complexity_history[-20:])
            if variance < 0.001:  # Too stable - might be stuck
                return True

        return False

    def get_vram_hash(self) -> str:
        """Generate perceptual hash of current VRAM state"""
        # Simple hash of downsampled VRAM for deduplication
        downsampled = self.vram[::4, ::4, 0]  # Sample every 4th pixel
        return hashlib.sha256(downsampled.tobytes()).hexdigest()[:16]

    def compress_vram_state(self) -> bytes:
        """Compress VRAM state for storage"""
        return zlib.compress(self.vram.tobytes())

    def decompress_vram_state(self, compressed_data: bytes):
        """Decompress and restore VRAM state"""
        decompressed = zlib.decompress(compressed_data)
        restored_vram = np.frombuffer(decompressed, dtype=np.uint8)
        restored_vram = restored_vram.reshape(self.height, self.width, self.bpp)
        self.vram[:, :, :] = restored_vram

    def clear_region(self, x: int, y: int, width: int, height: int):
        """Clear rectangular region in VRAM"""
        x_end = min(x + width, self.width)
        y_end = min(y + height, self.height)
        self.vram[y:y_end, x:x_end] = 0
