"""
Direct VRAM Input - keystrokes directly manipulate pixel memory
This is where your 'x' becomes pixels in VRAM!
"""

import numpy as np
from enum import Enum

class VRAMInputMode(Enum):
    DIRECT_PIXELS = 1    # Keys directly set pixel colors
    PATTERN_MODE = 2     # Keys trigger pattern recognition
    KERNEL_CODE = 3      # Keys build pixel-based programs

class VRAMInputSystem:
    def __init__(self, vram_width: int = 1024, vram_height: int = 768):
        self.vram = np.zeros((vram_height, vram_width, 3), dtype=np.uint8)
        self.cursor_x = 0
        self.cursor_y = 0
        self.input_mode = VRAMInputMode.PATTERN_MODE

        # Input history for pattern detection
        self.input_history = []

        # Character to pixel patterns
        self.char_pixels = self._init_char_mappings()

    def _init_char_mappings(self) -> dict:
        """Map characters to pixel patterns (8x8 bitmaps)"""
        return {
            'x': self._create_char_pixels('x', (255, 0, 0)),      # Red 'x'
            'y': self._create_char_pixels('y', (0, 255, 0)),      # Green 'y'
            'z': self._create_char_pixels('z', (0, 0, 255)),      # Blue 'z'
            'a': self._create_char_pixels('a', (255, 255, 0)),    # Yellow 'a'
            'b': self._create_char_pixels('b', (255, 0, 255)),    # Magenta 'b'
            'c': self._create_char_pixels('c', (0, 255, 255)),    # Cyan 'c'
            ' ': self._create_char_pixels(' ', (0, 0, 0)),        # Black space
        }

    def _create_char_pixels(self, char: str, color: tuple) -> np.ndarray:
        """Create 8x8 pixel pattern for a character"""
        patterns = {
            'x': [
                [1, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 1],
            ],
            'y': [
                [1, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
            ],
            'z': [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            'a': [
                [0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
            ],
            'b': [
                [1, 1, 1, 1, 1, 1, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 0],
            ],
            'c': [
                [0, 0, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1],
                [0, 0, 1, 1, 1, 1, 1, 0],
            ],
            ' ': [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        }

        char_pattern = patterns.get(char, patterns[' '])
        pixel_data = np.zeros((8, 8, 3), dtype=np.uint8)

        for y in range(8):
            for x in range(8):
                if char_pattern[y][x]:
                    pixel_data[y, x] = color

        return pixel_data

    def handle_keypress(self, key: str):
        """Process a keypress and update VRAM directly"""
        if key in self.char_pixels:
            # Add to history
            self.input_history.append(key)

            if self.input_mode == VRAMInputMode.DIRECT_PIXELS:
                self._draw_char_direct(key)
            elif self.input_mode == VRAMInputMode.PATTERN_MODE:
                self._draw_char_direct(key)  # Still draw it

            self._advance_cursor()

    def _draw_char_direct(self, char: str):
        """Draw character directly to VRAM at cursor position"""
        char_pixels = self.char_pixels[char]
        x_start = self.cursor_x * 8
        y_start = self.cursor_y * 8

        # Copy character pixels to VRAM
        if x_start + 8 <= self.vram.shape[1] and y_start + 8 <= self.vram.shape[0]:
            self.vram[y_start:y_start+8, x_start:x_start+8] = char_pixels

    def _advance_cursor(self):
        """Move cursor to next position"""
        self.cursor_x += 1
        if self.cursor_x >= (self.vram.shape[1] // 8):
            self.cursor_x = 0
            self.cursor_y += 1
            if self.cursor_y >= (self.vram.shape[0] // 8):
                self.cursor_y = 0  # Wrap around

    def get_vram(self) -> np.ndarray:
        """Get current VRAM state for display"""
        return self.vram

    def get_input_sequence(self) -> str:
        """Get the input sequence as a string"""
        return ''.join(self.input_history)

    def clear_vram(self):
        """Clear VRAM"""
        self.vram = np.zeros_like(self.vram)
        self.cursor_x = 0
        self.cursor_y = 0
        self.input_history = []
