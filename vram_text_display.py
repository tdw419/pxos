"""
VRAM Text Display System
Render LLM output as pixel text directly in VRAM
"""

import numpy as np
from typing import List, Dict, Any

class VRAMTextDisplay:
    def __init__(self, substrate, font_size=8):
        self.substrate = substrate
        self.font_size = font_size

        # Text display regions in VRAM
        self.llm_output_region = (100, 3500, 1024, 512)  # Large area for LLM text
        self.status_region = (3500, 0, 512, 256)         # Status messages
        self.debug_region = (3500, 300, 512, 256)        # Debug info

        # Simple 8x8 font (each character is 8x8 pixels)
        self.font = self._create_simple_font()

        print("📝 VRAM Text Display Initialized")
        print("   LLM output will render as pixel text in VRAM")

    def _create_simple_font(self) -> Dict[str, np.ndarray]:
        """Create a simple 8x8 pixel font"""
        font = {}

        # Basic characters: each is 8x8 array where 1=on, 0=off
        font['A'] = np.array([
            [0,0,1,1,1,0,0,0],
            [0,1,0,0,0,1,0,0],
            [1,0,0,0,0,0,1,0],
            [1,0,0,0,0,0,1,0],
            [1,1,1,1,1,1,1,0],
            [1,0,0,0,0,0,1,0],
            [1,0,0,0,0,0,1,0],
            [1,0,0,0,0,0,1,0]
        ])

        font['B'] = np.array([
            [1,1,1,1,1,1,0,0],
            [1,0,0,0,0,0,1,0],
            [1,0,0,0,0,0,1,0],
            [1,1,1,1,1,1,0,0],
            [1,0,0,0,0,0,1,0],
            [1,0,0,0,0,0,1,0],
            [1,0,0,0,0,0,1,0],
            [1,1,1,1,1,1,0,0]
        ])

        # Create basic alphanumeric set
        chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?;:()-_=+'
        for char in chars:
            if char not in font:
                # Create a simple block pattern for missing chars
                font[char] = np.ones((8, 8)) if char != ' ' else np.zeros((8, 8))

        return font

    def render_text_to_vram(self, text: str, x: int, y: int,
                           color: tuple = (1.0, 1.0, 1.0, 1.0),
                           background: tuple = (0.0, 0.0, 0.0, 1.0)):
        """
        Render text directly to VRAM at specified coordinates
        Each character is 8x8 pixels
        """
        current_x, current_y = x, y
        max_width = self.substrate.width - x
        chars_per_line = max_width // 8

        # Split text into lines that fit in VRAM
        lines = self._wrap_text(text, chars_per_line)

        for line in lines:
            for char in line:
                if char in self.font:
                    char_pattern = self.font[char]
                    self._render_char(char_pattern, current_x, current_y, color, background)
                    current_x += 8
                else:
                    # Skip unknown chars
                    current_x += 8

            current_x = x  # Reset to start of line
            current_y += 10  # Move to next line (8px char + 2px spacing)

            # Stop if we run out of vertical space
            if current_y + 8 > self.substrate.height:
                break

    def _render_char(self, char_pattern: np.ndarray, x: int, y: int,
                    color: tuple, background: tuple):
        """Render a single character to VRAM"""
        for row in range(8):
            for col in range(8):
                pixel_x, pixel_y = x + col, y + row
                if (0 <= pixel_x < self.substrate.width and
                    0 <= pixel_y < self.substrate.height):

                    if char_pattern[row, col] == 1:
                        # Character pixel
                        self.substrate.vram[pixel_y, pixel_x] = color
                    else:
                        # Background pixel
                        self.substrate.vram[pixel_y, pixel_x] = background

    def _wrap_text(self, text: str, max_chars: int) -> List[str]:
        """Wrap text to fit within VRAM width"""
        words = text.split(' ')
        lines = []
        current_line = []

        for word in words:
            if len(' '.join(current_line + [word])) <= max_chars:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def clear_text_region(self, x: int, y: int, width: int, height: int):
        """Clear a rectangular region in VRAM"""
        clear_pattern = np.zeros((height, width, 4), dtype=np.float32)
        self.substrate.inject(clear_pattern, x, y)

    def display_llm_response(self, llm_response: str, region: tuple = None):
        """Display LLM response as text in VRAM"""
        if region is None:
            region = self.llm_output_region

        x, y, w, h = region

        # Clear previous output
        self.clear_text_region(x, y, w, h)

        # Render new text
        self.render_text_to_vram(
            f"LLM RESPONSE: {llm_response}",
            x + 8, y + 8,  # Offset from region edge
            color=(0.0, 1.0, 1.0, 1.0),  # Cyan text
            background=(0.1, 0.1, 0.1, 1.0)  # Dark background
        )

        print(f"📝 LLM response displayed in VRAM at ({x},{y})")

    def display_status(self, message: str):
        """Display status message in VRAM status region"""
        x, y, w, h = self.status_region
        self.clear_text_region(x, y, w, h)
        self.render_text_to_vram(
            f"STATUS: {message}",
            x + 8, y + 8,
            color=(1.0, 1.0, 0.0, 1.0),  # Yellow text
            background=(0.0, 0.0, 0.0, 1.0)  # Black background
        )

    def display_debug(self, debug_info: Dict[str, Any]):
        """Display debug information in VRAM"""
        x, y, w, h = self.debug_region
        self.clear_text_region(x, y, w, h)

        debug_text = "DEBUG INFO:\n"
        for key, value in debug_info.items():
            debug_text += f"{key}: {value}\n"

        self.render_text_to_vram(
            debug_text,
            x + 8, y + 8,
            color=(1.0, 0.0, 1.0, 1.0),  # Magenta text
            background=(0.0, 0.0, 0.0, 1.0)
        )
