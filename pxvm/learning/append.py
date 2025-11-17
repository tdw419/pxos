#!/usr/bin/env python3
"""
pxvm/learning/append.py

Text-to-pixel rendering for self-expanding networks.

This module provides utilities to convert text into pixel rows
that can be appended to growing neural networks.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional
import os


def render_text_to_rows(
    text: str,
    width: int,
    max_lines: int = 100,
    font_size: int = 12,
    bg_color: tuple = (0, 0, 0, 255),
    fg_color: tuple = (255, 255, 255, 255)
) -> Optional[np.ndarray]:
    """
    Render text as pixel rows for appending to networks.

    Args:
        text: Text to render
        width: Width in pixels
        max_lines: Maximum number of lines to render
        font_size: Font size in points
        bg_color: Background RGBA color
        fg_color: Foreground (text) RGBA color

    Returns:
        RGBA numpy array of rendered text, or None on error
    """
    try:
        # Calculate line height based on font size
        line_height = font_size + 4

        # Split text into lines
        lines = text.split('\n')[:max_lines]

        if not lines:
            return None

        # Calculate required height
        height = len(lines) * line_height

        # Create image
        img = Image.new('RGBA', (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        # Try to load a monospace font
        font = None
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/System/Library/Fonts/Courier.dfont",
            "C:\\Windows\\Fonts\\consola.ttf"
        ]

        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue

        if font is None:
            # Fallback to default font
            font = ImageFont.load_default()

        # Render each line
        y = 0
        for line in lines:
            draw.text((4, y), line, fill=fg_color, font=font)
            y += line_height

        # Convert to numpy array
        return np.array(img)

    except Exception as e:
        print(f"Error rendering text: {e}")
        return None


def extract_text_from_pixels(pixel_array: np.ndarray) -> str:
    """
    Extract text from pixel rows (OCR-lite).

    This is a simplified version for the prototype.
    In production, this would use proper OCR or embedded metadata.

    Args:
        pixel_array: RGBA numpy array

    Returns:
        Extracted text (or placeholder for prototype)
    """
    # For prototype: return metadata about the pixels
    height, width = pixel_array.shape[:2]
    return f"[Pixel data: {width}x{height}]"


if __name__ == "__main__":
    # Test the rendering
    test_text = """PIXEL NETWORK TEST
==================

This is a test of the text-to-pixel rendering system.
- Each line becomes pixel rows
- Text is monospaced
- Can be appended to networks

Next line...
"""

    pixels = render_text_to_rows(test_text, width=800, max_lines=20)

    if pixels is not None:
        img = Image.fromarray(pixels.astype(np.uint8), 'RGBA')
        img.save('test_render.png')
        print(f"✅ Test render complete: {pixels.shape}")
        print(f"   Saved to: test_render.png")
    else:
        print("❌ Rendering failed")
