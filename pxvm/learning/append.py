#!/usr/bin/env python3
"""
pxvm/learning/append.py

Simplified text-to-pixel rendering for the learning system.
This is a lightweight implementation for the LM Studio bridge.
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Optional


def render_text_to_rows(
    text: str,
    width: int = 1024,
    max_lines: int = 100,
    font_size: int = 12,
    bg_color: tuple = (0, 0, 0, 255),
    text_color: tuple = (255, 255, 255, 255)
) -> Optional[np.ndarray]:
    """
    Render text as pixel rows for append-only learning.

    Args:
        text: Text to render
        width: Image width in pixels
        max_lines: Maximum number of text lines
        font_size: Font size
        bg_color: Background RGBA color
        text_color: Text RGBA color

    Returns:
        RGBA numpy array or None on error
    """
    try:
        # Calculate height based on text lines
        lines = text.split('\n')[:max_lines]
        line_height = font_size + 4
        height = len(lines) * line_height

        # Create image
        img = Image.new('RGBA', (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        # Load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
            except:
                font = ImageFont.load_default()

        # Render each line
        y = 0
        for line in lines:
            draw.text((2, y), line, fill=text_color, font=font)
            y += line_height

        return np.array(img)

    except Exception as e:
        print(f"Error rendering text: {e}")
        return None


def extract_text_from_pixels(
    pixel_array: np.ndarray,
    row_start: int = 0,
    row_end: Optional[int] = None
) -> str:
    """
    Extract text from pixel array (OCR-like functionality).

    This is a placeholder for future OCR implementation.
    For now, it returns a representation of the pixel data.

    Args:
        pixel_array: RGBA numpy array
        row_start: Starting row
        row_end: Ending row (None = all rows)

    Returns:
        Extracted text (or placeholder)
    """
    # Future: Implement actual OCR
    # For now, return metadata
    height, width = pixel_array.shape[:2]
    end = row_end if row_end else height

    return f"[Pixel data: rows {row_start}-{end}, width {width}]"


if __name__ == "__main__":
    # Test rendering
    test_text = """pxOS Test Render
==================

This is a test of the pixel rendering system.
Each line becomes part of the self-expanding network.

- Feature 1
- Feature 2
- Feature 3
"""

    pixels = render_text_to_rows(test_text, width=800, max_lines=20)

    if pixels is not None:
        img = Image.fromarray(pixels.astype(np.uint8), 'RGBA')
        img.save("/tmp/test_render.png")
        print(f"✅ Test render saved: /tmp/test_render.png")
        print(f"   Size: {pixels.shape}")
    else:
        print("❌ Rendering failed")
