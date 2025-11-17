#!/usr/bin/env python3
"""
pxvm/visual/text_render.py

Pixel-native text rendering using bitmap font atlas.

This module provides the FontAtlas class for rendering text into pixel images
using pre-generated bitmap font atlases.

Key features:
- Fast text rendering via bitmap blitting
- Supports both grayscale and RGBA images
- Handles word wrapping and multi-line text
- Foundation for pixel-native UI and debugging

Usage:
    from pxvm.visual.text_render import FontAtlas
    import numpy as np

    # Load font
    font = FontAtlas("fonts/ascii_16x16.png", "fonts/ascii_16x16.json")

    # Create canvas
    img = np.zeros((256, 512, 4), dtype=np.uint8)  # RGBA
    img[:, :, 3] = 255  # Opaque

    # Render text
    font.draw_text(img, "Hello, pxOS!", x=10, y=10)

    # Save
    from PIL import Image
    Image.fromarray(img).save("output.png")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


class FontAtlas:
    """
    Bitmap font atlas for pixel-native text rendering.

    A font atlas is a single PNG containing all glyphs in a grid,
    with metadata mapping characters to their positions.
    """

    def __init__(self, png_path: Path | str, json_path: Path | str):
        """
        Load bitmap font atlas.

        Args:
            png_path: Path to atlas PNG file
            json_path: Path to metadata JSON file
        """
        # Load atlas image (grayscale)
        atlas_img = Image.open(png_path).convert("L")
        self.atlas = np.array(atlas_img, dtype=np.uint8)

        # Load metadata
        with open(json_path, 'r') as f:
            meta = json.load(f)

        self.glyph_w = meta["glyph_width"]
        self.glyph_h = meta["glyph_height"]
        self.cols = meta["cols"]
        self.rows = meta["rows"]
        self.mapping = meta["mapping"]

        # Reverse mapping: char → index
        self.char_to_idx = self.mapping

        # Cache for glyph bitmaps (optional optimization)
        self._glyph_cache = {}

    def get_glyph(self, char: str) -> np.ndarray:
        """
        Extract glyph bitmap for a character.

        Args:
            char: Character to render

        Returns:
            Glyph bitmap as grayscale uint8 array [height, width]
        """
        # Use fallback for unknown characters
        if char not in self.char_to_idx:
            char = '?'
            if char not in self.char_to_idx:
                # Ultimate fallback: return blank glyph
                return np.zeros((self.glyph_h, self.glyph_w), dtype=np.uint8)

        # Check cache
        if char in self._glyph_cache:
            return self._glyph_cache[char]

        # Compute glyph position in atlas
        idx = self.char_to_idx[char]
        col = idx % self.cols
        row = idx // self.cols

        gx = col * self.glyph_w
        gy = row * self.glyph_h

        # Extract glyph
        glyph = self.atlas[gy:gy+self.glyph_h, gx:gx+self.glyph_w].copy()

        # Cache it
        self._glyph_cache[char] = glyph

        return glyph

    def draw_text(
        self,
        img: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: Optional[Tuple[int, int, int]] = None,
        wrap_width: Optional[int] = None
    ) -> None:
        """
        Render text onto an image.

        Args:
            img: Target image (grayscale [H,W] or RGBA [H,W,4])
            text: Text string to render
            x: X coordinate (left edge)
            y: Y coordinate (top edge)
            color: RGB color tuple (default: white). Ignored for grayscale.
            wrap_width: Optional wrap width in pixels (None = no wrap)
        """
        if color is None:
            color = (255, 255, 255)

        cursor_x = x
        cursor_y = y

        is_rgba = (len(img.shape) == 3 and img.shape[2] == 4)
        is_grayscale = (len(img.shape) == 2)

        for char in text:
            # Handle newlines
            if char == '\n':
                cursor_x = x
                cursor_y += self.glyph_h
                continue

            # Handle wrapping
            if wrap_width is not None and cursor_x + self.glyph_w > x + wrap_width:
                cursor_x = x
                cursor_y += self.glyph_h

            # Get glyph bitmap
            glyph = self.get_glyph(char)

            # Blit glyph
            self._blit_glyph(
                img, glyph, cursor_x, cursor_y, color, is_rgba, is_grayscale
            )

            # Advance cursor
            cursor_x += self.glyph_w

    def _blit_glyph(
        self,
        img: np.ndarray,
        glyph: np.ndarray,
        x: int,
        y: int,
        color: Tuple[int, int, int],
        is_rgba: bool,
        is_grayscale: bool
    ) -> None:
        """
        Blit a glyph onto the image.

        Args:
            img: Target image
            glyph: Glyph bitmap [glyph_h, glyph_w]
            x: Destination X
            y: Destination Y
            color: RGB color
            is_rgba: True if img is RGBA
            is_grayscale: True if img is grayscale
        """
        img_h, img_w = img.shape[:2]

        for dy in range(self.glyph_h):
            for dx in range(self.glyph_w):
                dest_x = x + dx
                dest_y = y + dy

                # Bounds check
                if dest_x < 0 or dest_x >= img_w:
                    continue
                if dest_y < 0 or dest_y >= img_h:
                    continue

                # Get glyph pixel intensity (0-255)
                intensity = glyph[dy, dx]

                if intensity == 0:
                    continue  # Skip transparent pixels

                # Normalize to [0, 1]
                alpha = intensity / 255.0

                if is_rgba:
                    # Blend with background using alpha
                    for c in range(3):  # RGB channels
                        bg = img[dest_y, dest_x, c]
                        fg = color[c]
                        img[dest_y, dest_x, c] = int(bg * (1 - alpha) + fg * alpha)
                    # Keep existing alpha
                elif is_grayscale:
                    # Direct write for grayscale
                    img[dest_y, dest_x] = intensity

    def measure_text(self, text: str) -> Tuple[int, int]:
        """
        Measure text dimensions in pixels.

        Args:
            text: Text to measure

        Returns:
            (width, height) in pixels
        """
        lines = text.split('\n')
        max_width = max(len(line) for line in lines) * self.glyph_w
        height = len(lines) * self.glyph_h
        return max_width, height

    def draw_multiline(
        self,
        img: np.ndarray,
        text: str,
        x: int,
        y: int,
        line_spacing: int = 0,
        color: Optional[Tuple[int, int, int]] = None
    ) -> None:
        """
        Render multi-line text.

        Args:
            img: Target image
            text: Text with newlines
            x: X coordinate
            y: Y coordinate (top of first line)
            line_spacing: Extra pixels between lines
            color: RGB color (default: white)
        """
        cursor_y = y
        for line in text.split('\n'):
            self.draw_text(img, line, x, cursor_y, color=color)
            cursor_y += self.glyph_h + line_spacing


def create_text_image(
    text: str,
    font_atlas: FontAtlas,
    padding: int = 10,
    background: Tuple[int, int, int, int] = (0, 0, 0, 255),
    text_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Create a new RGBA image containing rendered text.

    Args:
        text: Text to render
        font_atlas: Font atlas to use
        padding: Padding around text in pixels
        background: RGBA background color
        text_color: RGB text color

    Returns:
        RGBA image as uint8 array [H, W, 4]
    """
    # Measure text
    text_w, text_h = font_atlas.measure_text(text)

    # Create image with padding
    img_w = text_w + 2 * padding
    img_h = text_h + 2 * padding

    img = np.zeros((img_h, img_w, 4), dtype=np.uint8)

    # Fill background
    img[:, :] = background

    # Render text
    font_atlas.draw_multiline(img, text, x=padding, y=padding, color=text_color)

    return img


def main():
    """Demo of text rendering."""
    from pathlib import Path

    # Find font atlas
    root = Path(__file__).resolve().parents[2]
    font_png = root / "fonts" / "ascii_16x16.png"
    font_json = root / "fonts" / "ascii_16x16.json"

    if not font_png.exists():
        print("ERROR: Font atlas not found. Run:")
        print("  python3 -m pxvm.visual.font_atlas")
        return 1

    print("=" * 70)
    print(" TEXT RENDERING DEMO")
    print("=" * 70)
    print()

    # Load font
    print(f"Loading font: {font_png.name}")
    font = FontAtlas(font_png, font_json)
    print(f"  Glyph size: {font.glyph_w}×{font.glyph_h}")
    print()

    # Create demo text
    demo_text = """Hello, pxOS!

This is pixel-native text rendering.

All glyphs come from a bitmap atlas.
Future: OP_BLIT_GLYPH in the VM!"""

    print("Demo text:")
    print("-" * 70)
    print(demo_text)
    print("-" * 70)
    print()

    # Render to image
    print("Rendering text to image...")
    img = create_text_image(
        demo_text,
        font,
        padding=20,
        background=(30, 30, 40, 255),
        text_color=(220, 220, 255)
    )

    print(f"  Image size: {img.shape[1]}×{img.shape[0]} RGBA")
    print()

    # Save
    output_path = root / "fonts" / "demo_text.png"
    Image.fromarray(img).save(output_path)

    print(f"Saved: {output_path}")
    print()

    print("=" * 70)
    print(" DEMO COMPLETE")
    print("=" * 70)
    print()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
