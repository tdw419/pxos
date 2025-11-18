#!/usr/bin/env python3
"""
pxvm/visual/font_atlas.py

Generate bitmap font atlas for pixel-native text rendering.
"""
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def main():
    """Generates a default font atlas."""
    font_size = 14
    glyph_w, glyph_h = 16, 16
    cols, rows = 16, 16

    # Find a monospace font
    font_path = None
    font_names = ["DejaVuSansMono.ttf", "Menlo.ttc", "Consolas.ttf", "cour.ttf"]
    for name in font_names:
        try:
            font = ImageFont.truetype(name, font_size)
            font_path = name
            break
        except IOError:
            continue

    if font_path is None:
        print("Could not find a suitable monospace font. Using default.")
        font = ImageFont.load_default()
    else:
        print(f"Using font: {font_path}")

    atlas = Image.new("L", (cols * glyph_w, rows * glyph_h), 0)
    draw = ImageDraw.Draw(atlas)

    mapping = {}
    chars = [chr(i) for i in range(256)]

    for i, char in enumerate(chars):
        cx = (i % cols) * glyph_w
        cy = (i // cols) * glyph_h
        draw.text((cx, cy), char, font=font, fill=255)
        mapping[char] = i

    output_dir = Path("fonts")
    output_dir.mkdir(exist_ok=True)
    atlas.save(output_dir / "ascii_16x16.png")

    metadata = {
        "glyph_width": glyph_w,
        "glyph_height": glyph_h,
        "cols": cols,
        "rows": rows,
        "mapping": mapping,
    }
    with open(output_dir / "ascii_16x16.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Font atlas and metadata saved to {output_dir.resolve()}")

if __name__ == "__main__":
    main()
