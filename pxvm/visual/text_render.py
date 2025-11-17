#!/usr/bin/env python3
"""
pxvm/visual/text_render.py

Pixel-native text rendering using bitmap font atlas.
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from PIL import Image

class FontAtlas:
    def __init__(self, atlas_png: Path, atlas_json: Path):
        self.atlas_img = np.array(Image.open(atlas_png).convert("L"))
        with open(atlas_json, "r") as f:
            metadata = json.load(f)
        self.glyph_w = metadata["glyph_width"]
        self.glyph_h = metadata["glyph_height"]
        self.cols = metadata["cols"]
        self.mapping = metadata["mapping"]

    def get_glyph(self, char: str) -> np.ndarray:
        if char not in self.mapping:
            char = "?" # Fallback for unknown characters
        glyph_index = self.mapping[char]
        gx = (glyph_index % self.cols) * self.glyph_w
        gy = (glyph_index // self.cols) * self.glyph_h
        return self.atlas_img[gy:gy+self.glyph_h, gx:gx+self.glyph_w]

def draw_text(img: np.ndarray, font: FontAtlas, text: str, x: int, y: int, color: tuple = (255, 255, 255)):
    """Draws text onto an image."""
    for i, char in enumerate(text):
        glyph = font.get_glyph(char)
        if y < img.shape[0] and x + i * font.glyph_w < img.shape[1]:
            # This is a simplified blit that doesn't handle alpha blending
            # For RGBA, a more sophisticated approach would be needed.
            mask = glyph > 0
            img[y:y+font.glyph_h, x + i * font.glyph_w:x + (i + 1) * font.glyph_w][mask] = color[:3]

def create_text_image(text: str, font: FontAtlas, padding: int = 10, bg_color: tuple = (0, 0, 0)) -> Image.Image:
    """Creates an image with the given text."""
    lines = text.split('\n')
    max_len = max(len(line) for line in lines)
    img_w = max_len * font.glyph_w + 2 * padding
    img_h = len(lines) * font.glyph_h + 2 * padding

    img = np.full((img_h, img_w, 3), bg_color, dtype=np.uint8)

    for i, line in enumerate(lines):
        draw_text(img, font, line, padding, padding + i * font.glyph_h)

    return Image.fromarray(img)

if __name__ == "__main__":
    # Demo
    root = Path(__file__).resolve().parents[2]
    font_png = root / "fonts" / "ascii_16x16.png"
    font_json = root / "fonts" / "ascii_16x16.json"

    if not font_png.exists() or not font_json.exists():
        print("Font atlas not found. Please run `python3 -m pxvm.visual.font_atlas`")
    else:
        font = FontAtlas(font_png, font_json)
        text_img = create_text_image("Hello, pxOS!", font, padding=20, bg_color=(20, 20, 30))
        text_img.save("fonts/demo_text.png")
        print("Demo text image saved to fonts/demo_text.png")
