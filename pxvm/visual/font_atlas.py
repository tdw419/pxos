#!/usr/bin/env python3
"""
pxvm/visual/font_atlas.py

Generate bitmap font atlas for pixel-native text rendering.

A font atlas is a single PNG containing all glyphs arranged in a grid,
plus metadata mapping characters to their positions.

This enables:
- Pixel-native text rendering in the VM
- Visual output from programs (PNG → text → PNG)
- Debug overlays, status messages, console output
- Future: OP_BLIT_GLYPH opcode for self-rendering programs

Usage:
    python3 -m pxvm.visual.font_atlas
    python3 -m pxvm.visual.font_atlas --font-size 12 --output fonts/custom.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def find_system_font() -> str:
    """
    Find a monospace system font.

    Returns:
        Path to a monospace TTF font
    """
    # Common monospace fonts across systems
    candidates = [
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        # macOS
        "/System/Library/Fonts/Monaco.ttf",
        "/Library/Fonts/Courier New.ttf",
        # Windows (if running under WSL)
        "/mnt/c/Windows/Fonts/consola.ttf",
    ]

    for font_path in candidates:
        if Path(font_path).exists():
            return font_path

    # Fallback: try to load default
    raise FileNotFoundError(
        "No monospace font found. Please install DejaVu Sans Mono or specify --font-path"
    )


def generate_ascii_atlas(
    output_png: Path,
    output_json: Path,
    font_path: str = None,
    font_size: int = 14,
    glyph_width: int = 16,
    glyph_height: int = 16,
    grid_cols: int = 16,
    grid_rows: int = 16,
) -> Tuple[Image.Image, Dict]:
    """
    Generate bitmap font atlas for ASCII characters.

    Args:
        output_png: Path to save atlas PNG
        output_json: Path to save metadata JSON
        font_path: Path to TTF font file (or None to auto-detect)
        font_size: Font size in points
        glyph_width: Width of each glyph cell in pixels
        glyph_height: Height of each glyph cell in pixels
        grid_cols: Number of columns in atlas grid
        grid_rows: Number of rows in atlas grid

    Returns:
        (atlas_image, metadata_dict)
    """
    print("=" * 70)
    print(" GENERATING BITMAP FONT ATLAS")
    print("=" * 70)
    print()

    # Find or use specified font
    if font_path is None:
        font_path = find_system_font()

    print(f"Font: {font_path}")
    print(f"Font size: {font_size}pt")
    print(f"Glyph size: {glyph_width}×{glyph_height}px")
    print(f"Grid: {grid_cols}×{grid_rows} cells")
    print()

    # Load font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"ERROR: Could not load font: {e}")
        print("Falling back to default font (may look bad)")
        font = ImageFont.load_default()

    # Create atlas image (grayscale)
    atlas_width = grid_cols * glyph_width
    atlas_height = grid_rows * glyph_height
    atlas = Image.new("L", (atlas_width, atlas_height), 0)
    draw = ImageDraw.Draw(atlas)

    # Generate character set (ASCII printable + some extended)
    # Start at space (32), cover 256 cells
    num_glyphs = grid_cols * grid_rows
    chars = []
    for i in range(num_glyphs):
        char_code = 32 + i  # Start at space
        if char_code < 256:
            chars.append(chr(char_code))
        else:
            chars.append('?')  # Fallback for extended range

    # Draw each character into its grid cell
    mapping = {}

    print("Rendering glyphs...")
    for idx, ch in enumerate(chars):
        # Calculate grid position
        col = idx % grid_cols
        row = idx // grid_cols

        cx = col * glyph_width
        cy = row * glyph_height

        # Draw character centered in cell
        # Note: text position is top-left, we add small offset for centering
        draw.text((cx + 2, cy), ch, font=font, fill=255)

        # Map character to its index
        mapping[ch] = idx

    print(f"  Rendered {len(chars)} glyphs")
    print()

    # Save atlas image
    print(f"Saving atlas: {output_png}")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(output_png)

    atlas_size = output_png.stat().st_size
    print(f"  File size: {atlas_size:,} bytes")
    print()

    # Create metadata
    metadata = {
        "font_path": str(font_path),
        "font_size": font_size,
        "glyph_width": glyph_width,
        "glyph_height": glyph_height,
        "cols": grid_cols,
        "rows": grid_rows,
        "num_glyphs": num_glyphs,
        "mapping": mapping,
        "format": "grayscale_8bit",
        "version": "1.0"
    }

    # Save metadata
    print(f"Saving metadata: {output_json}")
    with open(output_json, 'w') as f:
        json.dump(metadata, f, indent=2)

    meta_size = output_json.stat().st_size
    print(f"  File size: {meta_size:,} bytes")
    print()

    print("=" * 70)
    print(" FONT ATLAS GENERATION COMPLETE")
    print("=" * 70)
    print()
    print("Usage:")
    print("  from pxvm.visual.text_render import FontAtlas")
    print(f"  font = FontAtlas('{output_png}', '{output_json}')")
    print("  font.draw_text(img, 'Hello, World!', x=10, y=10)")
    print()

    return atlas, metadata


def main():
    """CLI interface for font atlas generation."""
    parser = argparse.ArgumentParser(
        description="Generate bitmap font atlas for pixel-native text rendering"
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("fonts/ascii_16x16.png"),
        help="Path to save atlas PNG (default: fonts/ascii_16x16.png)"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("fonts/ascii_16x16.json"),
        help="Path to save metadata JSON (default: fonts/ascii_16x16.json)"
    )
    parser.add_argument(
        "--font-path",
        type=str,
        default=None,
        help="Path to TTF font file (auto-detect if not specified)"
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=14,
        help="Font size in points (default: 14)"
    )
    parser.add_argument(
        "--glyph-width",
        type=int,
        default=16,
        help="Glyph cell width in pixels (default: 16)"
    )
    parser.add_argument(
        "--glyph-height",
        type=int,
        default=16,
        help="Glyph cell height in pixels (default: 16)"
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    root = Path(__file__).resolve().parents[2]
    output_png = root / args.output_png
    output_json = root / args.output_json

    # Generate atlas
    generate_ascii_atlas(
        output_png=output_png,
        output_json=output_json,
        font_path=args.font_path,
        font_size=args.font_size,
        glyph_width=args.glyph_width,
        glyph_height=args.glyph_height,
    )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
