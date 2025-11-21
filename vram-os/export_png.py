#!/usr/bin/env python3
"""Export VRAM state as PNG image for visualization"""

from pathlib import Path
from core.vram_state import VRAMState

def main():
    # Load the bootloader VRAM state
    print("Loading bootloader VRAM state...")
    vram = VRAMState.load_json(Path("bootloader_vram.json"))

    # Export as PNG
    output_png = Path("bootloader_vram.png")
    print(f"Exporting to {output_png}...")

    try:
        vram.export_png(output_png)
        print(f"✓ PNG exported successfully!")
        print(f"\nYou can now:")
        print(f"1. View {output_png} to see the VRAM OS state as an image")
        print(f"2. Load this PNG into GPU VRAM for native execution")
        print(f"3. Zoom in to see individual instruction pixels")
    except ImportError as e:
        print(f"✗ Could not export PNG: {e}")
        print(f"\nTo export PNGs, install Pillow:")
        print(f"  pip install Pillow")

if __name__ == "__main__":
    main()
