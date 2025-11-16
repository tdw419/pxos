#!/usr/bin/env python3
"""
Pixel Module Inspector

Provides "eyes" into the pixel execution system.
Shows what code lives in pixel images and how it's stored.

Usage:
    python3 pixel_llm/core/inspect_pixel_module.py <module_name>
    python3 pixel_llm/core/inspect_pixel_module.py pixel_llm.core.pixelfs

Philosophy:
"If you can't see the pixels, you can't trust them.
 This tool makes the invisible visible."
"""

import sys
import json
from pathlib import Path
from typing import Optional

# Bootstrap path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pixel_llm.core.pixelfs import PixelFS
MANIFEST_PATH = ROOT / "pixel_llm" / "pixel_manifest.json"


def load_manifest() -> dict:
    """Load the pixel manifest"""
    if not MANIFEST_PATH.exists():
        print(f"❌ Manifest not found: {MANIFEST_PATH}")
        print()
        print("Run this first:")
        print("  python3 pixel_llm/core/encode_repo_to_pixels.py")
        sys.exit(1)

    return json.loads(MANIFEST_PATH.read_text())


def inspect_module(module_name: str):
    """Inspect a pixel module"""
    print("=" * 70)
    print(f"PIXEL MODULE INSPECTOR: {module_name}")
    print("=" * 70)
    print()

    # Load manifest
    manifest = load_manifest()

    # Check if module exists
    if module_name not in manifest:
        print(f"❌ Module not found in pixel manifest: {module_name}")
        print()
        print("Available modules:")
        for name in sorted(manifest.keys()):
            print(f"  • {name}")
        sys.exit(1)

    entry = manifest[module_name]

    # Module info
    print(f"Module: {module_name}")
    print(f"Pixel key: {entry['pixel_key']}")
    print(f"Pixel path: {entry['pixel_path']}")
    print(f"Original: {entry['original_path']}")
    print(f"Size: {entry['size_bytes']:,} bytes")
    print()

    # Read pixel image
    pixel_path = ROOT / entry['pixel_path']

    if not pixel_path.exists():
        print(f"❌ Pixel image not found: {pixel_path}")
        sys.exit(1)

    print("─" * 70)
    print("PIXEL IMAGE CONTENTS")
    print("─" * 70)
    print()

    fs = PixelFS()
    raw_bytes = fs.read(str(pixel_path))

    print(f"Raw bytes read: {len(raw_bytes):,}")
    print()

    # Show first N bytes as hex
    hex_preview_size = min(64, len(raw_bytes))
    print(f"First {hex_preview_size} bytes (hex):")
    hex_str = raw_bytes[:hex_preview_size].hex()
    for i in range(0, len(hex_str), 32):
        chunk = hex_str[i:i+32]
        # Format as groups of 2
        formatted = ' '.join(chunk[j:j+2] for j in range(0, len(chunk), 2))
        print(f"  {formatted}")
    print()

    # Decode as source
    try:
        source_code = raw_bytes.decode('utf-8')
        print("─" * 70)
        print("DECODED SOURCE (first 30 lines)")
        print("─" * 70)
        print()

        lines = source_code.split('\n')
        for i, line in enumerate(lines[:30], 1):
            print(f"{i:3d} │ {line}")

        if len(lines) > 30:
            print()
            print(f"... ({len(lines) - 30} more lines)")

        print()
        print(f"Total lines: {len(lines)}")

    except UnicodeDecodeError:
        print("⚠️  Cannot decode as UTF-8 (binary data?)")

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


def list_all_modules():
    """List all available pixel modules"""
    print("=" * 70)
    print("ALL PIXEL MODULES")
    print("=" * 70)
    print()

    manifest = load_manifest()

    print(f"Total modules: {len(manifest)}")
    print(f"Manifest: {MANIFEST_PATH.relative_to(ROOT)}")
    print()

    print(f"{'Module Name':<50} {'Size':>12} {'Pixel Key'}")
    print("─" * 70)

    total_bytes = 0
    for module_name in sorted(manifest.keys()):
        entry = manifest[module_name]
        size = entry['size_bytes']
        total_bytes += size
        pixel_key = entry['pixel_key']

        print(f"{module_name:<50} {size:>10,} B  {pixel_key}")

    print()
    print(f"Total: {total_bytes:,} bytes across {len(manifest)} modules")
    print()
    print("=" * 70)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Pixel Module Inspector")
        print()
        print("Usage:")
        print("  python3 pixel_llm/core/inspect_pixel_module.py <module_name>")
        print("  python3 pixel_llm/core/inspect_pixel_module.py --list")
        print()
        print("Examples:")
        print("  python3 pixel_llm/core/inspect_pixel_module.py pixel_llm.core.pixelfs")
        print("  python3 pixel_llm/core/inspect_pixel_module.py pixel_llm.core.gpu_interface")
        print("  python3 pixel_llm/core/inspect_pixel_module.py --list")
        sys.exit(1)

    if sys.argv[1] in ('--list', '-l'):
        list_all_modules()
    else:
        module_name = sys.argv[1]
        inspect_module(module_name)


if __name__ == "__main__":
    main()
