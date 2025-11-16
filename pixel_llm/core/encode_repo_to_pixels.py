#!/usr/bin/env python3
"""
Encode Repository to Pixels

This script represents a FUNDAMENTAL PARADIGM SHIFT in pxOS:

BEFORE: Python files are source of truth, pixels are experiments
AFTER:  Pixels are source of truth, Python is just a runtime

This encoder:
1. Walks the pxOS repository
2. Reads all Python source files
3. Encodes them as pixel images (.pxi) via PixelFS
4. Generates a manifest mapping module names to pixel images

Once encoded, the pixels become the canonical source.
Python files become generated artifacts or views.

Philosophy:
"Code that lives in pixels is substrate-native.
 Code that lives in text files is legacy."
"""

from pathlib import Path
import json
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pixelfs import PixelFS


# Paths
ROOT = Path(__file__).resolve().parents[2]  # /home/user/pxos
PIXEL_ROOT = ROOT / "pixel_llm" / "pixel_src_storage"
MANIFEST_PATH = ROOT / "pixel_llm" / "pixel_manifest.json"

# What to encode
ENCODE_ROOTS = [
    ROOT / "pixel_llm",
]

# What to skip
SKIP_PATTERNS = [
    "pixel_src_storage",  # don't encode the pixel storage itself
    "__pycache__",
    ".pytest_cache",
    "*.pyc",
    ".git",
]


def should_skip(path: Path) -> bool:
    """Check if path should be skipped"""
    path_str = str(path)
    for pattern in SKIP_PATTERNS:
        if pattern in path_str:
            return True
    return False


def module_name_from_path(path: Path, root: Path) -> str:
    """
    Convert file path to Python module name.

    Example:
      /home/user/pxos/pixel_llm/core/pixelfs.py
      → pixel_llm.core.pixelfs
    """
    try:
        rel = path.relative_to(root.parent)
        parts = list(rel.with_suffix("").parts)
        return ".".join(parts)
    except ValueError:
        # Fallback if path is not relative to root
        rel = path.relative_to(ROOT)
        parts = list(rel.with_suffix("").parts)
        return ".".join(parts)


def pixel_key_from_module(module_name: str) -> str:
    """
    Convert module name to pixel file key.

    Example:
      pixel_llm.core.pixelfs → src_pixel_llm_core_pixelfs.pxi
    """
    safe_name = module_name.replace(".", "_")
    return f"src_{safe_name}.pxi"


def main():
    print("=" * 70)
    print("ENCODING REPOSITORY TO PIXELS")
    print("=" * 70)
    print()
    print("🔄 Paradigm shift in progress...")
    print("   From: Python files = source of truth")
    print("   To:   Pixels = source of truth")
    print()

    # Initialize PixelFS
    fs = PixelFS()
    manifest = {}

    total_files = 0
    total_bytes = 0
    errors = []

    # Walk each root
    for encode_root in ENCODE_ROOTS:
        if not encode_root.exists():
            print(f"⚠️  Skipping non-existent root: {encode_root}")
            continue

        print(f"📁 Encoding: {encode_root.relative_to(ROOT)}")

        for py_file in encode_root.rglob("*.py"):
            if should_skip(py_file):
                continue

            try:
                # Read source
                data = py_file.read_bytes()

                # Generate module name
                mod_name = module_name_from_path(py_file, encode_root)

                # Generate pixel key
                pixel_key = pixel_key_from_module(mod_name)
                pixel_path = PIXEL_ROOT / pixel_key

                # Write to PixelFS
                fs.write(str(pixel_path), data)

                # Update manifest
                manifest[mod_name] = {
                    "pixel_key": pixel_key,
                    "pixel_path": str(pixel_path.relative_to(ROOT)),
                    "original_path": str(py_file.relative_to(ROOT)),
                    "size_bytes": len(data),
                }

                total_files += 1
                total_bytes += len(data)

                print(f"  ✅ {mod_name:<50} → {pixel_key}")

            except Exception as e:
                errors.append((str(py_file), str(e)))
                print(f"  ❌ {py_file.name}: {e}")

    print()

    # Write manifest
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    # Summary
    print("=" * 70)
    print("ENCODING COMPLETE")
    print("=" * 70)
    print()
    print(f"📊 Statistics:")
    print(f"  Total modules: {total_files}")
    print(f"  Total bytes:   {total_bytes:,}")
    print(f"  Manifest:      {MANIFEST_PATH.relative_to(ROOT)}")
    print()

    if errors:
        print(f"⚠️  Errors: {len(errors)}")
        for path, error in errors:
            print(f"    {path}: {error}")
        print()

    print("🎯 WHAT THIS MEANS:")
    print()
    print("  ✅ All Python source code now exists as pixel images")
    print("  ✅ Pixels are the canonical source")
    print("  ✅ Python files are now just 'views' of pixel data")
    print("  ✅ Ready to boot from pixels")
    print()
    print("Next step: python3 pxos_boot.py")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
