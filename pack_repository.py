#!/usr/bin/env python3
"""
Pack Repository to Pixel Archive

Packs the entire pxOS repository into a single .pxa file.

This creates ONE file containing:
  - All Python source files
  - Configuration files
  - Documentation
  - Everything needed to run pxOS

Output: pxos_repo.pxa (single pixel archive file)

Usage:
    python3 pack_repository.py

Philosophy:
"One file. One cartridge. Everything you need."
"""

import sys
from pathlib import Path

# Bootstrap path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pixel_llm.core.pixel_archive import PixelArchive


def should_skip(path: Path) -> bool:
    """Determine if a file should be skipped"""
    # Skip patterns
    skip_parts = {
        '__pycache__',
        '.pytest_cache',
        '.git',
        'htmlcov',
        'pixel_storage',
        'pixel_llm/data',
        'pixel_llm/pixel_src_storage',  # Don't pack the individual .pxi files
        '.coverage',
    }

    # Check if any part of the path matches skip patterns
    for part in path.parts:
        if part in skip_parts:
            return True

    # Skip .pyc files EXCEPT those in the bytecode/ directory
    if path.suffix in {'.pyc', '.pyo', '.pyd'}:
        # Allow bytecode/ directory .pyc files
        if 'bytecode' not in path.parts:
            return True

    return False


def collect_files(root: Path) -> list:
    """Collect all files to pack"""
    files = []

    # Python files
    for py_file in root.rglob("*.py"):
        if should_skip(py_file):
            continue
        files.append(py_file)

    # WGSL shader files
    for wgsl_file in root.rglob("*.wgsl"):
        if should_skip(wgsl_file):
            continue
        files.append(wgsl_file)

    # Markdown documentation
    for md_file in root.rglob("*.md"):
        if should_skip(md_file):
            continue
        files.append(md_file)

    # JSON configuration
    for json_file in root.rglob("*.json"):
        if should_skip(json_file):
            continue
        # Skip pixel_manifest.json (it's generated)
        if json_file.name == "pixel_manifest.json":
            continue
        files.append(json_file)

    # Text files at root
    for txt_pattern in ["*.txt", "*.cfg", "*.ini", ".gitignore", "LICENSE"]:
        for txt_file in root.glob(txt_pattern):
            if should_skip(txt_file):
                continue
            files.append(txt_file)

    # Bytecode files (.pyc) from bytecode/ directory
    bytecode_dir = root / "bytecode"
    if bytecode_dir.exists():
        for pyc_file in bytecode_dir.rglob("*.pyc"):
            if should_skip(pyc_file):
                continue
            files.append(pyc_file)

    return sorted(files)


def main():
    """Pack the repository"""
    print("=" * 70)
    print("PIXEL ARCHIVE: REPOSITORY PACKER")
    print("=" * 70)
    print()
    print("🔄 Collecting files from repository...")
    print()

    files = collect_files(ROOT)

    print(f"Found {len(files)} files to pack:")
    print()

    # Group by type
    by_ext = {}
    for f in files:
        ext = f.suffix or "no_ext"
        by_ext.setdefault(ext, []).append(f)

    for ext in sorted(by_ext.keys()):
        count = len(by_ext[ext])
        print(f"  {ext:12s} {count:4d} files")

    print()
    print("─" * 70)
    print("PACKING TO PIXEL ARCHIVE")
    print("─" * 70)
    print()

    # Create archive
    archive = PixelArchive()

    total_bytes = 0
    for file_path in files:
        # Calculate relative path for archive
        rel_path = str(file_path.relative_to(ROOT))

        # Add to archive
        archive.add_file_from_disk(file_path, rel_path)

        file_size = file_path.stat().st_size
        total_bytes += file_size

        print(f"  + {rel_path:<60s} {file_size:>10,} bytes")

    print()
    print(f"Total: {total_bytes:,} bytes from {len(files)} files")
    print()

    # Save archive
    output_path = ROOT / "pxos_repo.pxa"

    print("─" * 70)
    print(f"SAVING TO: {output_path.name}")
    print("─" * 70)
    print()

    archive.save(str(output_path))

    print()
    print("=" * 70)
    print("✅ PIXEL ARCHIVE COMPLETE")
    print("=" * 70)
    print()
    print(f"Archive: {output_path}")
    print(f"Files: {len(files)}")
    print(f"Total size: {total_bytes:,} bytes")
    print()
    print("You can now:")
    print(f"  • List contents:  python3 pixel_llm/core/pixel_archive.py list {output_path.name}")
    print(f"  • Extract all:    python3 pixel_llm/core/pixel_archive.py extract {output_path.name} extracted/")
    print(f"  • Read file:      python3 pixel_llm/core/pixel_archive.py read {output_path.name} <path>")
    print()


if __name__ == "__main__":
    main()
