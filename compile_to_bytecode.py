#!/usr/bin/env python3
"""
Compile Repository to Bytecode

Compiles all Python source files to .pyc bytecode for storage in pixel archive.

WHY BYTECODE?
  - Faster loading (no compilation overhead)
  - Smaller size (optimized bytecode)
  - Can distribute without source
  - True "execute from pixels" model

WORKFLOW:
  1. Scan repository for .py files
  2. Compile each to .pyc using py_compile
  3. Store in bytecode/ directory structure
  4. Pack into pixel archive with pack_repository.py

OUTPUT:
  bytecode/
    pixel_llm/
      core/
        pixelfs.pyc
        pixel_vm.pyc
        ...
      demos/
        ...

Philosophy:
"Source is for humans. Bytecode is for machines.
 Store both in pixels. Execute what's fastest."
"""

import sys
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def should_skip(path: Path) -> bool:
    """Check if file should be skipped"""
    skip_parts = {
        '__pycache__',
        '.pytest_cache',
        '.git',
        'htmlcov',
        'pixel_storage',
        'pixel_llm/data',
        'tests',  # Skip test files
    }

    # Check path parts
    for part in path.parts:
        if part in skip_parts:
            return True

    # Skip specific files
    if path.name in {
        'setup.py',
        'conftest.py',
        'pack_repository.py',
        'compile_to_bytecode.py',
    }:
        return True

    return False


def compile_file(source_path: Path, output_dir: Path) -> bool:
    """
    Compile one Python file to bytecode.

    Returns:
        True if successful, False if failed
    """
    # Calculate relative path
    rel_path = source_path.relative_to(ROOT)

    # Calculate output path
    pyc_path = output_dir / "bytecode" / rel_path.with_suffix(".pyc")

    # Create parent directory
    pyc_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Compile to bytecode
        py_compile.compile(
            str(source_path),
            cfile=str(pyc_path),
            doraise=True,
            optimize=-1  # No optimization (-1 = default)
        )

        size = pyc_path.stat().st_size
        source_size = source_path.stat().st_size

        # Calculate compression ratio
        ratio = (1 - size / source_size) * 100 if source_size > 0 else 0

        print(f"  ✅ {str(rel_path):<60s} {size:>8,} bytes ({ratio:+.1f}%)")

        return True

    except Exception as e:
        print(f"  ❌ {str(rel_path):<60s} FAILED: {e}")
        return False


def main():
    """Main compilation process"""
    output_dir = ROOT
    bytecode_dir = output_dir / "bytecode"

    print("=" * 80)
    print("PYTHON BYTECODE COMPILER")
    print("=" * 80)
    print()
    print(f"Source: {ROOT}")
    print(f"Output: {bytecode_dir}")
    print()
    print("─" * 80)
    print("COMPILING PYTHON FILES TO BYTECODE")
    print("─" * 80)
    print()

    # Find all Python files
    py_files = [f for f in ROOT.rglob("*.py") if not should_skip(f)]

    print(f"Found {len(py_files)} Python files to compile")
    print()

    # Compile each file
    compiled = 0
    failed = 0
    total_source_size = 0
    total_bytecode_size = 0

    for py_file in sorted(py_files):
        source_size = py_file.stat().st_size
        total_source_size += source_size

        if compile_file(py_file, output_dir):
            compiled += 1

            # Calculate bytecode size
            rel_path = py_file.relative_to(ROOT)
            pyc_path = bytecode_dir / rel_path.with_suffix(".pyc")
            if pyc_path.exists():
                total_bytecode_size += pyc_path.stat().st_size
        else:
            failed += 1

    print()
    print("─" * 80)
    print("COMPILATION COMPLETE")
    print("─" * 80)
    print()
    print(f"  ✅ Compiled: {compiled} files")
    if failed > 0:
        print(f"  ❌ Failed: {failed} files")
    print()
    print(f"  Source size:   {total_source_size:>12,} bytes")
    print(f"  Bytecode size: {total_bytecode_size:>12,} bytes")

    if total_source_size > 0:
        ratio = (1 - total_bytecode_size / total_source_size) * 100
        print(f"  Change:        {ratio:>12.1f}%")

    print()
    print(f"  Output directory: {bytecode_dir}")
    print()
    print("─" * 80)
    print("NEXT STEPS")
    print("─" * 80)
    print()
    print("  1. Pack bytecode into archive:")
    print("     python3 pack_repository.py")
    print()
    print("  2. Test bytecode loading:")
    print("     python3 test_bytecode_loading.py")
    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
