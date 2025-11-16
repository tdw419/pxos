#!/usr/bin/env python3
"""
Bytecode Loader - Execute "Real" Bytecode from Pixels

THE NEXT EVOLUTION:
  Not just: Python source in pixels
  Not just: PixelVM bytecode in pixels
  But:      ANY bytecode format in pixels

This layer allows pxOS to execute:
  - Python bytecode (.pyc)
  - WebAssembly (.wasm) - future
  - JVM bytecode (.class) - future
  - Lua bytecode - future

All living in the pixel archive, all executing on the pixel substrate.

ARCHITECTURE:
┌────────────────────────────────────┐
│  Source Code (Python, Rust, etc.)  │
└──────────────┬─────────────────────┘
               │ compile (existing tools)
┌──────────────▼─────────────────────┐
│  Bytecode (.pyc, .wasm, etc.)      │  ← Store in pixels
└──────────────┬─────────────────────┘
               │ execute
┌──────────────▼─────────────────────┐
│  Bytecode Engine Layer (this)      │  ← Load from pixels
└──────────────┬─────────────────────┘
               │ runs on
┌──────────────▼─────────────────────┐
│  Pixel Substrate (PixelFS + VM)    │
└────────────────────────────────────┘

Philosophy:
"pxOS is a bytecode hypervisor.
 Any language, any compiler, any bytecode format.
 As long as it lives in pixels."
"""

import sys
import marshal
import types
import importlib.abc
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any, Callable

# Bootstrap
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pixel_llm.core.pixel_archive import PixelArchiveReader


class PythonBytecodeLoader(importlib.abc.Loader):
    """
    Loads Python modules from compiled bytecode (.pyc) stored in pixels.

    This is different from pxos_loader which loads .py source:
      - pxos_loader: reads .py source → compiles → executes
      - This loader: reads .pyc bytecode → deserializes → executes

    Advantages:
      - No compilation overhead (already compiled)
      - Can load bytecode-only modules (no source needed)
      - Faster cold start
      - True "bytecode from pixels" execution
    """

    def __init__(self, archive_reader: PixelArchiveReader, bytecode_path: str, module_name: str):
        self.archive_reader = archive_reader
        self.bytecode_path = bytecode_path
        self.module_name = module_name

    def create_module(self, spec):
        """Use default module creation"""
        return None

    def exec_module(self, module):
        """Execute module from bytecode"""
        try:
            # Set module metadata
            module.__file__ = f"<pxos:bytecode:{self.bytecode_path}>"
            module.__loader__ = self

            # Read .pyc file from archive
            pyc_bytes = self.archive_reader.read_file(self.bytecode_path)

            # Parse .pyc format:
            # [magic:4][flags:4][mtime/hash:8][size:4][code_object:rest]
            # We just need the code object at the end

            # Skip .pyc header (16 bytes for Python 3.7+)
            header_size = 16
            if len(pyc_bytes) < header_size:
                raise ImportError(f"Invalid .pyc file: too small ({len(pyc_bytes)} bytes)")

            code_bytes = pyc_bytes[header_size:]

            # Deserialize code object
            code_obj = marshal.loads(code_bytes)

            if not isinstance(code_obj, types.CodeType):
                raise ImportError(f"Invalid .pyc: expected code object, got {type(code_obj)}")

            # Execute in module namespace
            exec(code_obj, module.__dict__)

        except Exception as e:
            raise ImportError(
                f"Failed to load bytecode module {self.module_name} "
                f"from {self.bytecode_path}: {e}"
            ) from e


class BytecodeFinder(importlib.abc.MetaPathFinder):
    """
    Meta path finder that loads modules from bytecode in pixel archive.

    This finder looks for .pyc files in the archive and loads them
    as bytecode-only modules.
    """

    def __init__(self, archive_reader: PixelArchiveReader):
        self.archive_reader = archive_reader
        self.debug = False

        # Build bytecode module index
        self._build_bytecode_index()

    def _build_bytecode_index(self):
        """Build index of available bytecode modules"""
        self.bytecode_modules: Dict[str, str] = {}

        for file_path in self.archive_reader.list_files():
            # Only .pyc files
            if not file_path.endswith(".pyc"):
                continue

            # Convert path to module name
            # e.g., bytecode/pixel_llm/core/pixelfs.pyc → pixel_llm.core.pixelfs

            # Store original path for reading
            original_path = file_path

            # Remove "bytecode/" prefix for module name extraction
            if file_path.startswith("bytecode/"):
                file_path = file_path[len("bytecode/"):]

            # Convert to module name
            module_name = file_path.replace("/", ".").replace(".pyc", "")

            # Store original path (with bytecode/ prefix)
            self.bytecode_modules[module_name] = original_path

    def find_spec(self, fullname, path, target=None):
        """Find module spec for bytecode modules"""
        # Check if this module has bytecode available
        if fullname not in self.bytecode_modules:
            if self.debug:
                print(f"[bytecode] {fullname} not in bytecode index")
            return None

        # Found bytecode!
        bytecode_path = self.bytecode_modules[fullname]

        if self.debug:
            print(f"[bytecode] Loading {fullname} from bytecode: {bytecode_path}")

        loader = PythonBytecodeLoader(self.archive_reader, bytecode_path, fullname)

        return importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=f"<pxos:bytecode:{bytecode_path}>"
        )


# Global finder instance
_bytecode_finder: Optional[BytecodeFinder] = None


def install_bytecode_importer(archive_path: str, debug: bool = False):
    """
    Install the bytecode import hook.

    After calling this, Python imports will check the archive for
    .pyc bytecode files before falling back to source.

    Args:
        archive_path: Path to .pxa archive containing bytecode
        debug: Enable debug output
    """
    global _bytecode_finder

    if _bytecode_finder is not None:
        print("⚠️  Bytecode importer already installed")
        return

    # Load archive
    archive_reader = PixelArchiveReader(archive_path)

    # Create finder
    _bytecode_finder = BytecodeFinder(archive_reader)
    _bytecode_finder.debug = debug

    # Insert at START of meta_path (highest priority)
    sys.meta_path.insert(0, _bytecode_finder)

    # Report
    num_modules = len(_bytecode_finder.bytecode_modules)

    print("=" * 70)
    print("✅ pxOS BYTECODE IMPORTER INSTALLED")
    print("=" * 70)
    print()
    print(f"  🔥 {num_modules} bytecode modules available")
    print(f"  💾 Archive: {Path(archive_path).name}")
    print()
    print("  🎯 Import priority:")
    print("     1. Python bytecode (.pyc) from archive ← FIRST")
    print("     2. Standard .py files                  ← fallback")
    print()
    print("  ⚡ BYTECODE EXECUTION FROM PIXELS")
    print()
    print("=" * 70)
    print()


def uninstall_bytecode_importer():
    """Remove the bytecode import hook"""
    global _bytecode_finder

    if _bytecode_finder is None:
        print("⚠️  Bytecode importer not installed")
        return

    try:
        sys.meta_path.remove(_bytecode_finder)
        _bytecode_finder = None
        print("✅ Bytecode importer uninstalled")
    except ValueError:
        print("⚠️  Bytecode importer not found in meta_path")


def get_bytecode_stats() -> Dict[str, Any]:
    """Get statistics about bytecode modules"""
    if _bytecode_finder is None:
        return {"installed": False}

    return {
        "installed": True,
        "total_modules": len(_bytecode_finder.bytecode_modules),
        "modules": sorted(_bytecode_finder.bytecode_modules.keys()),
    }


# ============================================================================
# Bytecode Compilation Utilities
# ============================================================================

def compile_python_to_bytecode(source_path: Path, output_path: Path):
    """
    Compile a Python source file to .pyc bytecode.

    This creates a .pyc file that can be stored in the pixel archive
    and loaded without the original source.

    Args:
        source_path: Path to .py source file
        output_path: Path to write .pyc file
    """
    import py_compile

    print(f"Compiling: {source_path} → {output_path}")

    try:
        py_compile.compile(
            str(source_path),
            cfile=str(output_path),
            doraise=True
        )

        size = output_path.stat().st_size
        print(f"✅ Compiled {size} bytes")

    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        raise


def compile_repository_to_bytecode(repo_root: Path, output_dir: Path):
    """
    Compile all Python files in repository to bytecode.

    This creates a parallel directory structure with .pyc files
    that can be packed into the pixel archive.

    Args:
        repo_root: Root of repository to compile
        output_dir: Directory to write .pyc files
    """
    import py_compile

    output_dir.mkdir(parents=True, exist_ok=True)

    compiled = 0
    failed = 0

    print("=" * 70)
    print("COMPILING REPOSITORY TO BYTECODE")
    print("=" * 70)
    print()

    for py_file in repo_root.rglob("*.py"):
        # Skip test files, build scripts, etc.
        if any(skip in py_file.parts for skip in ["__pycache__", "tests", "build"]):
            continue

        # Calculate output path
        rel_path = py_file.relative_to(repo_root)
        pyc_path = output_dir / "bytecode" / rel_path.with_suffix(".pyc")

        pyc_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            py_compile.compile(
                str(py_file),
                cfile=str(pyc_path),
                doraise=True
            )

            size = pyc_path.stat().st_size
            print(f"  ✅ {rel_path} → {size:,} bytes")
            compiled += 1

        except Exception as e:
            print(f"  ❌ {rel_path}: {e}")
            failed += 1

    print()
    print("=" * 70)
    print(f"Compiled: {compiled} modules")
    print(f"Failed: {failed} modules")
    print(f"Output: {output_dir}/bytecode/")
    print("=" * 70)
    print()


if __name__ == "__main__":
    print(__doc__)
    print()
    print("To use:")
    print("  from pixel_llm.core.bytecode_loader import install_bytecode_importer")
    print("  install_bytecode_importer('pxos_repo.pxa')")
    print()
    print("Then all imports will load bytecode from the pixel archive!")
