#!/usr/bin/env python3
"""
pxOS Archive Loader - Load Python Modules from Pixel Archive

This is the EVOLUTION of pxos_loader.py:

Before: Individual .pxi files per module (scattered)
After:  ONE .pxa archive with all modules (unified)

BENEFITS:
  - Single file access (easier for AI/development)
  - Memory-mapped efficiency
  - Faster cold starts (one image load vs many)
  - Simpler distribution (one file to share)

PARADIGM:
  Traditional Python: import → read .py file from disk
  pxOS Pixel v1:      import → read individual .pxi pixel
  pxOS Pixel v2:      import → read from .pxa ARCHIVE ← YOU ARE HERE

This loader reads ALL modules from a single pixel archive file.

Philosophy:
"One cartridge. One truth. Everything you need."
"""

import sys
import importlib.abc
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any

# Bootstrap PixelFS and PixelArchiveReader
try:
    from pixel_llm.core.pixelfs import PixelFS
    from pixel_llm.core.pixel_archive import PixelArchiveReader
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pixel_llm.core.pixelfs import PixelFS
    from pixel_llm.core.pixel_archive import PixelArchiveReader


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = ROOT / "pxos_repo.pxa"


class PixelArchiveLoader(importlib.abc.Loader):
    """
    Loader that reads Python source from pixel archive.

    Instead of reading individual .pxi files, this reads from
    a single .pxa archive containing all modules.
    """

    def __init__(self, archive_reader: PixelArchiveReader, archive_path: str, module_name: str):
        self.archive_reader = archive_reader
        self.archive_path = archive_path
        self.module_name = module_name

    def create_module(self, spec):
        """Use default module creation"""
        return None

    def exec_module(self, module):
        """Execute the module by loading from archive"""
        try:
            # Set __file__ to archive reference
            module.__file__ = f"<pxos:archive:{self.archive_path}:{self.module_name}>"
            module.__loader__ = self

            # Read from archive
            # Convert module name to path (e.g., pixel_llm.core.pixelfs → pixel_llm/core/pixelfs.py)
            module_file_path = self.module_name.replace(".", "/") + ".py"

            raw_bytes = self.archive_reader.read_file(module_file_path)

            # Decode to source
            source_code = raw_bytes.decode("utf-8")

            # Compile with special origin marker
            code_origin = f"<pxos:archive:{self.archive_path}:{module_file_path}>"
            compiled = compile(source_code, code_origin, "exec")

            # Execute in module namespace
            exec(compiled, module.__dict__)

        except FileNotFoundError:
            raise ImportError(
                f"Module {self.module_name} not found in archive {self.archive_path}"
            )
        except Exception as e:
            raise ImportError(
                f"Failed to load archive module {self.module_name}: {e}"
            ) from e


class PixelArchiveFinder(importlib.abc.MetaPathFinder):
    """
    Meta path finder that loads modules from pixel archive.

    This is the archive equivalent of PixelSourceFinder.
    Instead of checking a manifest, it checks the archive's file list.
    """

    def __init__(self, archive_path: str = None, archive_reader: PixelArchiveReader = None):
        """
        Create finder from archive path or reader.

        Args:
            archive_path: Path to .pxa file (if loading from disk)
            archive_reader: Existing reader (if already loaded)
        """
        if archive_path is None and archive_reader is None:
            raise ValueError("Either archive_path or archive_reader must be provided")

        if archive_reader is not None:
            # Use provided reader
            self.archive_reader = archive_reader
            self.archive_path = getattr(archive_reader, 'archive_path', '<memory>')
        else:
            # Load from path
            self.archive_path = archive_path
            self.archive_reader = PixelArchiveReader(archive_path)

        self.debug = False

        # Build module index from archive files
        self._build_module_index()

    def _build_module_index(self):
        """Build index of Python modules in archive"""
        self.modules: Dict[str, str] = {}

        for file_path in self.archive_reader.list_files():
            # Only Python files
            if not file_path.endswith(".py"):
                continue

            # Convert path to module name
            # e.g., pixel_llm/core/pixelfs.py → pixel_llm.core.pixelfs
            module_name = file_path.replace("/", ".").replace(".py", "")

            self.modules[module_name] = file_path

    def find_spec(self, fullname, path, target=None):
        """
        Find module spec for archive-based modules.

        Args:
            fullname: Fully qualified module name
            path: Package path
            target: Target module

        Returns:
            ModuleSpec if found in archive, None otherwise
        """
        # Check if module exists in archive
        if fullname not in self.modules:
            if self.debug:
                print(f"[pxos:archive] {fullname} not in archive")
            return None

        # Found in archive!
        if self.debug:
            print(f"[pxos:archive] Loading {fullname} from archive")

        loader = PixelArchiveLoader(self.archive_reader, self.archive_path, fullname)

        return importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=f"<pxos:archive:{self.archive_path}>"
        )


# Global finder instance
_archive_finder: Optional[PixelArchiveFinder] = None


def install_archive_importer(archive_path: Optional[str] = None, debug: bool = False):
    """
    Install the pixel archive import hook.

    After calling this, ALL Python imports will check the archive first.

    Args:
        archive_path: Path to .pxa archive (default: pxos_repo.pxa)
        debug: If True, print debug info for each import
    """
    global _archive_finder

    if _archive_finder is not None:
        print("⚠️  Pixel archive importer already installed")
        return

    # Use default archive if not specified
    if archive_path is None:
        archive_path = str(DEFAULT_ARCHIVE)

    # Check archive exists
    if not Path(archive_path).exists():
        print(f"❌ Archive not found: {archive_path}")
        print()
        print("Create archive first:")
        print("  python3 pack_repository.py")
        return

    # Create finder
    _archive_finder = PixelArchiveFinder(archive_path)
    _archive_finder.debug = debug

    # Insert at the START of meta_path (highest priority)
    sys.meta_path.insert(0, _archive_finder)

    # Report
    num_modules = len(_archive_finder.modules)
    archive_size = _archive_finder.archive_reader.total_size

    print("=" * 70)
    print("✅ pxOS PIXEL ARCHIVE IMPORTER INSTALLED")
    print("=" * 70)
    print()
    print(f"  📦 {num_modules} modules available from archive")
    print(f"  💾 Archive: {Path(archive_path).name}")
    print(f"  📊 Size: {archive_size:,} bytes")
    print()
    print("  🎯 Import priority:")
    print("     1. Pixel archive (.pxa) ← FIRST")
    print("     2. Standard .py files   ← fallback")
    print()
    print("  🔥 ONE FILE. ALL CODE. PURE SUBSTRATE.")
    print()
    print("=" * 70)
    print()


def install_archive_importer_from_reader(archive_reader: PixelArchiveReader, debug: bool = False):
    """
    Install the pixel archive import hook from an existing reader.

    This is used by the hypervisor to install the importer from an in-memory archive.

    Args:
        archive_reader: PixelArchiveReader instance (already loaded)
        debug: If True, print debug info for each import
    """
    global _archive_finder

    if _archive_finder is not None:
        print("⚠️  Pixel archive importer already installed")
        return

    # Create finder from reader
    _archive_finder = PixelArchiveFinder(archive_reader=archive_reader)
    _archive_finder.debug = debug

    # Insert at the START of meta_path (highest priority)
    sys.meta_path.insert(0, _archive_finder)

    # Report
    num_modules = len(_archive_finder.modules)
    archive_size = _archive_finder.archive_reader.total_size

    print("=" * 70)
    print("✅ pxOS PIXEL ARCHIVE IMPORTER INSTALLED (from memory)")
    print("=" * 70)
    print()
    print(f"  📦 {num_modules} modules available from archive")
    print(f"  💾 Archive: <memory>")
    print(f"  📊 Size: {archive_size:,} bytes")
    print()
    print("  🎯 Import priority:")
    print("     1. Pixel archive (memory) ← FIRST")
    print("     2. Standard .py files      ← fallback")
    print()
    print("  🔥 SELF-CONTAINED EXECUTION FROM PIXELS")
    print()
    print("=" * 70)
    print()


def uninstall_archive_importer():
    """Remove the pixel archive import hook"""
    global _archive_finder

    if _archive_finder is None:
        print("⚠️  Pixel archive importer not installed")
        return

    try:
        sys.meta_path.remove(_archive_finder)
        _archive_finder = None
        print("✅ Pixel archive importer uninstalled")
    except ValueError:
        print("⚠️  Pixel archive importer not found in meta_path")


def get_archive_stats() -> Dict[str, Any]:
    """Get statistics about archive-based modules"""
    if _archive_finder is None:
        return {"installed": False}

    return {
        "installed": True,
        "archive_path": _archive_finder.archive_path,
        "total_modules": len(_archive_finder.modules),
        "total_files": _archive_finder.archive_reader.num_files,
        "total_bytes": _archive_finder.archive_reader.total_size,
        "modules": sorted(_archive_finder.modules.keys()),
    }


if __name__ == "__main__":
    print(__doc__)
    print()
    print("To use:")
    print("  from pixel_llm.core.pxos_archive_loader import install_archive_importer")
    print("  install_archive_importer()")
    print()
    print("Then all imports will load from the pixel archive!")
