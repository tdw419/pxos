#!/usr/bin/env python3
"""
pxOS Pixel Loader - Custom Python Import System

This module implements a custom import hook that loads Python modules
from pixel images (.pxi) instead of traditional .py files.

PARADIGM:
  Traditional Python: import → read .py file → compile → execute
  pxOS Pixel Python:  import → read .pxi pixel → decode → compile → execute

This is the bridge between Python's import system and pixel-native code storage.

When you install this loader, ALL imports check pixels FIRST:
  import pixel_llm.core.pixelfs
    → looks up "pixel_llm.core.pixelfs" in manifest
    → finds "src_pixel_llm_core_pixelfs.pxi"
    → reads pixel image via PixelFS
    → decodes bytes to source
    → compiles and executes

Philosophy:
"The import statement no longer touches disk .py files.
 It reads pixels. Python becomes a pixel interpreter."
"""

import sys
import importlib.abc
import importlib.util
import json
from pathlib import Path
from typing import Optional, Dict, Any

# We need to import PixelFS the traditional way (for bootstrapping)
# Once the loader is installed, everything else comes from pixels
try:
    from pixel_llm.core.pixelfs import PixelFS
except ImportError:
    # Fallback for initial bootstrap
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pixel_llm.core.pixelfs import PixelFS


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "pixel_llm" / "pixel_manifest.json"


class PixelSourceLoader(importlib.abc.Loader):
    """
    Loader that reads Python source from pixel images.

    This replaces the standard file-based source loader.
    Instead of opening a .py file, it:
    1. Reads a .pxi pixel image
    2. Decodes the bytes
    3. Compiles to code
    4. Executes in module namespace
    """

    def __init__(self, fs: PixelFS, pixel_path: str, module_name: str):
        self.fs = fs
        self.pixel_path = pixel_path
        self.module_name = module_name

    def create_module(self, spec):
        """Use default module creation"""
        return None

    def exec_module(self, module):
        """Execute the module by loading from pixels"""
        try:
            # Set __file__ to the pixel path (for introspection)
            module.__file__ = self.pixel_path
            module.__loader__ = self

            # Read pixel image
            raw_bytes = self.fs.read(self.pixel_path)

            # Decode to source (pixels store raw UTF-8 Python source)
            source_code = raw_bytes.decode("utf-8")

            # Compile with special origin marker
            code_origin = f"<pxos:pixel:{self.pixel_path}>"
            compiled = compile(source_code, code_origin, "exec")

            # Execute in module namespace
            exec(compiled, module.__dict__)

        except Exception as e:
            raise ImportError(
                f"Failed to load pixel module {self.module_name} "
                f"from {self.pixel_path}: {e}"
            ) from e


class PixelSourceFinder(importlib.abc.MetaPathFinder):
    """
    Meta path finder that checks pixel manifest before falling back to filesystem.

    This is inserted at the START of sys.meta_path, so pixels take priority
    over traditional .py files.

    When Python tries to import a module:
    1. This finder checks if it exists in the pixel manifest
    2. If yes: returns a spec with PixelSourceLoader
    3. If no: returns None (fall back to standard import)
    """

    def __init__(self):
        self.fs = PixelFS()
        self.manifest: Dict[str, Any] = self._load_manifest()
        self.debug = False  # Set to True for import debugging

    def _load_manifest(self) -> Dict[str, Any]:
        """Load the pixel manifest"""
        if MANIFEST_PATH.exists():
            try:
                return json.loads(MANIFEST_PATH.read_text())
            except Exception as e:
                print(f"⚠️  Failed to load pixel manifest: {e}")
                return {}
        return {}

    def find_spec(self, fullname, path, target=None):
        """
        Find module spec for pixel-based modules.

        Args:
            fullname: Fully qualified module name (e.g., "pixel_llm.core.pixelfs")
            path: Package path (for submodules)
            target: Target module (usually None)

        Returns:
            ModuleSpec if found in pixels, None otherwise
        """
        # Check manifest
        entry = self.manifest.get(fullname)

        if not entry:
            # Not in pixels, let standard import handle it
            if self.debug:
                print(f"[pxos] {fullname} not in pixels, using standard import")
            return None

        # Found in pixels!
        if self.debug:
            print(f"[pxos] Loading {fullname} from pixels: {entry['pixel_key']}")

        pixel_path = str(ROOT / entry["pixel_path"])
        loader = PixelSourceLoader(self.fs, pixel_path, fullname)

        return importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=pixel_path
        )


# Global finder instance (created when loader is installed)
_pixel_finder: Optional[PixelSourceFinder] = None


def install_pixel_importer(debug: bool = False):
    """
    Install the pixel import hook.

    After calling this, ALL Python imports will check pixels first.

    Args:
        debug: If True, print debug info for each import attempt
    """
    global _pixel_finder

    if _pixel_finder is not None:
        print("⚠️  Pixel importer already installed")
        return

    # Create finder
    _pixel_finder = PixelSourceFinder()
    _pixel_finder.debug = debug

    # Insert at the START of meta_path (highest priority)
    sys.meta_path.insert(0, _pixel_finder)

    # Report
    num_modules = len(_pixel_finder.manifest)
    print("=" * 70)
    print("✅ pxOS PIXEL IMPORTER INSTALLED")
    print("=" * 70)
    print()
    print(f"  📦 {num_modules} modules available from pixels")
    print(f"  🔍 Manifest: {MANIFEST_PATH.relative_to(ROOT)}")
    print()
    print("  🎯 Import priority:")
    print("     1. Pixel images (.pxi) ← FIRST")
    print("     2. Standard .py files  ← fallback")
    print()
    print("=" * 70)
    print()


def uninstall_pixel_importer():
    """Remove the pixel import hook"""
    global _pixel_finder

    if _pixel_finder is None:
        print("⚠️  Pixel importer not installed")
        return

    try:
        sys.meta_path.remove(_pixel_finder)
        _pixel_finder = None
        print("✅ Pixel importer uninstalled")
    except ValueError:
        print("⚠️  Pixel importer not found in meta_path")


def get_pixel_stats() -> Dict[str, Any]:
    """Get statistics about pixel-based modules"""
    if _pixel_finder is None:
        return {"installed": False}

    manifest = _pixel_finder.manifest

    total_modules = len(manifest)
    total_bytes = sum(m["size_bytes"] for m in manifest.values())

    return {
        "installed": True,
        "total_modules": total_modules,
        "total_bytes": total_bytes,
        "manifest_path": str(MANIFEST_PATH),
    }


if __name__ == "__main__":
    print(__doc__)
    print()
    print("To use:")
    print("  from pixel_llm.core.pxos_loader import install_pixel_importer")
    print("  install_pixel_importer()")
    print()
    print("Then all imports will load from pixels!")
