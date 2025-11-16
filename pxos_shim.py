#!/usr/bin/env python3
"""
pxOS Shim - Self-Contained Execution Launcher

This is the ONLY file that lives on disk.
Everything else (hypervisor, code, dependencies) lives in the pixel archive.

WHAT THIS DOES:
  1. Loads pixel archive from disk (pxos_repo.pxa)
  2. Installs archive importer (makes imports come from archive)
  3. Loads hypervisor FROM THE ARCHIVE
  4. Asks hypervisor to run the requested entrypoint
  5. Hypervisor + all application code execute from pixels

THE KEY INSIGHT:
  This shim is TINY and DUMB.
  It only knows how to:
    - Load bytes from pixel archive
    - Install import hook
    - Call the hypervisor

  The hypervisor (which lives IN THE ARCHIVE) does everything else.

ARCHITECTURE:
┌──────────────────────────────────┐
│  pxos_shim.py (this file)        │  ← Only file on disk
│  ~100 lines of loader code       │
└──────────────┬───────────────────┘
               │ loads
┌──────────────▼───────────────────┐
│  pxos_repo.pxa (pixel archive)   │  ← Everything else
│  ├─ Hypervisor                   │
│  ├─ Application code             │
│  ├─ Dependencies                 │
│  └─ Manifest                     │
└──────────────────────────────────┘

Usage:
    # Run default entrypoint
    python3 pxos_shim.py run

    # Run specific entrypoint
    python3 pxos_shim.py run vm

    # Run direct module:func
    python3 pxos_shim.py run pixel_llm.programs.hello_world:main

    # List available entrypoints
    python3 pxos_shim.py list

Philosophy:
"The shim is the bootloader.
 The archive is the OS.
 The hypervisor is the kernel.
 All in pixels."
"""

import sys
import json
import argparse
from pathlib import Path

# Bootstrap: add current directory to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Import ONLY the pixel infrastructure
# (These must exist on disk for bootstrapping)
from pixel_llm.core.pixel_archive import PixelArchiveReader
from pixel_llm.core.pxos_archive_loader import install_archive_importer_from_reader


# Default archive location
ARCHIVE_PATH = ROOT / "pxos_repo.pxa"


def load_archive() -> PixelArchiveReader:
    """Load pixel archive"""
    if not ARCHIVE_PATH.exists():
        print(f"❌ Archive not found: {ARCHIVE_PATH}")
        print()
        print("Create archive first:")
        print("  python3 pack_repository.py")
        sys.exit(1)

    print(f"Loading archive: {ARCHIVE_PATH.name}...")
    reader = PixelArchiveReader(str(ARCHIVE_PATH))
    print(f"✅ Loaded {reader.num_files} files ({reader.total_size:,} bytes)")
    print()

    return reader


def load_manifest(reader: PixelArchiveReader) -> dict:
    """Load manifest from archive"""
    try:
        manifest_bytes = reader.read_file("pxos_manifest.json")
        return json.loads(manifest_bytes.decode("utf-8"))
    except FileNotFoundError:
        print("⚠️  No manifest found in archive (pxos_manifest.json)")
        return {}


def cmd_run(args):
    """Run an entrypoint"""
    # 1. Load archive
    reader = load_archive()

    # 2. Install importer (makes imports come from archive)
    install_archive_importer_from_reader(reader, debug=args.debug)

    # 3. Load manifest
    manifest = load_manifest(reader)

    # 4. Import hypervisor FROM THE ARCHIVE
    try:
        from pixel_llm.core.hypervisor import create_hypervisor
    except ImportError as e:
        print(f"❌ Failed to load hypervisor from archive: {e}")
        print()
        print("The hypervisor module must exist in the archive:")
        print("  pixel_llm/core/hypervisor.py")
        sys.exit(1)

    # 5. Create hypervisor
    hypervisor = create_hypervisor(manifest, reader)

    # 6. Validate runtime
    try:
        hypervisor.validate_runtime()
    except Exception as e:
        print(f"❌ Runtime validation failed: {e}")
        sys.exit(1)

    # 7. Run entrypoint
    try:
        hypervisor.run_entrypoint(args.entrypoint)
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def cmd_list(args):
    """List available entrypoints"""
    # Load archive
    reader = load_archive()

    # Install importer
    install_archive_importer_from_reader(reader, debug=False)

    # Load manifest
    manifest = load_manifest(reader)

    # Import hypervisor
    from pixel_llm.core.hypervisor import create_hypervisor

    # Create hypervisor and list entrypoints
    hypervisor = create_hypervisor(manifest, reader)
    hypervisor.list_entrypoints()


def cmd_inspect(args):
    """Inspect archive contents"""
    reader = load_archive()

    print("=" * 70)
    print("Archive Contents")
    print("=" * 70)
    print()

    # Group by type
    by_ext = {}
    for file_path in reader.list_files():
        ext = Path(file_path).suffix or "no_ext"
        by_ext.setdefault(ext, []).append(file_path)

    for ext in sorted(by_ext.keys()):
        files = by_ext[ext]
        print(f"{ext:12s} {len(files):4d} files")

    print()
    print(f"Total: {reader.num_files} files, {reader.total_size:,} bytes")
    print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="pxOS Self-Contained Execution Launcher",
        epilog="The shim loads everything from the pixel archive."
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run an entrypoint from the archive"
    )
    run_parser.add_argument(
        "entrypoint",
        nargs="?",
        help="Entrypoint name or module:func (default: from manifest)"
    )
    run_parser.set_defaults(func=cmd_run)

    # List command
    list_parser = subparsers.add_parser(
        "list",
        help="List available entrypoints"
    )
    list_parser.set_defaults(func=cmd_list)

    # Inspect command
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect archive contents"
    )
    inspect_parser.set_defaults(func=cmd_inspect)

    # Parse and execute
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
