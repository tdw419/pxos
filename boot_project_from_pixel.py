#!/usr/bin/env python3
"""
Project Boot Pixel Bootstrapper

Resurrect and run an entire pxOS project from a single 1x1 Boot Pixel.

This is the hyper-bootstrap: the minimal stub that brings everything back to life.

Usage:
    boot_project_from_pixel.py project_pxos.boot.png [--extract-only]
"""

import sys
import zlib
import json
import tarfile
import io
import tempfile
import importlib.util
from pathlib import Path
from PIL import Image


class ProjectBootPixelBootstrapper:
    """Resurrect entire projects from single pixels"""

    def __init__(self):
        self.registry_path = Path("project_boot_registry.json")
        self.blobs_dir = Path("project_blobs")
        self.registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load registry"""
        if not self.registry_path.exists():
            raise RuntimeError(
                f"Registry not found: {self.registry_path}\n"
                f"Cannot resurrect project without registry.\n"
                f"Make sure you're in the same directory where you packed the project."
            )
        with open(self.registry_path, 'r') as f:
            return json.load(f)

    def resurrect_project(
        self,
        pixel_path: str,
        extract_only: bool = False,
        extract_to: str = None
    ):
        """Resurrect entire project from Boot Pixel"""

        print("╔═══════════════════════════════════════════════════════════╗")
        print("║        PROJECT BOOT PIXEL BOOTSTRAPPER                    ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print()

        # Load the Boot Pixel
        print(f"Loading Boot Pixel: {pixel_path}")
        img = Image.open(pixel_path).convert("RGBA")

        if img.size != (1, 1):
            raise RuntimeError(
                f"Invalid Boot Pixel: must be 1x1, got {img.size[0]}x{img.size[1]}"
            )

        R, G, B, A = img.getpixel((0, 0))
        print(f"  Color: RGBA({R}, {G}, {B}, {A})")

        # Look up in registry
        color_key = f"{R},{G},{B},{A}"
        if color_key not in self.registry:
            raise RuntimeError(
                f"Boot Pixel not found in registry.\n"
                f"Color: {color_key}\n"
                f"Available projects: {len(self.registry)}"
            )

        entry = self.registry[color_key]
        print(f"  Found: {entry['name']}")
        print(f"  Description: {entry['description']}")
        print()

        # Load compressed blob
        blob_path = Path(entry["blob"])
        if not blob_path.exists():
            raise RuntimeError(f"Compressed blob not found: {blob_path}")

        print(f"Loading compressed blob: {blob_path}")
        compressed = blob_path.read_bytes()
        print(f"  Compressed size: {len(compressed):,} bytes")

        # Decompress
        print("Decompressing...")
        raw_tar = zlib.decompress(compressed)
        print(f"  Decompressed: {len(raw_tar):,} bytes")
        print()

        # Determine extract location
        if extract_to:
            extract_dir = Path(extract_to)
        elif extract_only:
            extract_dir = Path.cwd() / f"pxos_extracted_{entry['name'].replace(' ', '_').lower()}"
        else:
            extract_dir = Path(tempfile.mkdtemp(prefix="pxos_boot_"))

        print(f"Extracting to: {extract_dir}")
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Extract tar
        with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r:gz") as tf:
            tf.extractall(extract_dir)

        print(f"✓ Extraction complete")
        print()

        if extract_only:
            print("╔═══════════════════════════════════════════════════════════╗")
            print("║              EXTRACTION COMPLETE                          ║")
            print("╚═══════════════════════════════════════════════════════════╝")
            print()
            print(f"Project extracted to: {extract_dir}")
            print()
            print("To run:")
            print(f"  cd {extract_dir}")
            print(f"  python3 pxos_boot.py --list")
            return extract_dir

        # Boot the project
        print("╔═══════════════════════════════════════════════════════════╗")
        print("║              BOOTING RESURRECTED PROJECT                  ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print()

        # Import pxos_boot from extracted directory
        pxos_boot_path = extract_dir / "pxos_boot.py"
        if not pxos_boot_path.exists():
            raise RuntimeError(
                f"pxos_boot.py not found in extracted project.\n"
                f"Expected: {pxos_boot_path}"
            )

        print(f"Importing pxos_boot from {pxos_boot_path}")

        # Load module dynamically
        spec = importlib.util.spec_from_file_location("pxos_boot", pxos_boot_path)
        pxos_boot = importlib.util.module_from_spec(spec)
        sys.modules["pxos_boot"] = pxos_boot

        # Add extracted dir to path so imports work
        sys.path.insert(0, str(extract_dir))

        # Execute the module
        spec.loader.exec_module(pxos_boot)

        print()
        print("Handing control to pxos_boot.main()...")
        print("─" * 60)
        print()

        # Run the bootloader
        if hasattr(pxos_boot, "main"):
            pxos_boot.main()
        else:
            print("⚠️  Warning: pxos_boot.main() not found")
            print("    Resurrection complete but nothing to run")
            print()
            print(f"    Extracted to: {extract_dir}")

        return extract_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Boot entire pxOS project from a single Boot Pixel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  boot_project_from_pixel.py project_pxos.boot.png
  boot_project_from_pixel.py project_pxos.boot.png --extract-only
  boot_project_from_pixel.py project_pxos.boot.png --extract-to ./my_pxos
        """
    )

    parser.add_argument("pixel", help="Path to Project Boot Pixel (1x1 PNG)")
    parser.add_argument("--extract-only", "-e", action="store_true",
                        help="Only extract, don't boot")
    parser.add_argument("--extract-to", "-o", help="Extract to specific directory")

    args = parser.parse_args()

    bootstrapper = ProjectBootPixelBootstrapper()

    try:
        bootstrapper.resurrect_project(
            args.pixel,
            extract_only=args.extract_only,
            extract_to=args.extract_to
        )
    except Exception as e:
        print()
        print("╔═══════════════════════════════════════════════════════════╗")
        print("║                     BOOT FAILED                           ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print()
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
