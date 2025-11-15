#!/usr/bin/env python3
"""
Project Boot Pixel Packer

Pack the ENTIRE pxOS project (code + worlds + docs) into a single 1x1 pixel.

This is the ultimate compression: not just universes, but the toolchain itself.

Usage:
    pack_project_boot_pixel.py [project_root] [name]
"""

import os
import io
import tarfile
import zlib
import json
import hashlib
from pathlib import Path
from PIL import Image
import sys


class ProjectBootPixelPacker:
    """Pack entire projects into single pixels"""

    def __init__(self):
        self.registry_path = Path("project_boot_registry.json")
        self.blobs_dir = Path("project_blobs")
        self.blobs_dir.mkdir(exist_ok=True)
        self.registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load project boot registry"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        """Save registry"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def tar_directory(self, root: Path, exclude=None) -> bytes:
        """Create tar.gz of directory"""
        exclude = exclude or []

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for dirpath, dirnames, filenames in os.walk(root):
                # Filter out excluded directories
                dirnames[:] = [d for d in dirnames if not any(d.startswith(e) for e in exclude)]

                dirpath = Path(dirpath)
                for fname in filenames:
                    fp = dirpath / fname
                    try:
                        rel = fp.relative_to(root)
                        # Skip excluded files
                        if any(str(rel).startswith(e) for e in exclude):
                            continue
                        tf.add(fp, arcname=str(rel))
                    except (ValueError, OSError) as e:
                        print(f"Warning: Skipping {fp}: {e}")
                        continue

        return buf.getvalue()

    def pack_project(
        self,
        project_root: str,
        name: str = "pxOS",
        description: str = "Complete pxOS God Pixel project"
    ):
        """Pack entire project into a Boot Pixel"""

        root = Path(project_root).resolve()

        if not root.exists():
            raise ValueError(f"Project root {root} does not exist")

        print("╔═══════════════════════════════════════════════════════════╗")
        print("║          PROJECT BOOT PIXEL PACKER                        ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print()
        print(f"Project root: {root}")
        print(f"Name: {name}")
        print()

        # Create tar.gz of project
        print("Creating tar.gz archive...")
        exclude = [
            ".git",
            "__pycache__",
            "project_blobs",
            ".gitignore",
            "*.pyc",
            ".DS_Store",
            "test_*.png",
            "resurrected.png",
            "bootstrap_god.png"
        ]

        raw_tar = self.tar_directory(root, exclude=exclude)
        print(f"  Raw tar.gz: {len(raw_tar):,} bytes")

        # Additional zlib compression
        print("Compressing with zlib...")
        compressed = zlib.compress(raw_tar, level=9)
        print(f"  Compressed: {len(compressed):,} bytes")
        print(f"  Ratio: {len(compressed) / len(raw_tar):.2%}")

        # Generate pixel color from hash
        hash_obj = hashlib.sha256(compressed)
        hash_bytes = hash_obj.digest()

        R = hash_bytes[0]
        G = hash_bytes[1]
        B = hash_bytes[2]
        A = hash_bytes[3]

        pid = (R << 24) | (G << 16) | (B << 8) | A

        print()
        print(f"Project Boot Pixel Color: RGBA({R}, {G}, {B}, {A})")
        print(f"Hex: #{R:02X}{G:02X}{B:02X}{A:02X}")
        print(f"ID: 0x{pid:08X}")

        # Save compressed blob
        blob_path = self.blobs_dir / f"project_{pid:08x}.bin"
        blob_path.write_bytes(compressed)
        print()
        print(f"Compressed blob saved: {blob_path}")

        # Create 1x1 boot pixel
        boot_pixel = Image.new("RGBA", (1, 1), (R, G, B, A))
        output_name = f"project_{name.replace(' ', '_').lower()}.boot.png"
        boot_pixel.save(output_name)

        # Update registry
        color_key = f"{R},{G},{B},{A}"
        self.registry[color_key] = {
            "name": name,
            "description": description,
            "root": str(root),
            "blob": str(blob_path),
            "pixel": [R, G, B, A],
            "id": f"0x{pid:08X}",
            "hash": hash_obj.hexdigest(),
            "size": len(compressed),
            "original_size": len(raw_tar)
        }
        self._save_registry()

        print(f"Boot pixel saved: {output_name}")
        print()
        print("╔═══════════════════════════════════════════════════════════╗")
        print("║              PROJECT BOOT PIXEL CREATED                   ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print()
        print(f"✓ Original size: {len(raw_tar):,} bytes")
        print(f"✓ Compressed: {len(compressed):,} bytes")
        print(f"✓ Compression: {(1 - len(compressed) / len(raw_tar)) * 100:.2f}%")
        print()
        print("This ONE PIXEL contains:")
        print("  - All Python code (PXI_CPU, compressor, bootloader, etc.)")
        print("  - All God Pixels (TestPattern, LifeSim)")
        print("  - All documentation")
        print("  - All tools and utilities")
        print()
        print("To resurrect:")
        print(f"  python3 boot_project_from_pixel.py {output_name}")

        return (R, G, B, A)

    def list_projects(self):
        """List all packed projects"""
        if not self.registry:
            print("No projects packed yet.")
            return

        print("╔═══════════════════════════════════════════════════════════╗")
        print("║              PACKED PROJECTS                              ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print()

        for i, (color, meta) in enumerate(self.registry.items(), 1):
            name = meta.get("name", "Unnamed")
            desc = meta.get("description", "")
            size = meta.get("size", 0)
            orig = meta.get("original_size", 0)

            r, g, b, a = [int(x) for x in color.split(",")]

            print(f"{i}. {name}")
            print(f"   Color: RGBA({r}, {g}, {b}, {a})")
            print(f"   Size: {orig:,} → {size:,} bytes ({(1 - size/orig)*100:.1f}% reduction)")
            print(f"   Description: {desc}")
            print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pack entire pxOS project into a single Boot Pixel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pack_project_boot_pixel.py .                    # Pack current directory
  pack_project_boot_pixel.py /path/to/pxos        # Pack specific path
  pack_project_boot_pixel.py . --name "My pxOS"  # Custom name
  pack_project_boot_pixel.py --list              # List packed projects
        """
    )

    parser.add_argument("project_root", nargs="?", default=".",
                        help="Root directory of project to pack")
    parser.add_argument("--name", "-n", default="pxOS",
                        help="Name for this project")
    parser.add_argument("--desc", "-d", default="Complete pxOS God Pixel project",
                        help="Description")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List all packed projects")

    args = parser.parse_args()

    packer = ProjectBootPixelPacker()

    if args.list:
        packer.list_projects()
    else:
        packer.pack_project(args.project_root, args.name, args.desc)


if __name__ == "__main__":
    main()
