#!/usr/bin/env python3
"""
pixelfs_builder.py - PixelFS Virtual File System Builder

Creates and manages a virtual filesystem that maps paths to file_ids (boot pixels).

PixelFS is the directory tree for pxOS - every file, module, world, and model
gets a logical path that resolves to a sub-boot pixel.

Usage:
  python3 pixelfs_builder.py init
  python3 pixelfs_builder.py add /boot/00_kernel pxi_cpu.py --type pxi_module
  python3 pixelfs_builder.py add /worlds/lifesim god.png --type world
  python3 pixelfs_builder.py list
  python3 pixelfs_builder.py tree
  python3 pixelfs_builder.py resolve /boot/00_kernel
"""

import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List
import hashlib

class PixelFS:
    """Virtual filesystem for pxOS - maps paths to file_ids"""

    def __init__(self):
        self.fs_path = Path("pixelfs.json")
        self.registry_path = Path("file_boot_registry.json")

        # Load or create filesystem
        if self.fs_path.exists():
            with open(self.fs_path, 'r') as f:
                self.fs = json.load(f)
        else:
            self.fs = {
                "version": "1.0",
                "description": "PixelFS virtual filesystem for pxOS",
                "entries": {}
            }

        # Load file boot registry if it exists
        self.registry = {}
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)

    def _save(self):
        """Save filesystem to disk"""
        with open(self.fs_path, 'w') as f:
            json.dump(self.fs, f, indent=2)

    def add_path(self, path: str, file_path: str = None, file_id: int = None,
                 file_type: str = "data", description: str = None):
        """
        Add a path to PixelFS

        Args:
            path: Logical path in PixelFS (e.g., /boot/00_kernel)
            file_path: Physical file path (will pack if needed)
            file_id: Existing file_id from registry (if already packed)
            file_type: Type of file (py, pxi_module, world, model, config, data)
            description: Human-readable description
        """
        # Validate path
        if not path.startswith('/'):
            raise ValueError(f"Path must start with /: {path}")

        # Get or create file_id
        if file_id is None:
            if file_path is None:
                raise ValueError("Must provide either file_path or file_id")

            # Check if file already packed
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Look for existing file_id in registry
            file_id = self._find_file_id(file_path)

            if file_id is None:
                # Need to pack this file first
                print(f"⚠️  File not yet packed: {file_path}")
                print(f"   Run: python3 pack_file_to_boot_pixel.py add {file_path} --type {file_type}")
                print(f"   Then try again.")
                return False

        # Get file info from registry
        entry = self.registry.get(str(file_id))
        if not entry:
            print(f"❌ File ID {file_id} not found in registry")
            return False

        # Add to PixelFS
        self.fs["entries"][path] = {
            "file_id": file_id,
            "file_id_hex": f"0x{file_id:08X}",
            "type": file_type,
            "description": description or entry.get('name', 'No description'),
            "pixel": entry.get('pixel', [0, 0, 0, 0]),
            "original_path": file_path or entry.get('original_path', 'unknown'),
            "size": entry.get('original_size', 0)
        }

        self._save()

        pixel = entry.get('pixel', [0, 0, 0, 0])
        print(f"✅ Added to PixelFS:")
        print(f"   Path:     {path}")
        print(f"   File ID:  0x{file_id:08X}")
        print(f"   RGBA:     ({pixel[0]}, {pixel[1]}, {pixel[2]}, {pixel[3]})")
        print(f"   Type:     {file_type}")

        return True

    def _find_file_id(self, file_path: str) -> Optional[int]:
        """Find file_id for a given file path in registry"""
        file_path_obj = Path(file_path)

        # Search registry for matching original_path
        for file_id_str, entry in self.registry.items():
            if entry.get('original_path') == str(file_path_obj):
                return int(file_id_str)

        return None

    def resolve(self, path: str) -> Optional[Dict]:
        """Resolve a path to its file_id and metadata"""
        entry = self.fs["entries"].get(path)
        if not entry:
            return None

        return entry

    def list_all(self):
        """List all entries in PixelFS"""
        if not self.fs["entries"]:
            print("PixelFS is empty. Use 'add' to add paths.")
            return

        print(f"\n{'Path':<40} {'File ID':<12} {'Type':<15} {'Size':<10}")
        print("=" * 80)

        for path in sorted(self.fs["entries"].keys()):
            entry = self.fs["entries"][path]
            file_id = entry['file_id_hex']
            ftype = entry['type']
            size = entry.get('size', 0)

            print(f"{path:<40} {file_id:<12} {ftype:<15} {size:<10}")

        print(f"\nTotal: {len(self.fs['entries'])} entries\n")

    def tree(self):
        """Display PixelFS as a tree structure"""
        if not self.fs["entries"]:
            print("PixelFS is empty.")
            return

        # Build tree structure
        tree = {}
        for path in sorted(self.fs["entries"].keys()):
            parts = [p for p in path.split('/') if p]
            current = tree
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # Leaf node
                    current[part] = self.fs["entries"][path]
                else:
                    # Directory node
                    if part not in current:
                        current[part] = {}
                    current = current[part]

        # Print tree
        print("\nPixelFS Tree:")
        print("=" * 60)
        self._print_tree(tree, prefix="")
        print()

    def _print_tree(self, node, prefix="", is_last=True):
        """Recursively print tree structure"""
        items = list(node.items())
        for i, (name, value) in enumerate(items):
            is_last_item = (i == len(items) - 1)

            # Print branch characters
            if prefix == "":
                branch = ""
            else:
                branch = "└── " if is_last_item else "├── "

            if isinstance(value, dict) and not any(k in value for k in ['file_id', 'type']):
                # Directory
                print(f"{prefix}{branch}{name}/")

                # Recurse
                new_prefix = prefix + ("    " if is_last_item else "│   ")
                self._print_tree(value, new_prefix, is_last_item)
            else:
                # File
                file_id = value.get('file_id_hex', 'unknown')
                ftype = value.get('type', 'unknown')
                pixel = value.get('pixel', [0, 0, 0, 0])

                print(f"{prefix}{branch}{name}  [{ftype}]  {file_id}  RGBA({pixel[0]},{pixel[1]},{pixel[2]},{pixel[3]})")

    def auto_discover(self):
        """Auto-discover files from file_boot_registry and suggest PixelFS paths"""
        if not self.registry:
            print("No file_boot_registry.json found. Run pack_file_to_boot_pixel.py first.")
            return

        print("\nAuto-discovering files from registry...")
        print("=" * 60)

        suggestions = []

        for file_id_str, entry in self.registry.items():
            name = entry.get('name', 'unknown')
            ftype = entry.get('type', 'data')

            # Suggest path based on type
            if ftype == 'py':
                suggested_path = f"/system/{name}"
            elif ftype == 'pxi_module':
                suggested_path = f"/apps/{name}"
            elif ftype == 'world':
                suggested_path = f"/worlds/{name}"
            elif ftype == 'llm_model' or ftype == 'model':
                suggested_path = f"/models/{name}"
            elif ftype == 'config':
                suggested_path = f"/config/{name}"
            else:
                suggested_path = f"/data/{name}"

            # Check if already in PixelFS
            already_added = any(
                e['file_id'] == int(file_id_str)
                for e in self.fs["entries"].values()
            )

            if not already_added:
                suggestions.append({
                    'path': suggested_path,
                    'file_id': int(file_id_str),
                    'type': ftype,
                    'name': name
                })

        if not suggestions:
            print("All files already in PixelFS!")
            return

        print(f"\nFound {len(suggestions)} unpacked files:\n")
        for s in suggestions:
            print(f"  {s['path']:<40} ({s['type']})")

        print("\nTo add them:")
        print(f"  python3 pixelfs_builder.py auto-add")

    def auto_add(self):
        """Auto-add all discovered files to PixelFS"""
        if not self.registry:
            print("No file_boot_registry.json found.")
            return

        count = 0
        for file_id_str, entry in self.registry.items():
            name = entry.get('name', 'unknown')
            ftype = entry.get('type', 'data')
            file_id = int(file_id_str)

            # Check if already added
            already_added = any(
                e['file_id'] == file_id
                for e in self.fs["entries"].values()
            )

            if already_added:
                continue

            # Determine path
            if ftype == 'py':
                path = f"/system/{name}"
            elif ftype == 'pxi_module':
                path = f"/apps/{name}"
            elif ftype == 'world':
                path = f"/worlds/{name}"
            elif ftype in ['llm_model', 'model']:
                path = f"/models/{name}"
            elif ftype == 'config':
                path = f"/config/{name}"
            else:
                path = f"/data/{name}"

            # Add to PixelFS
            self.fs["entries"][path] = {
                "file_id": file_id,
                "file_id_hex": f"0x{file_id:08X}",
                "type": ftype,
                "description": entry.get('name', 'Auto-discovered'),
                "pixel": entry.get('pixel', [0, 0, 0, 0]),
                "original_path": entry.get('original_path', 'unknown'),
                "size": entry.get('original_size', 0)
            }

            count += 1

        self._save()
        print(f"✅ Auto-added {count} files to PixelFS")
        self.tree()


def main():
    parser = argparse.ArgumentParser(description="PixelFS Virtual File System Builder")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Init command
    subparsers.add_parser('init', help='Initialize PixelFS')

    # Add command
    add_parser = subparsers.add_parser('add', help='Add a path to PixelFS')
    add_parser.add_argument('path', help='Logical path (e.g., /boot/00_kernel)')
    add_parser.add_argument('file', nargs='?', help='Physical file path')
    add_parser.add_argument('--file-id', type=lambda x: int(x, 16),
                           help='Existing file_id (hex)')
    add_parser.add_argument('--type', default='data',
                           choices=['py', 'pxi_module', 'world', 'model', 'config', 'data'],
                           help='File type')
    add_parser.add_argument('--desc', help='Description')

    # List command
    subparsers.add_parser('list', help='List all PixelFS entries')

    # Tree command
    subparsers.add_parser('tree', help='Show PixelFS as tree')

    # Resolve command
    resolve_parser = subparsers.add_parser('resolve', help='Resolve a path to file_id')
    resolve_parser.add_argument('path', help='Path to resolve')

    # Auto-discover command
    subparsers.add_parser('auto-discover', help='Suggest paths for unpacked files')

    # Auto-add command
    subparsers.add_parser('auto-add', help='Auto-add all discovered files')

    args = parser.parse_args()

    fs = PixelFS()

    if args.command == 'init':
        fs._save()
        print("✅ Initialized PixelFS")

    elif args.command == 'add':
        fs.add_path(args.path, args.file, args.file_id, args.type, args.desc)

    elif args.command == 'list':
        fs.list_all()

    elif args.command == 'tree':
        fs.tree()

    elif args.command == 'resolve':
        entry = fs.resolve(args.path)
        if entry:
            print(f"\n{args.path}:")
            print(f"  File ID:  {entry['file_id_hex']}")
            print(f"  Type:     {entry['type']}")
            print(f"  RGBA:     ({entry['pixel'][0]}, {entry['pixel'][1]}, {entry['pixel'][2]}, {entry['pixel'][3]})")
            print(f"  Size:     {entry.get('size', 0)} bytes")
            print(f"  Desc:     {entry.get('description', 'N/A')}")
        else:
            print(f"❌ Path not found: {args.path}")

    elif args.command == 'auto-discover':
        fs.auto_discover()

    elif args.command == 'auto-add':
        fs.auto_add()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
