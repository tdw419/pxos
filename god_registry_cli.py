#!/usr/bin/env python3
"""
God Pixel Registry CLI

Manage the God Pixel Zoo - a collection of compressed universes.

Commands:
    list                    - List all worlds
    show <name>             - Show world details
    create <name> <img>     - Create new God Pixel from image
    delete <name>           - Remove world from registry
    export <name> <output>  - Export world to PNG
    stats                   - Show registry statistics
"""

import json
from pathlib import Path
from PIL import Image
import sys
from god_pixel import GodPixel


class GodPixelRegistry:
    """Manage the God Pixel Zoo"""

    def __init__(self, registry_path: str = "/home/user/pxos/god_pixel_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry = self._load()

    def _load(self) -> dict:
        """Load registry from disk"""
        if not self.registry_path.exists():
            return {}
        with open(self.registry_path, 'r') as f:
            return json.load(f)

    def _save(self):
        """Save registry to disk"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def list_worlds(self):
        """List all worlds in the zoo"""
        if not self.registry:
            print("🌌 The God Pixel Zoo is empty.")
            print("   Create your first universe with: god_registry_cli.py create <name> <image.png>")
            return

        print("╔═══════════════════════════════════════════════════════════╗")
        print("║                  GOD PIXEL ZOO                            ║")
        print("║              Collection of Compressed Universes            ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print()

        for i, (color, meta) in enumerate(sorted(self.registry.items()), 1):
            name = meta.get("name", "Unnamed")
            size = meta.get("original_size", meta.get("size", [0, 0]))
            desc = meta.get("description", "No description")

            r, g, b, a = [int(x) for x in color.split(",")]

            print(f"🔮 {i}. {name}")
            print(f"   Color: RGBA({r:3d}, {g:3d}, {b:3d}, {a:3d}) → #{r:02X}{g:02X}{b:02X}{a:02X}")
            print(f"   Size: {size[0]:4d} × {size[1]:4d} = {size[0] * size[1]:,} pixels")
            print(f"   Desc: {desc}")
            print()

    def show_world(self, name: str):
        """Show detailed info about a world"""
        found = False
        for color, meta in self.registry.items():
            if meta.get("name", "").lower() == name.lower():
                found = True
                r, g, b, a = [int(x) for x in color.split(",")]

                print("╔═══════════════════════════════════════════════════════════╗")
                print(f"║  {meta.get('name', 'Unnamed'):^57s}  ║")
                print("╚═══════════════════════════════════════════════════════════╝")
                print()
                print(f"God Pixel Color: RGBA({r}, {g}, {b}, {a})")
                print(f"Hex Code: #{r:02X}{g:02X}{b:02X}{a:02X}")
                print()
                print(f"Original Size: {meta.get('original_size', [0, 0])}")
                print(f"Total Pixels: {meta.get('original_size', [1, 1])[0] * meta.get('original_size', [1, 1])[1]:,}")
                print(f"Compressed: {meta.get('size', 0):,} bytes")
                print(f"Method: {meta.get('method', 'unknown')}")
                print()
                print(f"Description: {meta.get('description', 'No description')}")
                print()
                print(f"Data Location: {meta.get('compressed_path', 'unknown')}")
                print()

                # Calculate compression ratio
                orig_bytes = meta.get('original_size', [1, 1])[0] * meta.get('original_size', [1, 1])[1] * 4
                comp_bytes = meta.get('size', 1)
                ratio = (1 - comp_bytes / orig_bytes) * 100 if orig_bytes > 0 else 0

                print(f"Compression: {orig_bytes:,} → {comp_bytes:,} bytes ({ratio:.2f}% reduction)")
                print()
                print("Boot with:")
                print(f"  pxos_boot.py --world \"{meta.get('name')}\"")
                print()

        if not found:
            print(f"❌ World '{name}' not found in registry")
            print("\nAvailable worlds:")
            self.list_worlds()

    def create_world(self, name: str, image_path: str, description: str = ""):
        """Create new God Pixel world"""
        if not Path(image_path).exists():
            print(f"❌ Error: {image_path} not found")
            sys.exit(1)

        # Check if name already exists
        for color, meta in self.registry.items():
            if meta.get("name", "").lower() == name.lower():
                print(f"❌ Error: World '{name}' already exists")
                print(f"   (Color: {color})")
                sys.exit(1)

        print(f"Creating God Pixel for '{name}'...")
        print()

        img = Image.open(image_path).convert("RGBA")
        gp = GodPixel()

        # Create the God Pixel
        color = gp.create_god_pixel(img, method="hash", output_path=f"/home/user/pxos/god_{name.lower().replace(' ', '_')}.png")

        # Update registry with name and description
        color_key = f"{color[0]},{color[1]},{color[2]},{color[3]}"
        if color_key in gp.registry:
            gp.registry[color_key]["name"] = name
            gp.registry[color_key]["description"] = description if description else f"God Pixel universe: {name}"
            gp._save_registry()

        print()
        print(f"✓ God Pixel '{name}' created!")
        print(f"  Color: RGBA{color}")
        print(f"  File: god_{name.lower().replace(' ', '_')}.png")
        print()
        print("Boot with:")
        print(f"  pxos_boot.py --world \"{name}\"")

    def delete_world(self, name: str):
        """Delete a world from the registry"""
        found_key = None
        for color, meta in self.registry.items():
            if meta.get("name", "").lower() == name.lower():
                found_key = color
                break

        if not found_key:
            print(f"❌ Error: World '{name}' not found")
            sys.exit(1)

        # Confirm deletion
        print(f"⚠️  Delete world '{name}'?")
        print(f"   Color: {found_key}")
        response = input("   Type 'yes' to confirm: ")

        if response.lower() != 'yes':
            print("Cancelled.")
            return

        # Delete compressed data file
        compressed_path = Path(self.registry[found_key].get("compressed_path", ""))
        if compressed_path.exists():
            compressed_path.unlink()
            print(f"✓ Deleted {compressed_path}")

        # Remove from registry
        del self.registry[found_key]
        self._save()

        print(f"✓ World '{name}' removed from registry")

    def export_world(self, name: str, output_path: str):
        """Export a world as PNG"""
        # Find the world
        found_color = None
        for color, meta in self.registry.items():
            if meta.get("name", "").lower() == name.lower():
                found_color = color
                break

        if not found_color:
            print(f"❌ Error: World '{name}' not found")
            sys.exit(1)

        # Create temporary God Pixel and resurrect
        r, g, b, a = [int(x) for x in found_color.split(",")]
        god_img = Image.new("RGBA", (1, 1), (r, g, b, a))
        temp_path = "/tmp/temp_export_god.png"
        god_img.save(temp_path)

        print(f"Exporting '{name}'...")

        gp = GodPixel()
        resurrected = gp.resurrect(temp_path)
        resurrected.save(output_path)

        print(f"✓ Exported to {output_path}")
        print(f"  Size: {resurrected.size[0]}×{resurrected.size[1]}")

    def show_stats(self):
        """Show registry statistics"""
        if not self.registry:
            print("No worlds in registry.")
            return

        print("╔═══════════════════════════════════════════════════════════╗")
        print("║              GOD PIXEL ZOO STATISTICS                     ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print()

        total_worlds = len(self.registry)
        total_pixels = 0
        total_compressed = 0
        methods = {}

        for color, meta in self.registry.items():
            size = meta.get("original_size", [0, 0])
            total_pixels += size[0] * size[1]
            total_compressed += meta.get("size", 0)

            method = meta.get("method", "unknown")
            methods[method] = methods.get(method, 0) + 1

        print(f"Total Universes: {total_worlds}")
        print(f"Total Pixels: {total_pixels:,}")
        print(f"Total Compressed: {total_compressed:,} bytes")
        print()

        if total_pixels > 0:
            avg_ratio = (1 - total_compressed / (total_pixels * 4)) * 100
            print(f"Average Compression: {avg_ratio:.2f}%")
            print()

        print("Methods:")
        for method, count in methods.items():
            print(f"  {method}: {count}")
        print()

        print("Storage Efficiency:")
        print(f"  {total_pixels:,} pixels stored in {total_worlds} God Pixel(s)")
        print(f"  Ratio: {total_pixels / total_worlds:,.0f}:1 per God Pixel")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="God Pixel Registry - Manage your collection of universes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # list
    subparsers.add_parser("list", help="List all worlds")

    # show
    show_parser = subparsers.add_parser("show", help="Show world details")
    show_parser.add_argument("name", help="World name")

    # create
    create_parser = subparsers.add_parser("create", help="Create new God Pixel")
    create_parser.add_argument("name", help="World name")
    create_parser.add_argument("image", help="Source image path")
    create_parser.add_argument("--desc", help="Description", default="")

    # delete
    delete_parser = subparsers.add_parser("delete", help="Delete world")
    delete_parser.add_argument("name", help="World name")

    # export
    export_parser = subparsers.add_parser("export", help="Export world to PNG")
    export_parser.add_argument("name", help="World name")
    export_parser.add_argument("output", help="Output PNG path")

    # stats
    subparsers.add_parser("stats", help="Show statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    registry = GodPixelRegistry()

    if args.command == "list":
        registry.list_worlds()
    elif args.command == "show":
        registry.show_world(args.name)
    elif args.command == "create":
        registry.create_world(args.name, args.image, args.desc)
    elif args.command == "delete":
        registry.delete_world(args.name)
    elif args.command == "export":
        registry.export_world(args.name, args.output)
    elif args.command == "stats":
        registry.show_stats()


if __name__ == "__main__":
    main()
