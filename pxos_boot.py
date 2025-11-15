#!/usr/bin/env python3
"""
pxOS Universal Bootloader

Boot any pxOS cartridge:
- Normal PXI images (any size)
- God Pixel (1×1 compressed universes)
- Self-extracting archives

Usage:
    pxos_boot.py <image.png>
    pxos_boot.py god.png
    pxos_boot.py --list            # List all God Pixels
    pxos_boot.py --world "name"    # Boot by world name
"""

from pathlib import Path
from PIL import Image
import sys
import json
from pxi_cpu import PXICPU
from god_pixel import GodPixel


class pxOSBootloader:
    """Universal bootloader for pxOS cartridges"""

    def __init__(self):
        self.registry_path = Path("/home/user/pxos/god_pixel_registry.json")

    def detect_format(self, image_path: str) -> str:
        """Detect cartridge format"""
        img = Image.open(image_path)

        if img.size == (1, 1):
            return "god_pixel"
        elif img.size[0] <= 32 and img.size[1] <= 32:
            # Might be self-extracting archive
            r, g, b, a = img.getpixel((0, 0))
            if (r, g, b, a) == (0x4D, 0x45, 0x54, 0x41):  # "META" marker
                return "self_extracting"

        return "raw_pxi"

    def boot(self, image_path: str, verbose: bool = True):
        """Boot a pxOS cartridge"""

        if not Path(image_path).exists():
            print(f"❌ Error: {image_path} not found")
            sys.exit(1)

        format_type = self.detect_format(image_path)

        if verbose:
            print("╔═══════════════════════════════════════════════════════════╗")
            print("║                    pxOS BOOTLOADER                        ║")
            print("╚═══════════════════════════════════════════════════════════╝")
            print()

        if format_type == "god_pixel":
            return self._boot_god_pixel(image_path, verbose)
        elif format_type == "self_extracting":
            return self._boot_self_extracting(image_path, verbose)
        else:
            return self._boot_raw_pxi(image_path, verbose)

    def _boot_god_pixel(self, image_path: str, verbose: bool):
        """Boot from God Pixel (1×1)"""
        if verbose:
            print("🔮 Detected: GOD PIXEL")
            print("    Format: 1×1 compressed universe")
            print()
            print("Resurrecting universe from single pixel...")

        gp = GodPixel()

        try:
            resurrected = gp.resurrect(image_path)

            if verbose:
                print(f"✓ Resurrection complete: {resurrected.size[0]}×{resurrected.size[1]}")
                print()
                print("Booting PXI_CPU...")
                print("─" * 60)
                print()

            cpu = PXICPU(resurrected)
            cpu.run()

            if verbose:
                print()
                print("─" * 60)
                print("✓ Execution complete")

        except Exception as e:
            print(f"❌ Error resurrecting God Pixel: {e}")
            sys.exit(1)

    def _boot_self_extracting(self, image_path: str, verbose: bool):
        """Boot from self-extracting archive"""
        if verbose:
            print("📦 Detected: SELF-EXTRACTING ARCHIVE")
            print(f"    Size: {Image.open(image_path).size}")
            print()
            print("Extracting...")

        from pxi_compress import PXICompressor
        compressor = PXICompressor()

        try:
            extracted = compressor.extract(Image.open(image_path))

            if verbose:
                print(f"✓ Extracted: {extracted.size[0]}×{extracted.size[1]}")
                print()
                print("Booting PXI_CPU...")
                print("─" * 60)
                print()

            cpu = PXICPU(extracted)
            cpu.run()

            if verbose:
                print()
                print("─" * 60)
                print("✓ Execution complete")

        except Exception as e:
            print(f"❌ Error extracting archive: {e}")
            sys.exit(1)

    def _boot_raw_pxi(self, image_path: str, verbose: bool):
        """Boot from raw PXI image"""
        img = Image.open(image_path)

        if verbose:
            print("💾 Detected: RAW PXI CARTRIDGE")
            print(f"    Size: {img.size[0]}×{img.size[1]} pixels")
            print(f"    Total: {img.size[0] * img.size[1]:,} instructions")
            print()
            print("Booting PXI_CPU...")
            print("─" * 60)
            print()

        cpu = PXICPU(img.convert("RGBA"))
        cpu.run()

        if verbose:
            print()
            print("─" * 60)
            print("✓ Execution complete")

    def list_worlds(self):
        """List all registered God Pixel worlds"""
        if not self.registry_path.exists():
            print("No God Pixel worlds found.")
            return

        with open(self.registry_path, 'r') as f:
            registry = json.load(f)

        if not registry:
            print("No God Pixel worlds found.")
            return

        print("╔═══════════════════════════════════════════════════════════╗")
        print("║                  GOD PIXEL ZOO                            ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print()

        for i, (color, meta) in enumerate(registry.items(), 1):
            name = meta.get("name", "Unnamed")
            size = meta.get("original_size", meta.get("size", [0, 0]))
            desc = meta.get("description", "No description")
            method = meta.get("method", "unknown")

            r, g, b, a = [int(x) for x in color.split(",")]

            print(f"{i}. {name}")
            print(f"   Color: RGBA({r}, {g}, {b}, {a}) | #{r:02X}{g:02X}{b:02X}{a:02X}")
            print(f"   Size: {size[0]}×{size[1]} pixels")
            print(f"   Method: {method}")
            print(f"   Description: {desc}")
            print()

    def boot_by_name(self, world_name: str):
        """Boot a world by its name"""
        if not self.registry_path.exists():
            print(f"❌ Error: No registry found")
            sys.exit(1)

        with open(self.registry_path, 'r') as f:
            registry = json.load(f)

        # Find by name
        for color, meta in registry.items():
            if meta.get("name", "").lower() == world_name.lower():
                # Create temporary God Pixel
                r, g, b, a = [int(x) for x in color.split(",")]
                god_img = Image.new("RGBA", (1, 1), (r, g, b, a))
                temp_path = "/tmp/temp_god.png"
                god_img.save(temp_path)

                print(f"Found world: {meta.get('name')}")
                print(f"God Pixel: RGBA({r}, {g}, {b}, {a})")
                print()

                return self.boot(temp_path)

        print(f"❌ Error: World '{world_name}' not found")
        print("\nAvailable worlds:")
        self.list_worlds()
        sys.exit(1)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="pxOS Universal Bootloader - Boot any pixel universe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pxos_boot.py god.png              # Boot God Pixel
  pxos_boot.py program.png          # Boot raw PXI cartridge
  pxos_boot.py --list               # List all worlds
  pxos_boot.py --world "Chat"       # Boot world by name
        """
    )

    parser.add_argument("image", nargs="?", help="PXI image or God Pixel to boot")
    parser.add_argument("--list", "-l", action="store_true", help="List all God Pixel worlds")
    parser.add_argument("--world", "-w", help="Boot world by name")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode (less output)")

    args = parser.parse_args()

    bootloader = pxOSBootloader()

    if args.list:
        bootloader.list_worlds()
    elif args.world:
        bootloader.boot_by_name(args.world)
    elif args.image:
        bootloader.boot(args.image, verbose=not args.quiet)
    else:
        parser.print_help()
        print("\n" + "─" * 60)
        print("Available worlds:")
        print()
        bootloader.list_worlds()


if __name__ == "__main__":
    main()
