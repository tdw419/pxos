#!/usr/bin/env python3
"""
pxi_module_loader.py - Generate pixel-native module loader code

Creates PXI assembly that can load and execute sub-boot pixel modules
entirely within pxOS (no external OS needed).

The loader:
  1. Takes a file_id (from sub-boot pixel RGBA)
  2. Calls SYS_BLOB to load module into memory
  3. Parses PXIM header to find entrypoint
  4. Calls the entrypoint

Usage:
  python3 pxi_module_loader.py --output module_loader.pxi.png
  python3 pxi_module_loader.py --generate-loader-call 0x41E2939A --output load_example.pxi.png
"""

from PIL import Image
import argparse
import struct

# Import opcodes from pxi_cpu
from pxi_cpu import (
    OP_NOP, OP_HALT, OP_LOAD, OP_STORE, OP_ADD, OP_SUB,
    OP_JMP, OP_JNZ, OP_CALL, OP_RET, OP_PUSH, OP_POP,
    OP_DRAW, OP_PRINT, OP_SYS_LLM, OP_SYS_BLOB
)

# Memory layout constants
MODULE_BUFFER_ADDR = 4096   # Where modules are loaded
MODULE_BUFFER_SIZE = 8192   # Max module size
ENTRY_POINT_ADDR   = 12288  # Temp storage for entry point

class PXIModuleLoaderGen:
    """Generates pixel-native PXI code for loading modules"""

    def __init__(self, width=128, height=128):
        self.width = width
        self.height = height
        self.img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        self.pc = 0

    def emit(self, opcode, arg1=0, arg2=0, arg3=0):
        """Emit one instruction pixel"""
        x = self.pc % self.width
        y = self.pc // self.width
        self.img.putpixel((x, y), (opcode, arg1, arg2, arg3))
        self.pc += 1
        return self.pc - 1

    def generate_loader(self):
        """
        Generate the universal module loader

        Expects:
          R4 = file_id (32-bit ID from sub-boot pixel)

        Returns:
          R0 = 0 on error, 1 on success
        """
        print("Generating universal module loader...")

        # Entry point
        start_pc = self.pc

        # Step 1: Call SYS_BLOB to load module
        # R0 = file_id (from R4)
        # R1 = MODULE_BUFFER_ADDR
        # R2 = MODULE_BUFFER_SIZE
        # R3 = 0x01 (decompress flag)

        self.emit(OP_LOAD, 0, R4_LOW := 4, 0)  # R0 = R4 (file_id)
        self.emit(OP_LOAD, 1, (MODULE_BUFFER_ADDR >> 8) & 0xFF, MODULE_BUFFER_ADDR & 0xFF)
        self.emit(OP_LOAD, 2, (MODULE_BUFFER_SIZE >> 8) & 0xFF, MODULE_BUFFER_SIZE & 0xFF)
        self.emit(OP_LOAD, 3, 0x01, 0)  # flags: decompress

        self.emit(OP_SYS_BLOB, 0, 0, 0)  # Call SYS_BLOB

        # Check if load succeeded (R0 > 0)
        error_handler_pc = None  # Will set later
        self.emit(OP_JNZ, 0, 0, 2)  # Jump ahead 2 if R0 != 0 (success)

        # Error: return 0
        error_handler_pc = self.pc
        self.emit(OP_LOAD, 0, 0, 0)  # R0 = 0
        self.emit(OP_RET, 0, 0, 0)

        # Step 2: Parse PXIM header
        # Header format (first 16 bytes / 4 pixels):
        #   Pixel 0: 'P', 'X', 'I', 'M'
        #   Pixel 1: version_major, version_minor, reserved, reserved
        #   Pixel 2: entry_point_low, entry_point_high, num_functions, reserved
        #   Pixel 3: reserved, reserved, reserved, reserved

        # For now, just read entry_point from pixel 2 at MODULE_BUFFER_ADDR + 2
        # entry_point = (pixel[2].G << 8) | pixel[2].R

        # This would require reading from PXI memory, which we don't have an opcode for yet
        # So for this initial version, we'll use a fixed entrypoint convention:
        # Modules always start execution at MODULE_BUFFER_ADDR + 16 (after header)

        # Step 3: Call the module entry point
        entry_point = MODULE_BUFFER_ADDR + 16  # Skip 16-byte header

        self.emit(OP_LOAD, 5, (entry_point >> 8) & 0xFF, entry_point & 0xFF)
        self.emit(OP_CALL, 5, 0, 0)  # CALL R5 (entry point)

        # Step 4: Return success
        self.emit(OP_LOAD, 0, 1, 0)  # R0 = 1 (success)
        self.emit(OP_RET, 0, 0, 0)

        print(f"✅ Generated module loader ({self.pc} instructions)")
        return start_pc

    def generate_loader_call(self, file_id):
        """
        Generate code that loads and runs a specific module

        Args:
            file_id: 32-bit file ID from sub-boot pixel RGBA
        """
        print(f"Generating loader call for file_id 0x{file_id:08X}")

        # Load file_id into R4
        # Since file_id is 32-bit but we can only load 8-bit immediates,
        # we need to build it up in pieces

        # For simplicity, we'll just directly call SYS_BLOB with the file_id
        # R0 = file_id (we'll approximate with lower 16 bits for now)
        # In a real implementation, we'd need multi-byte register loading

        file_id_low = file_id & 0xFFFF
        file_id_high = (file_id >> 16) & 0xFFFF

        # Load low word
        self.emit(OP_LOAD, 0, (file_id_low >> 8) & 0xFF, file_id_low & 0xFF)

        # For now, simplified: just pass the full file_id
        # (This is a limitation of the 8-bit immediate architecture)
        # In practice, we'd store file_ids in a lookup table

        # Call SYS_BLOB
        self.emit(OP_LOAD, 1, (MODULE_BUFFER_ADDR >> 8) & 0xFF, MODULE_BUFFER_ADDR & 0xFF)
        self.emit(OP_LOAD, 2, (MODULE_BUFFER_SIZE >> 8) & 0xFF, MODULE_BUFFER_SIZE & 0xFF)
        self.emit(OP_LOAD, 3, 0x01, 0)  # decompress

        # NOTE: This is simplified - actual implementation would need
        # 32-bit file_id handling via memory lookup table

        self.emit(OP_SYS_BLOB, 0, 0, 0)

        # Execute module (fixed entry point)
        entry_point = MODULE_BUFFER_ADDR + 16
        self.emit(OP_LOAD, 5, (entry_point >> 8) & 0xFF, entry_point & 0xFF)
        self.emit(OP_CALL, 5, 0, 0)

        # Halt
        self.emit(OP_HALT, 0, 0, 0)

        print(f"✅ Generated loader call ({self.pc} instructions)")

    def save(self, output_path):
        """Save the generated loader to a file"""
        self.img.save(output_path)
        print(f"💾 Saved to {output_path}")


class BootPixelDesktop:
    """
    Generate a visual 'desktop' showing all available sub-boot pixels
    that can be clicked/selected to launch modules
    """

    def __init__(self, width=256, height=256):
        self.width = width
        self.height = height
        self.img = Image.new("RGBA", (width, height), (20, 20, 20, 255))

    def add_file_registry(self, registry_path="file_boot_registry.json"):
        """Load file registry and create visual tiles"""
        import json
        from pathlib import Path

        if not Path(registry_path).exists():
            print(f"Registry not found: {registry_path}")
            return

        with open(registry_path, 'r') as f:
            registry = json.load(f)

        print(f"\nCreating desktop with {len(registry)} sub-boot pixels...")

        # Arrange in grid
        tile_size = 32
        tiles_per_row = self.width // tile_size
        x_pos = 0
        y_pos = 0

        for file_id_str, entry in registry.items():
            if x_pos >= tiles_per_row:
                x_pos = 0
                y_pos += 1

            # Get pixel color
            r, g, b, a = entry['pixel']

            # Draw tile (scaled up from 1×1 to tile_size × tile_size)
            for dy in range(tile_size):
                for dx in range(tile_size):
                    px = x_pos * tile_size + dx
                    py = y_pos * tile_size + dy
                    if px < self.width and py < self.height:
                        self.img.putpixel((px, py), (r, g, b, 255))

            # Draw border
            for dx in range(tile_size):
                px = x_pos * tile_size + dx
                py_top = y_pos * tile_size
                py_bot = y_pos * tile_size + tile_size - 1
                if px < self.width:
                    if py_top < self.height:
                        self.img.putpixel((px, py_top), (255, 255, 255, 255))
                    if py_bot < self.height:
                        self.img.putpixel((px, py_bot), (255, 255, 255, 255))

            for dy in range(tile_size):
                px_left = x_pos * tile_size
                px_right = x_pos * tile_size + tile_size - 1
                py = y_pos * tile_size + dy
                if py < self.height:
                    if px_left < self.width:
                        self.img.putpixel((px_left, py), (255, 255, 255, 255))
                    if px_right < self.width:
                        self.img.putpixel((px_right, py), (255, 255, 255, 255))

            print(f"  [{x_pos},{y_pos}] {entry['name']}: RGBA({r},{g},{b},{a})")

            x_pos += 1

    def save(self, output_path):
        """Save desktop image"""
        self.img.save(output_path)
        print(f"💾 Desktop saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate PXI module loader code")
    parser.add_argument('--output', default='module_loader.pxi.png',
                       help='Output PXI image file')
    parser.add_argument('--loader', action='store_true',
                       help='Generate universal loader')
    parser.add_argument('--load-module', type=lambda x: int(x, 16),
                       help='Generate code to load specific module (hex file_id)')
    parser.add_argument('--desktop', action='store_true',
                       help='Generate visual desktop of sub-boot pixels')

    args = parser.parse_args()

    if args.desktop:
        desktop = BootPixelDesktop()
        desktop.add_file_registry()
        desktop.save(args.output)

    elif args.load_module:
        gen = PXIModuleLoaderGen()
        gen.generate_loader_call(args.load_module)
        gen.save(args.output)

    elif args.loader:
        gen = PXIModuleLoaderGen()
        gen.generate_loader()
        gen.save(args.output)

    else:
        print("Please specify --loader, --load-module FILE_ID, or --desktop")
        parser.print_help()


if __name__ == "__main__":
    main()
