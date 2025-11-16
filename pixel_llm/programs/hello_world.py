#!/usr/bin/env python3
"""
Hello World - First Pixel VM Program

This script creates a pixel VM bytecode program and stores it as a .pxi file.
The program prints "Hello from pixels!" and computes 2 + 2.

THIS IS THE MOMENT:
  Code is not Python.
  Code is not text.
  Code is bytecode stored as pixels.
"""

import sys
from pathlib import Path

# Bootstrap
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pixel_llm.core.pixel_vm import PixelVM, assemble_program
from pixel_llm.core.pixelfs import PixelFS


def create_hello_world_program():
    """Create a simple Hello World program"""
    program = assemble_program([
        # Print "Hello from pixels!"
        (PixelVM.OP_PRINT_STR, "Hello from pixels!"),

        # Compute 2 + 2
        (PixelVM.OP_PUSH, 2),
        (PixelVM.OP_PUSH, 2),
        (PixelVM.OP_ADD,),

        # Print result
        (PixelVM.OP_PRINT,),

        # Halt
        (PixelVM.OP_HALT,),
    ])

    return program


def main():
    print("=" * 70)
    print("CREATING FIRST PIXEL VM PROGRAM")
    print("=" * 70)
    print()

    # Create program
    program = create_hello_world_program()

    print(f"Program size: {len(program)} bytes")
    print(f"Bytecode: {program.hex()}")
    print()

    # Save to pixel image
    output_path = Path(__file__).parent / "hello_world.pxi"

    fs = PixelFS()
    fs.write(str(output_path), program)

    print(f"✅ Saved to: {output_path}")
    print()

    # Test execution
    print("Testing execution...")
    print()

    vm = PixelVM(debug=False)
    vm.load_from_pixel(str(output_path))
    vm.run()

    print()
    print("=" * 70)
    print("SUCCESS: First pixel-native program executed!")
    print("=" * 70)
    print()
    print("🎯 WHAT THIS MEANS:")
    print()
    print("  ✅ Bytecode stored as pixel image (.pxi)")
    print("  ✅ Loaded from pixels (not text files)")
    print("  ✅ Executed by pixel VM")
    print("  ✅ Python is just the runtime")
    print()
    print("This is substrate-native code execution.")
    print("The code IS pixels.")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
