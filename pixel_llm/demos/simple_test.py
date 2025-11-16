#!/usr/bin/env python3
"""
Simple Test Demo

A minimal program to demonstrate self-contained execution from pixels.

This is loaded and executed entirely from the pixel archive.
"""


def main():
    """Main entrypoint"""
    print("=" * 70)
    print("SIMPLE TEST DEMO")
    print("=" * 70)
    print()

    print("✅ This code is running from the pixel archive!")
    print()

    # Test importing another module from the archive
    print("Testing cross-module imports...")
    from pixel_llm.core.pixel_vm import PixelVM

    print(f"✅ Imported PixelVM: {PixelVM}")
    print(f"   Origin: {PixelVM.__module__}")
    print()

    # Create a simple VM instance
    print("Creating PixelVM instance...")
    vm = PixelVM(debug=False)

    print(f"✅ Created VM: {vm}")
    print(f"   Stack size: {len(vm.stack)}")
    print(f"   Memory size: {len(vm.memory)}")
    print()

    # Run a tiny program
    print("Running simple bytecode program...")
    print()

    # Simple program: PUSH 42, PRINT, HALT
    import struct
    program = bytearray()
    program.append(0x01)  # PUSH
    program.extend(struct.pack('<i', 42))
    program.append(0x10)  # PRINT
    program.append(0xFF)  # HALT

    vm.load_program(bytes(program))
    vm.run()

    print()
    print("=" * 70)
    print("✅ SELF-CONTAINED EXECUTION SUCCESSFUL")
    print("=" * 70)
    print()

    return 0


if __name__ == "__main__":
    main()
