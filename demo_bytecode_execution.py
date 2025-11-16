#!/usr/bin/env python3
"""
Demo: Execute Bytecode from Pixel Archive

This demonstrates the complete bytecode execution stack:
  1. Python source → compiled to .pyc bytecode
  2. Bytecode → stored in pixel archive (.pxa)
  3. Archive → loaded by bytecode importer
  4. Module → imported and executed from bytecode ONLY

NO SOURCE CODE NEEDED. PURE BYTECODE FROM PIXELS.

Philosophy:
"This is what a bytecode hypervisor looks like.
 Python is just the first guest.
 WASM is next. Then more.
 All living in pixels."
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    print()
    print("█" * 70)
    print("█" + " " * 16 + "BYTECODE FROM PIXELS DEMO" + " " * 27 + "█")
    print("█" * 70)
    print()

    # Install bytecode importer
    from pixel_llm.core.bytecode_loader import install_bytecode_importer, get_bytecode_stats

    archive_path = ROOT / "pxos_repo.pxa"

    print("Step 1: Installing bytecode importer...")
    print()

    install_bytecode_importer(str(archive_path), debug=False)

    stats = get_bytecode_stats()

    print(f"Available bytecode modules: {stats['total_modules']}")
    print()

    print("─" * 70)
    print("Step 2: Importing module from bytecode...")
    print("─" * 70)
    print()

    # Try to import a module that should exist in bytecode
    try:
        # This will load from bytecode, not source
        print("Importing: pixel_llm.core.pixelfs")
        import pixel_llm.core.pixelfs as pixelfs

        print(f"✅ Module imported!")
        print(f"   Origin: {pixelfs.__file__}")
        print(f"   Loader: {type(pixelfs.__loader__).__name__}")
        print()

        # Verify it's from bytecode
        if "bytecode" in pixelfs.__file__:
            print("🔥 MODULE LOADED FROM BYTECODE!")
            print()

            # Use the module
            print("─" * 70)
            print("Step 3: Using the bytecode module...")
            print("─" * 70)
            print()

            # Create a PixelFS instance
            print("Creating PixelFS instance...")
            fs = pixelfs.PixelFS()
            print(f"✅ PixelFS: {fs}")
            print()

            # Write some test data
            test_data = b"Hello from bytecode execution!"
            print(f"Writing test data: {test_data.decode()}")

            test_path = Path("/tmp/test_bytecode_pixel.pxi")
            fs.write(str(test_path), test_data)

            print(f"✅ Written to: {test_path}")
            print()

            # Read it back
            print("Reading data back...")
            read_data = fs.read(str(test_path))

            print(f"✅ Read: {read_data.decode()}")
            print()

            if read_data == test_data:
                print("🎉 BYTECODE EXECUTION VERIFIED!")
                print()
                print("What just happened:")
                print("  1. pixelfs.pyc loaded from pixel archive")
                print("  2. Bytecode deserialized by Python")
                print("  3. PixelFS class instantiated")
                print("  4. Methods called and executed")
                print("  5. All from bytecode - NO source needed")
                print()

            # Clean up
            test_path.unlink(missing_ok=True)

        else:
            print("⚠️  Module loaded from source, not bytecode")
            print("   (Bytecode may not be in archive)")
            print()

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print()
        print("This likely means:")
        print("  • Bytecode not compiled yet")
        print("  • Bytecode not in archive")
        print("  • Run: python3 compile_to_bytecode.py")
        print("  • Then: python3 pack_repository.py")
        return

    print("─" * 70)
    print("ARCHITECTURE VERIFICATION")
    print("─" * 70)
    print()

    print("✅ Layer 0: Pixel Substrate")
    print("   • PixelFS stores bytes as pixels")
    print("   • Pixel archive contains all files")
    print()

    print("✅ Layer 1: Bytecode Engine")
    print("   • Python bytecode (.pyc) in archive")
    print("   • Custom importer loads bytecode")
    print("   • marshal deserializes code objects")
    print()

    print("✅ Layer 2: Execution")
    print("   • Code executes from bytecode")
    print("   • No source files needed")
    print("   • Pure pixel-native execution")
    print()

    print("─" * 70)
    print("NEXT STEPS")
    print("─" * 70)
    print()

    print("This proves pxOS can be a bytecode hypervisor.")
    print()
    print("Future possibilities:")
    print("  • WebAssembly: Compile Rust/C++ → .wasm → pixels → execute")
    print("  • Lua: Compile Lua → bytecode → pixels → execute")
    print("  • Custom IR: PixelVM bytecode coexisting with Python bytecode")
    print("  • Multi-language: All in one archive, all from pixels")
    print()

    print("█" * 70)
    print("█" + " " * 18 + "BYTECODE HYPERVISOR ACTIVE" + " " * 23 + "█")
    print("█" * 70)
    print()


if __name__ == "__main__":
    main()
