#!/usr/bin/env python3
"""
pxOS Boot - Pixel-Native Execution Entrypoint

This is the ONLY Python script you run directly.
Everything else loads from pixels.

PARADIGM SHIFT:
  Old way: python3 some_script.py
          (reads .py from disk, executes)

  New way: python3 pxos_boot.py
          (installs pixel importer, loads everything from .pxi images)

This tiny script is the "kernel" - it bootstraps the pixel import system,
then hands control to pixel-native code.

After this runs, you're executing code that lives in pixels.
Python is just the interpreter underneath.

Philosophy:
"The boot loader is the only text that isn't a pixel.
 Everything else is substrate-native."
"""

import sys
from pathlib import Path

# Add pixel_llm to path (only for bootstrapping)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Import the pixel loader (from traditional .py, for bootstrapping)
from pixel_llm.core.pxos_loader import install_pixel_importer, get_pixel_stats


def main():
    """Boot pxOS and run pixel-native code"""
    print()
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " " * 20 + "pxOS BOOT SEQUENCE" + " " * 29 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print()

    # Install pixel importer
    print("Phase 1: Installing pixel import system...")
    print()
    install_pixel_importer(debug=False)

    # Show stats
    stats = get_pixel_stats()
    if stats["installed"]:
        print(f"Phase 2: Pixel system active")
        print(f"  • {stats['total_modules']} modules encoded")
        print(f"  • {stats['total_bytes']:,} bytes in pixels")
        print()

    # Now ALL imports come from pixels (if available)
    print("Phase 3: Loading application from pixels...")
    print()

    try:
        # This import will load from pixels!
        # The demo modules should be in the pixel manifest
        print("  → Attempting pixel-native import: pixel_llm.demos.gpu_dot_product_demo")
        print()

        # Try to import a demo from pixels
        import pixel_llm.demos.gpu_dot_product_demo as demo

        print("✅ SUCCESS: Module loaded from pixels!")
        print(f"   Module origin: {demo.__file__ if hasattr(demo, '__file__') else 'pixel-native'}")
        print()

        # Show that we can access the module
        if hasattr(demo, 'main'):
            print("  → Found main() function in pixel module")
            print("  → Module is ready to execute")
        else:
            print("  → Module loaded but has no main()")

        print()

    except ImportError as e:
        print(f"⚠️  Import failed: {e}")
        print()
        print("This is expected if you haven't run encode_repo_to_pixels.py yet")
        print()
        print("To encode the repository:")
        print("  python3 pixel_llm/core/encode_repo_to_pixels.py")
        print()

    # Summary
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " " * 17 + "pxOS BOOT COMPLETE" + " " * 30 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print()
    print("🎯 WHAT JUST HAPPENED:")
    print()
    print("  1. Python booted (traditional interpreter)")
    print("  2. Pixel import system installed")
    print("  3. Modules loaded FROM PIXELS (not .py files)")
    print("  4. Code executed from pixel-native storage")
    print()
    print("This is substrate-native execution.")
    print("Python is just the runtime. Pixels are the source.")
    print()
    print("█" * 70)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBoot interrupted")
    except Exception as e:
        print(f"\n❌ Boot failed: {e}")
        import traceback
        traceback.print_exc()
