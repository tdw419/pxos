#!/usr/bin/env python3
from pathlib import Path
from PIL import Image
from pxi_cpu import PXICPU

def boot(image_path: str):
    """
    Boots a pxOS program, which can be either a raw PXI image
    or a 1x1 God Pixel.
    """
    try:
        img = Image.open(image_path).convert("RGBA")
        cpu = PXICPU(img)
        cpu.run()

        # Save the final frame for inspection
        output_path = Path(image_path).stem + "_frame.png"
        cpu.frame.save(output_path)
        print(f"Execution complete. Output frame saved to '{output_path}'")

    except FileNotFoundError:
        print(f"Error: Boot image not found at '{image_path}'")
    except Exception as e:
        print(f"An error occurred during boot: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: pxos_boot.py <path_to_pxi_or_god_pixel.png>")
        sys.exit(1)

    main(sys.argv)

def main(argv):
    if len(argv) < 2:
        print("Usage: pxos_boot.py <path_to_pxi_or_god_pixel.png>")
        return
    boot(argv[1])
