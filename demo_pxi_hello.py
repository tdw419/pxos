#!/usr/bin/env python3
"""
demo_pxi_hello.py - First pixel-boot demo for pxOS.

Program encoded into pxi_hello.png:

  0: LOAD R0, 8        (set x = 8)
  1: LOAD R1, 8        (set y = 8)
  2: DRAW (yellow)     (draw pixel at (8,8) with color #FFFF00)
  3: HALT

All other pixels = NOP.

Then we run the pixel CPU and produce pxi_hello_frame.png.
"""

from PIL import Image
from pxi_cpu import PXICPU


def build_pxi_hello(path: str, width: int = 16, height: int = 16):
    img = Image.new("RGBA", (width, height), (0x00, 0x00, 0x00, 0xFF))  # default NOP

    # Helper to write one instruction pixel by linear PC index
    def write_instr(pc: int, rgba):
        y, x = divmod(pc, width)  # row, col
        img.putpixel((x, y), rgba)

    # Opcodes:
    # 0x00 = NOP (default)
    # 0x10 = LOAD R[G] = B
    # 0x60 = DRAW at (R0,R1) with color (G,B,A)
    # 0xFF = HALT

    # 0: LOAD R0, 8
    write_instr(0, (0x10, 0, 8, 0))

    # 1: LOAD R1, 8
    write_instr(1, (0x10, 1, 8, 0))

    # 2: DRAW yellow (#FFFF00) at (R0,R1)
    write_instr(2, (0x60, 0xFF, 0xFF, 0x00))

    # 3: HALT
    write_instr(3, (0xFF, 0, 0, 0))

    img.save(path)
    return img


def main():
    pxi_path = "pxi_hello.png"
    out_frame_path = "pxi_hello_frame.png"

    print(f"Building PXI program → {pxi_path}")
    build_pxi_hello(pxi_path)

    print("Booting pixel CPU…")
    cpu = PXICPU.run_file(pxi_path, out_frame_path)

    print("Done.")
    print(f"PC halted at: {cpu.pc}")
    print(f"Registers: {cpu.regs}")
    print(f"Output frame saved to: {out_frame_path}")
    print("You should see a single yellow pixel at (8,8).")


if __name__ == "__main__":
    main()
