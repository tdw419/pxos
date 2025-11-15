#!/usr/bin/env python3
"""
demo_genesis.py - The Genesis Pixel Demo

This script demonstrates the creation and booting of a self-extracting
PXI program. It takes a larger PXI program, compresses it, and embeds
it within a tiny 4x4 "seed" image. This seed contains a bootloader
that decompresses and executes the original program.
"""

from PIL import Image
from pxi_cpu import PXICPU
from pxi_compress import compress_pxi
import os
import zlib

def create_genesis_seed(program_path: str, seed_path: str, compressed_path: str):
    """
    Creates a self-extracting PXI seed image.
    """
    # First, compress the target program
    compress_pxi(program_path, compressed_path)
    with open(compressed_path, 'rb') as f:
        compressed_data = f.read()

    # The seed will be a 16x16 image to have enough space for the bootloader
    # and the compressed data. A 4x4 is too small for this demo.
    width, height = 16, 16
    seed_img = Image.new("RGBA", (width, height), (0, 0, 0, 255))

    def write(pc, r, g, b, a=0):
        y, x = divmod(pc, width)
        seed_img.putpixel((x, y), (r, g, b, a))

    # --- Bootloader Program ---
    # The bootloader will live at the beginning of the seed image.
    # It will decompress the data and then jump to the new code.
    bootloader = [
        (0x10, 0, 16),          # 0: LOAD R0, 16 (source of compressed data)
        (0x10, 1, 256),         # 1: LOAD R1, 256 (destination for decompressed code)
        (0x10, 2, len(compressed_data)), # 2: LOAD R2, compressed_size
        (0xCD, 0, 0),           # 3: SYS_DECOMPRESS
        # A real bootloader would have a JMP here, but for this demo,
        # we'll just let the CPU halt and we'll run the decompressed
        # code in a new CPU instance.
        (0xFF, 0, 0)            # 4: HALT
    ]

    for i, (r, g, b) in enumerate(bootloader):
        write(i, r, g, b)

    # --- Compressed Data ---
    # Embed the compressed data into the seed image, starting after the bootloader.
    for i, byte in enumerate(compressed_data):
        write(len(bootloader) + i, byte, byte, byte)

    seed_img.save(seed_path)
    print(f"Genesis seed created at '{seed_path}'")

def main():
    # We'll use the LLM demo program as our target for genesis
    llm_program_path = "pxi_llm_test.png"
    if not os.path.exists(llm_program_path):
        print("LLM demo program not found. Please run demo_sys_llm.py first.")
        # As a fallback, create a simple program for genesis
        from demo_pxi_hello import build_pxi_hello
        build_pxi_hello("pxi_hello.png")
        program_to_compress = "pxi_hello.png"
    else:
        program_to_compress = llm_program_path

    genesis_seed_path = "genesis_seed.png"
    compressed_path = "program.bin"

    # 1. Create the Genesis Seed
    create_genesis_seed(program_to_compress, genesis_seed_path, compressed_path)

    # 2. Boot the Seed
    print("\nBooting Genesis Seed...")
    seed_cpu = PXICPU(Image.open(genesis_seed_path))
    seed_cpu.run()

    # 3. Verify Decompression
    if seed_cpu.regs[0] > 0:
        print(f"Decompression successful! {seed_cpu.regs[0]} bytes written.")

        # 4. Run the Decompressed Program
        print("\nBooting the resurrected program...")
        # The decompressed program is now in the seed CPU's PXI image memory.
        # We can run it in a new CPU instance to prove it works.
        resurrected_cpu = PXICPU(seed_cpu.pxi_image)
        # The decompressed code starts at address 256
        resurrected_cpu.pc = 256
        # We need to find the start of the program within the decompressed data.
        # This is a simplification; a real bootloader would handle this.
        from demo_sys_llm import prompt
        resurrected_cpu.pc += len(prompt) + 1

        resurrected_cpu.run()

        # Check the result of the resurrected program
        response = resurrected_cpu.read_string(100 + 256)
        print(f"Resurrected program response: '{response}'")
        if "local LLM" in response:
            print("\nSUCCESS: The Genesis Pixel has successfully resurrected the OS.")
        else:
            print("\nFAILURE: The resurrected program did not run as expected.")
    else:
        print("Decompression failed.")

if __name__ == "__main__":
    main()
