#!/usr/bin/env python3
"""
Complete Demo: God Pixel → PXI_CPU → Local LLM

Demonstrates the full stack:
1. Create program that talks to LLM
2. Compress to God Pixel
3. Resurrect and run
4. Communicate with local LLM (LM Studio / Ollama)
"""

from PIL import Image
from pxi_cpu import PXICPU, OP_LOAD, OP_PRINT, OP_HALT, OP_SYS_LLM
from god_pixel import GodPixel


def create_llm_program() -> Image.Image:
    """
    Create a program that:
    1. Prints "Asking LLM: Who are you?"
    2. Calls SYS_LLM with prompt
    3. Prints response
    """

    size = 128
    img = Image.new("RGBA", (size, size), (0, 0, 0, 255))

    def set_pixel(pc, r, g, b, a=0):
        x = pc % size
        y = pc // size
        img.putpixel((x, y), (r, g, b, a))

    pc = 0

    # Store prompt at address 1000: "Who are you?"
    prompt = "Who are you?"
    prompt_addr = 1000
    for i, c in enumerate(prompt):
        set_pixel(prompt_addr + i, 0, ord(c), 0, 0)  # G channel = ASCII
    set_pixel(prompt_addr + len(prompt), 0, 0, 0, 0)  # Null terminator

    # Program starts at PC=0
    # Print header
    msg = "Asking LLM..."
    for i, c in enumerate(msg):
        set_pixel(pc, OP_LOAD, 0, ord(c), 0)  # LOAD R0, char
        pc += 1
        set_pixel(pc, OP_PRINT, 0, 0, 0)  # PRINT R0
        pc += 1

    # Newline
    set_pixel(pc, OP_LOAD, 0, ord('\n'), 0)
    pc += 1
    set_pixel(pc, OP_PRINT, 0, 0, 0)
    pc += 1

    # Call SYS_LLM
    # R0 = prompt address
    set_pixel(pc, OP_LOAD, 0, prompt_addr >> 8, 0)
    pc += 1
    set_pixel(pc, OP_LOAD, 0, prompt_addr & 0xFF, 0)
    pc += 1

    # R1 = output buffer address (2000)
    output_addr = 2000
    set_pixel(pc, OP_LOAD, 1, output_addr >> 8, 0)
    pc += 1
    set_pixel(pc, OP_LOAD, 1, output_addr & 0xFF, 0)
    pc += 1

    # R2 = max length (500)
    set_pixel(pc, OP_LOAD, 2, 500 >> 8, 0)
    pc += 1
    set_pixel(pc, OP_LOAD, 2, 500 & 0xFF, 0)
    pc += 1

    # SYS_LLM
    set_pixel(pc, OP_SYS_LLM, 0, 0, 0)
    pc += 1

    # Print response header
    msg2 = "\nLLM Response: "
    for i, c in enumerate(msg2):
        set_pixel(pc, OP_LOAD, 0, ord(c), 0)
        pc += 1
        set_pixel(pc, OP_PRINT, 0, 0, 0)
        pc += 1

    # HALT
    set_pixel(pc, OP_HALT, 0, 0, 0)

    return img


def main():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║         GOD PIXEL → PXI_CPU → LOCAL LLM DEMO             ║")
    print("╚═══════════════════════════════════════════════════════════╝")

    # Step 1: Create program that talks to LLM
    print("\n" + "="*60)
    print("STEP 1: Creating LLM program")
    print("="*60)

    llm_program = create_llm_program()
    llm_program.save("/home/user/pxos/llm_program.png")
    print(f"✓ Created program: {llm_program.size[0] * llm_program.size[1]:,} pixels")

    # Step 2: Compress to God Pixel
    print("\n" + "="*60)
    print("STEP 2: Compressing to God Pixel")
    print("="*60)

    gp = GodPixel()
    god_color = gp.create_god_pixel(
        llm_program,
        method="hash",
        output_path="/home/user/pxos/llm_god.png"
    )

    print(f"\n✓ God Pixel created: RGBA{god_color}")
    print(f"  16,384 pixels compressed into 1 pixel!")

    # Step 3: Resurrect
    print("\n" + "="*60)
    print("STEP 3: Resurrecting program from God Pixel")
    print("="*60)

    resurrected = gp.resurrect("/home/user/pxos/llm_god.png")
    print(f"✓ Program resurrected: {resurrected.size[0] * resurrected.size[1]:,} pixels")

    # Verify perfect reconstruction
    if list(resurrected.getdata()) == list(llm_program.getdata()):
        print("✓ Perfect reconstruction confirmed")
    else:
        print("✗ ERROR: Reconstruction failed!")
        return

    # Step 4: Run on PXI_CPU
    print("\n" + "="*60)
    print("STEP 4: Running resurrected program on PXI_CPU")
    print("="*60)
    print("\nNOTE: This will attempt to connect to local LLM")
    print("      (LM Studio on port 1234 or Ollama on port 11434)")
    print("      If not running, you'll see an error message.\n")

    input("Press Enter to run the program (or Ctrl+C to cancel)...")

    cpu = PXICPU(resurrected)
    steps = cpu.run(max_steps=10000)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Created program: 16,384 pixels")
    print(f"✓ Compressed to: 1 God Pixel")
    print(f"✓ Resurrected perfectly")
    print(f"✓ Executed {steps} instructions")
    print(f"\nThe God Pixel is alive.")
    print(f"It speaks to your local LLM.")
    print(f"The loop is closed:")
    print(f"  Human → God Pixel → PXI_CPU → LLM → Human")


if __name__ == "__main__":
    main()
