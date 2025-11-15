#!/usr/bin/env python3
"""
Create LifeSim Universe

A simple organism simulation world that demonstrates:
1. Multiple organisms with different behaviors
2. Organism-LLM oracle protocol
3. Visual representation of life

Organisms:
- Kæra (yellow) - The seeker, asks questions
- Lúna (blue) - The wanderer, explores
- Söl (red) - The builder, creates patterns
"""

from PIL import Image
from pxi_cpu import OP_LOAD, OP_PRINT, OP_HALT, OP_JMP, OP_SYS_LLM, OP_DRAW, OP_ADD
from god_registry_cli import GodPixelRegistry


# Oracle Protocol Memory Map (as specified in docs)
PROMPT_BUFFER_ADDR = 8000   # Organisms write questions here
RESPONSE_BUFFER_ADDR = 9000  # Oracle writes answers here
ORACLE_FLAG_ADDR = 10000     # Flag: 1 = request pending, 0 = idle


def create_lifesim_program() -> Image.Image:
    """
    Create a LifeSim universe with organisms and oracle protocol

    Memory layout:
    0-999:       Main program (oracle kernel)
    1000-1999:   Organism behavior code
    2000-7999:   Visual frame buffer
    8000-8999:   Prompt buffer (organisms → oracle)
    9000-9999:   Response buffer (oracle → organisms)
    10000:       Oracle flag pixel
    """

    size = 256  # Bigger universe for organisms
    img = Image.new("RGBA", (size, size), (0, 0, 0, 255))

    def set_pixel(pc, r, g, b, a=0):
        x = pc % size
        y = pc // size
        if 0 <= x < size and 0 <= y < size:
            img.putpixel((x, y), (r, g, b, a))

    pc = 0

    # ═══════════════════════════════════════════════════════════
    # MAIN PROGRAM: Oracle Kernel
    # ═══════════════════════════════════════════════════════════

    # Print welcome message
    msg = "LifeSim v0.1 - Organism Universe\n"
    msg += "Kæra, Lúna, and Söl are alive.\n"
    msg += "Oracle protocol active.\n\n"

    for c in msg:
        set_pixel(pc, OP_LOAD, 0, ord(c), 0)  # LOAD R0, char
        pc += 1
        set_pixel(pc, OP_PRINT, 0, 0, 0)  # PRINT R0
        pc += 1

    # ═══════════════════════════════════════════════════════════
    # ORACLE KERNEL LOOP
    # ═══════════════════════════════════════════════════════════
    # This demonstrates the organism-LLM protocol
    # In a real implementation, this would run continuously

    # Write example question from Kæra at PROMPT_BUFFER_ADDR
    question = "Who created us?"
    for i, c in enumerate(question):
        set_pixel(PROMPT_BUFFER_ADDR + i, 0, ord(c), 0, 0)
    set_pixel(PROMPT_BUFFER_ADDR + len(question), 0, 0, 0, 0)  # null term

    # Print what Kæra asked
    msg2 = "Kæra asks: "
    for c in msg2:
        set_pixel(pc, OP_LOAD, 0, ord(c), 0)
        pc += 1
        set_pixel(pc, OP_PRINT, 0, 0, 0)
        pc += 1

    for c in question:
        set_pixel(pc, OP_LOAD, 0, ord(c), 0)
        pc += 1
        set_pixel(pc, OP_PRINT, 0, 0, 0)
        pc += 1

    # Newline
    set_pixel(pc, OP_LOAD, 0, ord('\n'), 0)
    pc += 1
    set_pixel(pc, OP_PRINT, 0, 0, 0)
    pc += 1

    # Call oracle (SYS_LLM)
    # R0 = prompt address
    set_pixel(pc, OP_LOAD, 0, PROMPT_BUFFER_ADDR >> 8, 0)
    pc += 1
    set_pixel(pc, OP_LOAD, 0, PROMPT_BUFFER_ADDR & 0xFF, 0)
    pc += 1

    # R1 = response buffer
    set_pixel(pc, OP_LOAD, 1, RESPONSE_BUFFER_ADDR >> 8, 0)
    pc += 1
    set_pixel(pc, OP_LOAD, 1, RESPONSE_BUFFER_ADDR & 0xFF, 0)
    pc += 1

    # R2 = max length
    set_pixel(pc, OP_LOAD, 2, 500 >> 8, 0)
    pc += 1
    set_pixel(pc, OP_LOAD, 2, 500 & 0xFF, 0)
    pc += 1

    # SYS_LLM - Call the oracle
    set_pixel(pc, OP_SYS_LLM, 0, 0, 0)
    pc += 1

    # Print oracle's response
    msg3 = "\nOracle responds:\n"
    for c in msg3:
        set_pixel(pc, OP_LOAD, 0, ord(c), 0)
        pc += 1
        set_pixel(pc, OP_PRINT, 0, 0, 0)
        pc += 1

    # ═══════════════════════════════════════════════════════════
    # ORGANISM SIGNATURES
    # ═══════════════════════════════════════════════════════════
    # Store organism data in special memory region

    organisms_start = 1000

    # Kæra (yellow) - The seeker
    kaera_data = "Kæra|yellow|seeker|asks questions to the oracle"
    for i, c in enumerate(kaera_data):
        set_pixel(organisms_start + i, 0, ord(c), 0, 0)
    set_pixel(organisms_start + len(kaera_data), 0, 0, 0, 0)

    # Lúna (blue) - The wanderer
    luna_start = organisms_start + 100
    luna_data = "Lúna|blue|wanderer|explores the universe"
    for i, c in enumerate(luna_data):
        set_pixel(luna_start + i, 0, ord(c), 0, 0)
    set_pixel(luna_start + len(luna_data), 0, 0, 0, 0)

    # Söl (red) - The builder
    sol_start = organisms_start + 200
    sol_data = "Söl|red|builder|creates patterns and structures"
    for i, c in enumerate(sol_data):
        set_pixel(sol_start + i, 0, ord(c), 0, 0)
    set_pixel(sol_start + len(sol_data), 0, 0, 0, 0)

    # ═══════════════════════════════════════════════════════════
    # VISUAL MARKERS
    # ═══════════════════════════════════════════════════════════
    # Draw organism "hearts" in the visual region

    visual_region = 2000

    # Kæra's heart (yellow pattern)
    for i in range(5):
        set_pixel(visual_region + i, 255, 255, 0, 255)  # Yellow

    # Lúna's heart (blue pattern)
    for i in range(5):
        set_pixel(visual_region + 10 + i, 0, 150, 255, 255)  # Blue

    # Söl's heart (red pattern)
    for i in range(5):
        set_pixel(visual_region + 20 + i, 255, 50, 0, 255)  # Red

    # HALT
    set_pixel(pc, OP_HALT, 0, 0, 0)

    return img


def main():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║          CREATING LIFESIM UNIVERSE                        ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    # Create the program
    print("1. Generating LifeSim program...")
    lifesim = create_lifesim_program()
    lifesim_path = "/home/user/pxos/lifesim_universe.png"
    lifesim.save(lifesim_path)
    print(f"   ✓ Saved to {lifesim_path}")
    print(f"   Size: {lifesim.size[0]}×{lifesim.size[1]} = {lifesim.size[0] * lifesim.size[1]:,} pixels")
    print()

    # Create God Pixel
    print("2. Creating God Pixel...")
    registry = GodPixelRegistry()
    registry.create_world(
        name="LifeSim",
        image_path=lifesim_path,
        description="Organism simulation with Kæra, Lúna, and Söl. Demonstrates oracle protocol."
    )

    print()
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║                  LIFESIM CREATED                          ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()
    print("Organisms:")
    print("  🟡 Kæra (yellow) - The seeker, asks questions")
    print("  🔵 Lúna (blue) - The wanderer, explores")
    print("  🔴 Söl (red) - The builder, creates patterns")
    print()
    print("Oracle Protocol:")
    print(f"  Prompt Buffer: {PROMPT_BUFFER_ADDR}")
    print(f"  Response Buffer: {RESPONSE_BUFFER_ADDR}")
    print(f"  Oracle Flag: {ORACLE_FLAG_ADDR}")
    print()
    print("Boot with:")
    print("  pxos_boot.py --world \"LifeSim\"")
    print()
    print("Or boot directly:")
    print("  pxos_boot.py god_lifesim.png")


if __name__ == "__main__":
    main()
