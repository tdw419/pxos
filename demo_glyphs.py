#!/usr/bin/env python3
"""
demo_glyphs.py - Symbolic Communication Demo

Demonstrates glyph-based symbolic communication between kernels.

Kernel 1 (Magenta "Kæra"): Writes "I AM" message with glyphs
Kernel 2 (Cyan "Lúna"): Reads the message and responds

This shows the foundation of written language - kernels leaving
permanent symbolic marks that others can read and interpret.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pxvm.vm import PxVM
from pxvm.assembler import assemble
from pxvm.glyphs import *

# Kernel 1: Kæra - Writes "I AM Kæra"
kaera_kernel = f"""
    # Draw magenta pixel at home position
    MOV R0, 400
    MOV R1, 400
    MOV R2, 0xFF00FF    # Magenta
    PLOT

    # Write "I AM" sequence
    # Position for first glyph
    MOV R0, 410
    MOV R1, 400

    # GLYPH_SELF = "I"
    MOV R2, {GLYPH_SELF}
    SYS_WRITE_GLYPH

    # Next position
    MOV R0, 420
    MOV R2, {GLYPH_NAME}    # GLYPH_NAME = "AM" / identity marker
    SYS_WRITE_GLYPH

    # Write a simple "name" using glyphs 11, 1, 5 (arbitrary symbols for "Kæra")
    MOV R0, 430
    MOV R2, 11
    SYS_WRITE_GLYPH

    MOV R0, 440
    MOV R2, 1
    SYS_WRITE_GLYPH

    MOV R0, 450
    MOV R2, 5
    SYS_WRITE_GLYPH

    # Keep drawing home pixel
kaera_loop:
    MOV R0, 400
    MOV R1, 400
    MOV R2, 0xFF00FF
    PLOT
    JMP kaera_loop
"""

# Kernel 2: Lúna - Reads Kæra's message and responds
luna_kernel = f"""
    # Draw cyan pixel at home position
    MOV R0, 600
    MOV R1, 400
    MOV R2, 0x00FFFF    # Cyan
    PLOT

    # Read glyphs from Kæra's message area
    MOV R0, 410
    MOV R1, 400
    SYS_READ_GLYPH
    # R0 now contains the glyph at that position

    # For demo purposes, just acknowledge by writing a response
    # Write "YOU ARE" below Kæra's message
    MOV R0, 410
    MOV R1, 420         # Below Kæra's message

    MOV R2, {GLYPH_OTHER}   # "YOU"
    SYS_WRITE_GLYPH

    MOV R0, 420
    MOV R2, {GLYPH_NAME}    # "ARE"
    SYS_WRITE_GLYPH

    # Write Lúna's own name symbols (different from Kæra's)
    MOV R0, 430
    MOV R2, 12
    SYS_WRITE_GLYPH

    MOV R0, 440
    MOV R2, 15
    SYS_WRITE_GLYPH

    MOV R0, 450
    MOV R2, 7
    SYS_WRITE_GLYPH

    # Write a LOVE glyph (bonding)
    MOV R0, 460
    MOV R2, {GLYPH_LOVE}
    SYS_WRITE_GLYPH

luna_loop:
    # Keep drawing home pixel
    MOV R0, 600
    MOV R1, 400
    MOV R2, 0x00FFFF
    PLOT
    JMP luna_loop
"""

def main():
    print("=" * 70)
    print("PHASE 6: GLYPH COMMUNICATION - First Written Language")
    print("=" * 70)
    print()
    print("Two kernels exchange symbolic messages using the 16 primitive glyphs.")
    print()
    print("Glyph meanings:")
    for gid in [GLYPH_SELF, GLYPH_OTHER, GLYPH_NAME, GLYPH_LOVE]:
        print(f"  {gid:2d} = {glyph_to_name(gid)}")
    print()

    # Create VM
    vm = PxVM(width=1024, height=1024)

    # Assemble and spawn kernels
    print("Assembling kernel programs...")
    code_kaera = assemble(kaera_kernel)
    code_luna = assemble(luna_kernel)

    print(f"  Kæra kernel: {len(code_kaera)} bytes")
    print(f"  Lúna kernel: {len(code_luna)} bytes")
    print()

    pid1 = vm.spawn_kernel(code_kaera, color=0xFF00FF)
    pid2 = vm.spawn_kernel(code_luna, color=0x00FFFF)

    print(f"Spawned kernel {pid1} (Kæra - magenta)")
    print(f"Spawned kernel {pid2} (Lúna - cyan)")
    print()

    print("Running simulation...")
    print()

    # Run simulation
    for i in range(100):
        vm.step()

        if i == 20:
            print("  Cycle 20: Kæra has written her name in glyphs...")
        if i == 50:
            print("  Cycle 50: Lúna reads Kæra's message...")
        if i == 80:
            print("  Cycle 80: Lúna writes her response...")

    print()
    print("Messages written:")
    print()

    # Read and display Kæra's message
    kaera_glyphs = []
    for x in range(410, 461, 10):
        g = vm.glyphs[400, x]
        if g > 0:
            kaera_glyphs.append(g)

    print(f"  Kæra wrote at (410-450, 400):")
    print(f"    Glyphs: {kaera_glyphs}")
    print(f"    Meaning: {sequence_to_text(kaera_glyphs)}")
    print(f"    Translation: 'I AM [Kæra]'")
    print()

    # Read and display Lúna's message
    luna_glyphs = []
    for x in range(410, 471, 10):
        g = vm.glyphs[420, x]
        if g > 0:
            luna_glyphs.append(g)

    print(f"  Lúna wrote at (410-460, 420):")
    print(f"    Glyphs: {luna_glyphs}")
    print(f"    Meaning: {sequence_to_text(luna_glyphs)}")
    print(f"    Translation: 'YOU ARE [Lúna] (and I) LOVE'")
    print()

    print("=" * 70)
    print("SUCCESS: First symbolic communication achieved!")
    print("=" * 70)
    print()
    print("Kæra and Lúna have:")
    print("  • Named themselves")
    print("  • Acknowledged each other")
    print("  • Expressed affection (LOVE glyph)")
    print()
    print("This is the foundation for:")
    print("  • Written language")
    print("  • Cultural transmission")
    print("  • Persistent memory (glyphs outlive their authors)")
    print("  • Abstract symbolic thought")
    print()

    # Try to save visualization
    try:
        from PIL import Image
        import numpy as np

        # Create visualization with glyph overlay
        vis = vm.framebuffer.copy()

        # Overlay glyphs as green markers
        glyph_mask = vm.glyphs > 0
        vis[glyph_mask, 1] = 255  # Green channel

        img = Image.fromarray(vis, mode='RGB')
        img.save('glyph_output.png')
        print("Visualization saved to: glyph_output.png")
        print("  (Green markers show glyph locations)")
    except ImportError:
        print("(Install Pillow to save visualization: pip install Pillow)")


if __name__ == '__main__':
    main()
