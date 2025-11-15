#!/usr/bin/env python3
"""
demo_two_kernels.py - First Multi-Kernel Demo

Demonstrates two kernels running in parallel, both drawing to the
same shared framebuffer. They can see each other's output immediately.

Kernel 1 (Green): Draws a vertical line
Kernel 2 (Red): Draws a diagonal line

This is the foundation of collective evolution - multiple entities
sharing the same world.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pxvm.vm import PxVM
from pxvm.assembler import assemble

# Kernel 1: Draw a green vertical line at x=400
green_kernel = """
    MOV R0, 400         # X position (constant)
    MOV R1, 0           # Y position (starts at 0)
    MOV R2, 0x00FF00    # Green color
    MOV R3, 600         # Maximum Y
    MOV R4, 1           # Increment value

loop:
    PLOT                # Draw pixel at (R0, R1) with color R2
    ADD R1, R4          # Y++ (R1 += 1)
    CMP R1, R3          # Check if Y >= 600
    JZ done             # If equal, stop
    JMP loop            # Otherwise continue

done:
    HALT
"""

# Kernel 2: Draw a red diagonal line
red_kernel = """
    MOV R0, 100         # X position (starts at 100)
    MOV R1, 100         # Y position (starts at 100)
    MOV R2, 0xFF0000    # Red color
    MOV R3, 1           # Increment value
    MOV R4, 500         # Maximum coordinate

loop:
    PLOT                # Draw pixel at (R0, R1)
    ADD R0, R3          # X++
    ADD R1, R3          # Y++
    CMP R0, R4          # Check if X >= 500
    JZ done
    JMP loop

done:
    HALT
"""

def main():
    print("=" * 60)
    print("PHASE 5: COLLECTIVE EVOLUTION - FOUNDATION")
    print("=" * 60)
    print()
    print("Creating multi-kernel VM with shared framebuffer...")

    # Create VM
    vm = PxVM(width=1024, height=1024)

    # Assemble and spawn kernels
    print("Assembling kernel programs...")
    code_green = assemble(green_kernel)
    code_red = assemble(red_kernel)

    print(f"  Green kernel: {len(code_green)} bytes")
    print(f"  Red kernel: {len(code_red)} bytes")

    pid1 = vm.spawn_kernel(code_green, color=0x00FF00)
    pid2 = vm.spawn_kernel(code_red, color=0xFF0000)

    print(f"  Spawned kernel {pid1} (green)")
    print(f"  Spawned kernel {pid2} (red)")
    print()

    print("Running kernels...")
    print("(Note: No graphical display, but framebuffer is being updated)")
    print()

    # Run for several cycles
    for i in range(100):
        vm.step()

        if i % 20 == 0:
            alive = vm.alive_count()
            print(f"  Cycle {vm.cycle:4d}: {alive} kernels alive")

            # Show some stats
            for k in vm.kernels:
                if not k.halted:
                    print(f"    Kernel {k.pid}: PC={k.pc:4d}, R0={k.regs[0]:3d}, R1={k.regs[1]:3d}, cycles={k.cycles}")

        # Stop if all halted
        if vm.alive_count() == 0:
            print()
            print(f"All kernels halted at cycle {vm.cycle}")
            break

    print()
    print("Final state:")
    print(f"  Total cycles: {vm.cycle}")
    print(f"  Kernels alive: {vm.alive_count()}")

    # Count non-black pixels
    import numpy as np
    non_black = np.any(vm.framebuffer > 0, axis=2).sum()
    print(f"  Pixels drawn: {non_black}")

    print()
    print("=" * 60)
    print("SUCCESS: Two kernels ran in parallel!")
    print("=" * 60)
    print()
    print("The framebuffer is shared - each kernel can see the other's work.")
    print("This is the foundation for:")
    print("  - Pheromone communication (Phase 5.1)")
    print("  - Language and glyphs (Phase 6)")
    print("  - Tool-making (Phase 7)")
    print()
    print("The digital biosphere is ready to be born.")

    # Try to save an image if PIL is available
    try:
        from PIL import Image
        img = Image.fromarray(vm.framebuffer, mode='RGB')
        img.save('framebuffer_output.png')
        print()
        print("Framebuffer saved to: framebuffer_output.png")
    except ImportError:
        print()
        print("(Install Pillow to save framebuffer as image: pip install Pillow)")


if __name__ == '__main__':
    main()
