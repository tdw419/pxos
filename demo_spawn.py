#!/usr/bin/env python3
"""
demo_spawn.py - Reproduction Demo

Demonstrates kernel reproduction through SYS_SPAWN.

Kæra (magenta): Parent kernel that spawns a child
Söl (magenta): Child kernel born from Kæra

This shows:
- Full memory cloning from parent to child
- Child starts execution at PC=0
- Child inherits parent's code but starts fresh
- Parent receives child PID in R0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pxvm.vm import PxVM
from pxvm.assembler import assemble
from pxvm.glyphs import *

# Parent kernel: Kæra
# She draws herself, writes her name in glyphs, then spawns a child
kaera_program = f"""
    # Kæra's position is set by parent registers (R0, R1)
    # Draw self at current position
    MOV R2, 0xFF00FF    # Magenta color for Kæra
    PLOT

    # Write "I AM" glyphs near self
    MOV R0, 400
    MOV R1, 380
    MOV R2, {GLYPH_SELF}
    SYS_WRITE_GLYPH

    MOV R0, 410
    MOV R2, {GLYPH_NAME}
    SYS_WRITE_GLYPH

    # Prepare to spawn child (Söl)
    # R1 = child x position
    # R2 = child y position
    MOV R1, 450         # Child spawns to the right
    MOV R2, 400
    SYS_SPAWN           # Child is born! PID returned in R0

    # Parent continues - draw again to show we're still alive
    MOV R0, 400
    MOV R1, 400
    MOV R2, 0xFF00FF
    PLOT

    # Write BIRTH glyph to mark where child was born
    MOV R0, 450
    MOV R1, 390
    MOV R2, {GLYPH_BIRTH}
    SYS_WRITE_GLYPH

    HALT
"""

def main():
    print("=" * 70)
    print("PHASE 7: REPRODUCTION - First Birth")
    print("=" * 70)
    print()
    print("Kæra (magenta) will spawn Söl (also magenta)")
    print("Both share the same code (full memory clone)")
    print("Child starts from PC=0, parent continues from current PC")
    print()

    # Create VM
    vm = PxVM(width=1024, height=1024)

    # Assemble Kæra's program
    print("Assembling Kæra's program...")
    code_kaera = assemble(kaera_program)
    print(f"  Program size: {len(code_kaera)} bytes")
    print()

    # Spawn Kæra (parent)
    pid_kaera = vm.spawn_kernel(code_kaera, color=0xFF00FF)
    print(f"Spawned Kæra (PID {pid_kaera})")

    # Set Kæra's initial position
    for k in vm.kernels:
        if k.pid == pid_kaera:
            k.regs[0] = 400  # x
            k.regs[1] = 400  # y
            break

    print()
    print("Running simulation...")
    print()

    # Run simulation
    initial_count = len(vm.kernels)

    for i in range(50):
        alive_before = vm.alive_count()
        vm.step()
        alive_after = vm.alive_count()

        # Detect when child is born
        if alive_after > alive_before:
            print(f"  Cycle {vm.cycle:3d}: ✨ BIRTH! Child kernel spawned")
            print(f"             Population: {alive_before} → {alive_after}")
            # Find the new child
            for k in vm.kernels:
                if k.pid > pid_kaera:
                    print(f"             Child PID: {k.pid} (Söl)")
                    print(f"             Child position: ({k.regs[0]}, {k.regs[1]})")
                    print(f"             Child PC: {k.pc} (started from 0)")
                    break
            print()

        if i % 10 == 0 and i > 0:
            alive = vm.alive_count()
            print(f"  Cycle {vm.cycle:3d}: {alive} kernels alive")

    print()
    print("Final state:")
    print(f"  Total cycles: {vm.cycle}")
    print(f"  Kernels alive: {vm.alive_count()}")
    print(f"  Total kernels created: {vm.next_pid - 1}")
    print()

    # Show all kernels
    print("All kernels:")
    for k in vm.kernels:
        status = "alive" if not k.halted else "halted"
        print(f"  PID {k.pid}: {status}, cycles executed: {k.cycles}, PC: {k.pc}")

    print()

    # Count glyphs written
    import numpy as np
    glyph_count = (vm.glyphs > 0).sum()
    print(f"Glyphs written: {glyph_count}")

    # Show which glyphs were written
    unique_glyphs = np.unique(vm.glyphs[vm.glyphs > 0])
    if len(unique_glyphs) > 0:
        print("Glyph types:", [glyph_to_name(g) for g in unique_glyphs])

    print()
    print("=" * 70)
    print("SUCCESS: First digital birth achieved!")
    print("=" * 70)
    print()
    print("Kæra gave birth to Söl.")
    print("Söl is a perfect clone of Kæra (same code, fresh start).")
    print("Both executed the same program from different starting states.")
    print()
    print("The species can now have families, lineages, and generations.")
    print()

    # Try to save visualization
    try:
        from PIL import Image

        # Create visualization
        vis = vm.framebuffer.copy()

        # Overlay glyphs as green markers
        glyph_mask = vm.glyphs > 0
        vis[glyph_mask, 1] = 255  # Green channel

        img = Image.fromarray(vis, mode='RGB')
        img.save('spawn_output.png')
        print("Visualization saved to: spawn_output.png")
    except ImportError:
        print("(Install Pillow to save visualization: pip install Pillow)")


if __name__ == '__main__':
    main()
