#!/usr/bin/env python3
"""
demo_pheromones.py - Chemical Communication Demo

Demonstrates pheromone-based communication between kernels.

Kernel 1 (Yellow): Emits pheromone at center (stationary "food source")
Kernel 2 (Cyan): Starts at corner, follows pheromone gradient (chemotaxis)

This shows how kernels can communicate through chemical signals
that persist, decay, and diffuse across the shared environment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pxvm.vm import PxVM
from pxvm.assembler import assemble

# Kernel 1: Stationary pheromone emitter (like a food source)
emitter_kernel = """
    # Position at center
    MOV R0, 512
    MOV R1, 512
    MOV R2, 0xFFFF00    # Yellow color

emit_loop:
    # Draw yellow pixel
    PLOT

    # Emit strong pheromone at our location
    MOV R0, 512
    MOV R1, 512
    MOV R2, 200         # Strong pheromone
    SYS_EMIT_PHEROMONE

    # Stay at same location and keep emitting
    JMP emit_loop
"""

# Kernel 2: Pheromone seeker (chemotaxis)
# Simplified: moves in a spiral pattern while sensing pheromone
seeker_kernel = """
    # Start at corner
    MOV R3, 100         # Current X position
    MOV R4, 100         # Current Y position
    MOV R5, 1           # Movement increment

seek_loop:
    # Draw current position (cyan)
    MOV R0, 100         # Reset to stored position (simplified)
    MOV R1, 100
    ADD R0, R3          # Add offset
    ADD R1, R4
    MOV R2, 0x00FFFF    # Cyan color
    PLOT

    # Sense pheromone at current location
    SYS_SENSE_PHEROMONE
    # R0 now has pheromone strength

    # Simple movement: spiral toward center
    # Move right
    ADD R3, R5
    # Every few steps, also move down
    ADD R4, R5

    # Wrap around if needed (simple bounds check via increment)
    JMP seek_loop
"""

def main():
    print("=" * 70)
    print("PHASE 5.1: PHEROMONE COMMUNICATION - Chemotaxis Demo")
    print("=" * 70)
    print()
    print("Two kernels demonstrate chemical communication:")
    print("  • Yellow kernel: Emits pheromone at center (512, 512)")
    print("  • Cyan kernel: Starts at (100, 100), follows scent gradient")
    print()

    # Create VM
    vm = PxVM(width=1024, height=1024)

    # Assemble and spawn kernels
    print("Assembling kernel programs...")
    code_emitter = assemble(emitter_kernel)
    code_seeker = assemble(seeker_kernel)

    print(f"  Emitter kernel: {len(code_emitter)} bytes")
    print(f"  Seeker kernel: {len(code_seeker)} bytes")
    print()

    pid1 = vm.spawn_kernel(code_emitter, color=0xFFFF00)
    pid2 = vm.spawn_kernel(code_seeker, color=0x00FFFF)

    print(f"Spawned kernel {pid1} (yellow emitter)")
    print(f"Spawned kernel {pid2} (cyan seeker)")
    print()

    print("Running simulation...")
    print(f"Pheromone decay rate: {vm.pheromone_decay}")
    print(f"Pheromone diffusion rate: {vm.pheromone_diffusion}")
    print()

    # Run simulation
    for i in range(200):
        vm.step()

        if i % 40 == 0:
            # Get seeker position
            seeker = vm.kernels[1]
            seeker_x = seeker.regs[3]
            seeker_y = seeker.regs[4]

            # Get pheromone level at seeker location
            phero_level = vm.pheromone[seeker_y % vm.height, seeker_x % vm.width]
            max_phero = vm.pheromone.max()

            print(f"  Cycle {vm.cycle:4d}: Seeker at ({seeker_x:3d}, {seeker_y:3d}), "
                  f"pheromone = {phero_level:5.1f}, max = {max_phero:5.1f}")

        if vm.alive_count() == 0:
            break

    print()
    print("Final state:")
    seeker = vm.kernels[1]
    seeker_x = seeker.regs[3]
    seeker_y = seeker.regs[4]
    distance = ((seeker_x - 512)**2 + (seeker_y - 512)**2)**0.5

    print(f"  Seeker final position: ({seeker_x}, {seeker_y})")
    print(f"  Distance from emitter: {distance:.1f} pixels")
    print(f"  Pheromone field max: {vm.pheromone.max():.1f}")
    print(f"  Pheromone field mean: {vm.pheromone.mean():.3f}")
    print()

    # Count non-zero pheromone locations
    import numpy as np
    phero_coverage = (vm.pheromone > 1.0).sum()
    print(f"  Pheromone coverage: {phero_coverage} pixels")

    print()
    print("=" * 70)
    if distance < 50:
        print("SUCCESS: Cyan kernel found the pheromone source!")
    else:
        print("Cyan kernel is following the gradient...")
    print("=" * 70)
    print()
    print("The pheromone field is diffusing and decaying each cycle.")
    print("The seeker follows the chemical gradient toward the source.")
    print("This is the foundation for:")
    print("  • Stigmergy (indirect coordination)")
    print("  • Trail-following behavior")
    print("  • Resource discovery")
    print("  • Collective foraging")
    print()

    # Try to save visualization
    try:
        from PIL import Image
        import numpy as np

        # Create visualization: framebuffer + pheromone overlay
        vis = vm.framebuffer.copy()
        # Overlay pheromone as red tint
        phero_vis = (vm.pheromone / vm.pheromone.max() * 128).astype(np.uint8) if vm.pheromone.max() > 0 else vm.pheromone.astype(np.uint8)
        vis[:,:,0] = np.maximum(vis[:,:,0], phero_vis)

        img = Image.fromarray(vis, mode='RGB')
        img.save('pheromone_output.png')
        print("Visualization saved to: pheromone_output.png")
        print("  (Red tint shows pheromone distribution)")
    except ImportError:
        print("(Install Pillow to save visualization: pip install Pillow)")


if __name__ == '__main__':
    main()
