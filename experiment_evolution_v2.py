#!/usr/bin/env python3
"""
experiment_evolution_v2.py - Evolution with PROPER forager (no HALT)

Previous issue: Organisms had HALT, so they only ran once and starved.
Solution: True infinite loop with foraging behavior.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pxvm.vm import PxVM
from pxvm.assembler import assemble
from pxvm.glyphs import *
import csv
import time

# ETERNAL FORAGER - Never halts, constantly forages
eternal_forager = f"""
loop:
    # Draw self at current position
    MOV R2, 0x00FF00    # Green
    PLOT

    # Try to eat at current position
    SYS_EAT

    # Move to next location (simple increment)
    MOV R3, 5
    ADD R0, R3
    ADD R1, R3

    # Eat again at new location
    SYS_EAT

    # Spawn child occasionally (every ~10 loop iterations)
    # For now, just always try - energy system will limit it
    MOV R1, 520
    MOV R2, 510
    SYS_SPAWN

    # Mark territory with glyph
    MOV R2, {GLYPH_SELF}
    SYS_WRITE_GLYPH

    # Jump back to start (infinite loop)
    JMP loop
"""

def main():
    print("=" * 70)
    print("EVOLUTION V2 - ETERNAL FORAGERS")
    print("=" * 70)
    print()
    print("Fix: Organisms now loop forever (no HALT)")
    print("They will forage, eat, move, and spawn until they starve")
    print()

    # More permissive parameters to allow survival
    vm = PxVM(
        width=1024,
        height=1024,
        mutation_rate=0.001,      # Lower mutation (less disruption)
        energy_per_cycle=0.05,    # VERY low hunger (easy survival)
        spawn_energy_cost=0.3     # 30% spawn cost
    )

    # MASSIVE food seeding
    print("Seeding abundant food...")
    vm.seed_food(count=500, amount=100.0)
    print(f"  Total food: {vm.food.sum():.1f}")
    print(f"  Food regeneration: {vm.food_regen_rate} per cycle per pixel")
    print()

    # Compile forager
    code = assemble(eternal_forager)
    print(f"Forager genome: {len(code)} bytes")
    print()

    # Spawn 3 founders (lower initial pop to avoid crowding)
    print("Spawning 3 founder organisms...")
    for i in range(3):
        pid = vm.spawn_kernel(code, color=0x00FF00)
        vm.kernels[i].regs[0] = 300 + i * 200
        vm.kernels[i].regs[1] = 300 + i * 200
        vm.kernels[i].energy = 2000.0  # Start with MORE energy

    # Data collection
    data_file = "evolution_data_v2.csv"
    with open(data_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'cycle', 'alive', 'total_births', 'total_deaths',
            'total_food', 'avg_energy', 'max_energy', 'avg_children'
        ])

    print("=" * 70)
    print("Starting evolution...")
    print("=" * 70)
    print()

    start_time = time.time()
    max_cycles = 5000  # Lower target for testing

    for cycle in range(max_cycles):
        vm.step()

        # Data collection every 50 cycles
        if cycle % 50 == 0:
            alive = vm.alive_count()

            if alive > 0:
                live = [k for k in vm.kernels if not k.halted]
                avg_energy = sum(k.energy for k in live) / alive
                max_energy = max(k.energy for k in live)
                avg_children = sum(k.children_count for k in live) / alive
            else:
                avg_energy = max_energy = avg_children = 0

            with open(data_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    vm.cycle, alive, vm.total_births, vm.total_deaths,
                    vm.food.sum(), avg_energy, max_energy, avg_children
                ])

            # Report every 500 cycles
            if cycle % 500 == 0:
                elapsed = time.time() - start_time
                rate = vm.cycle / elapsed if elapsed > 0 else 0
                print(f"Cycle {vm.cycle:5d} | "
                      f"Pop: {alive:3d} | "
                      f"Births: {vm.total_births:4d} | "
                      f"Deaths: {vm.total_deaths:4d} | "
                      f"Food: {vm.food.sum():10.1f} | "
                      f"Avg E: {avg_energy:7.1f} | "
                      f"Max E: {max_energy:7.1f}")

        # Check for extinction
        if vm.alive_count() == 0:
            print()
            print(f"⚰️  EXTINCTION at cycle {vm.cycle}")
            print()
            break

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Duration: {vm.cycle} cycles in {elapsed:.1f}s ({vm.cycle/elapsed:.1f} cyc/s)")
    print(f"Final population: {vm.alive_count()}")
    print(f"Total organisms created: {vm.next_pid - 1}")
    print(f"Births: {vm.total_births}")
    print(f"Deaths: {vm.total_deaths}")
    print()

    if vm.alive_count() > 0:
        print("✅ POPULATION SURVIVED")
        print()
        live = sorted([k for k in vm.kernels if not k.halted],
                     key=lambda k: k.children_count, reverse=True)

        print("Most successful organisms (by children):")
        for k in live[:5]:
            eff = k.total_energy_gained / k.total_energy_consumed if k.total_energy_consumed > 0 else 0
            print(f"  PID {k.pid}: {k.children_count} children, "
                  f"energy={k.energy:.1f}, "
                  f"efficiency={eff:.2f}")
    else:
        print("⚰️  POPULATION EXTINCT")

    print()
    print(f"Data: {data_file}")
    print(f"Analyze: python analyze_evolution.py")
    print()

if __name__ == '__main__':
    main()
