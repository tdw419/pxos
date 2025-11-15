#!/usr/bin/env python3
"""
experiment_evolution_longrun.py - Long-run evolutionary experiment

Run 10,000 cycles of evolution with data collection.
Track: population, births, deaths, energy, mutations

Goal: Observe adaptation, speciation, or extinction
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pxvm.vm import PxVM
from pxvm.assembler import assemble
from pxvm.glyphs import *
import csv
import time

# EVOLVED FORAGER v1
# Strategy: Random walk + eat + spawn when energy > 600
forager_v1 = f"""
start:
    # Current position in R0, R1

    # Try to eat at current location
    SYS_EAT

    # Draw self (color based on energy level)
    MOV R2, 0x00FF00    # Green = alive
    PLOT

    # Random walk (simplified: just increment position)
    MOV R3, 10
    ADD R0, R3
    ADD R1, R3

    # Bounds check (wrap around)
    MOV R4, 1024
    CMP R0, R4
    # (no JGE yet, so we'll just let it wrap via modulo in PLOT)

    # Try to eat at new location
    SYS_EAT

    # Spawn if we have energy (simplified: just spawn periodically)
    MOV R1, 550
    MOV R2, 500
    SYS_SPAWN

    # Leave a glyph marking our path
    MOV R0, 500
    MOV R1, 500
    MOV R2, {GLYPH_SELF}
    SYS_WRITE_GLYPH

    # Loop forever
    NOP
    NOP
    NOP
    HALT
"""

def main():
    print("=" * 70)
    print("LONG-RUN EVOLUTION EXPERIMENT")
    print("=" * 70)
    print()
    print("Parameters:")
    print("  Duration: 10,000 cycles")
    print("  Mutation rate: 0.002 (0.2% per byte)")
    print("  Energy per cycle: 0.2")
    print("  Initial food: 300 locations")
    print()
    print("Expected outcomes:")
    print("  - Adaptation (more efficient foragers)")
    print("  - Lineage divergence")
    print("  - Possible extinction events")
    print("  - Emergent behaviors")
    print()

    # Create VM with moderate selection pressure
    vm = PxVM(
        width=1024,
        height=1024,
        mutation_rate=0.002,      # 0.2% - higher than default
        energy_per_cycle=0.2,     # Moderate hunger
        spawn_energy_cost=0.4     # 40% of parent energy
    )

    # Heavy initial food seeding
    print("Seeding food across world...")
    vm.seed_food(count=300, amount=80.0)
    print(f"  Total food: {vm.food.sum():.1f}")
    print()

    # Assemble and spawn initial population (5 founders)
    code = assemble(forager_v1)
    print(f"Spawning {5} founder organisms...")
    print(f"  Genome size: {len(code)} bytes")
    print()

    for i in range(5):
        pid = vm.spawn_kernel(code, color=0x00FF00)
        # Spread them out
        vm.kernels[i].regs[0] = 200 + i * 150
        vm.kernels[i].regs[1] = 200 + i * 150

    # Data collection
    data_file = "evolution_data.csv"
    with open(data_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'cycle', 'alive', 'total_births', 'total_deaths',
            'total_food', 'avg_energy', 'avg_age', 'avg_children'
        ])

    print("=" * 70)
    print("Starting evolution simulation...")
    print("=" * 70)
    print()

    start_time = time.time()
    last_report = 0

    for cycle in range(10000):
        vm.step()

        # Collect data every 100 cycles
        if cycle % 100 == 0:
            alive = vm.alive_count()

            if alive > 0:
                live_kernels = [k for k in vm.kernels if not k.halted]
                avg_energy = sum(k.energy for k in live_kernels) / alive
                avg_age = sum(k.cycles for k in live_kernels) / alive
                avg_children = sum(k.children_count for k in live_kernels) / alive
            else:
                avg_energy = 0
                avg_age = 0
                avg_children = 0

            # Write data
            with open(data_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    vm.cycle,
                    alive,
                    vm.total_births,
                    vm.total_deaths,
                    vm.food.sum(),
                    avg_energy,
                    avg_age,
                    avg_children
                ])

            # Report every 1000 cycles
            if cycle - last_report >= 1000:
                elapsed = time.time() - start_time
                rate = vm.cycle / elapsed if elapsed > 0 else 0
                print(f"Cycle {vm.cycle:5d} | "
                      f"Pop: {alive:3d} | "
                      f"Births: {vm.total_births:4d} | "
                      f"Deaths: {vm.total_deaths:4d} | "
                      f"Food: {vm.food.sum():8.1f} | "
                      f"Avg Energy: {avg_energy:6.1f} | "
                      f"Rate: {rate:.1f} cyc/s")
                last_report = cycle

        # Stop if extinction
        if vm.alive_count() == 0:
            print()
            print(f"EXTINCTION at cycle {vm.cycle}")
            print()
            break

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Total cycles: {vm.cycle}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Average rate: {vm.cycle / elapsed:.1f} cycles/sec")
    print()
    print(f"Final population: {vm.alive_count()}")
    print(f"Total births: {vm.total_births}")
    print(f"Total deaths: {vm.total_deaths}")
    print(f"Net growth: {vm.total_births - vm.total_deaths}")
    print()

    if vm.alive_count() > 0:
        live = [k for k in vm.kernels if not k.halted]
        print("Survivors:")
        for k in live[:10]:  # Show first 10
            efficiency = k.total_energy_gained / k.total_energy_consumed if k.total_energy_consumed > 0 else 0
            print(f"  PID {k.pid}: "
                  f"Age {k.cycles}, "
                  f"Energy {k.energy:.1f}, "
                  f"Children {k.children_count}, "
                  f"Efficiency {efficiency:.2f}")
        print()

    print(f"Data saved to: {data_file}")
    print()
    print("Analyze with:")
    print(f"  python analyze_evolution.py")
    print()

if __name__ == '__main__':
    main()
