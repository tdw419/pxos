#!/usr/bin/env python3
"""
experiment_ultra_longrun.py - 50K cycle evolution experiment

Goal: Observe long-term evolutionary dynamics
- Adaptation to environment
- Energy efficiency improvements
- Potential speciation
- Population equilibrium

This will take ~20-30 minutes to run.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pxvm.vm import PxVM
from pxvm.assembler import assemble
from pxvm.glyphs import *
import csv
import time

# Optimized forager with better energy efficiency
optimized_forager = f"""
loop:
    # Eat at current position FIRST (before wasting energy on drawing)
    SYS_EAT

    # Draw self only if we have energy
    MOV R2, 0x00FF00
    PLOT

    # Move to next location (smaller steps = more thorough coverage)
    MOV R3, 3
    ADD R0, R3
    ADD R1, R3

    # Eat at new location
    SYS_EAT

    # Spawn only if we've accumulated significant energy
    # (energy check will be implicit - SPAWN fails if low energy)
    MOV R1, 515
    MOV R2, 505
    SYS_SPAWN

    # Continue foraging
    JMP loop
"""

def main():
    print("=" * 70)
    print("ULTRA-LONG EVOLUTION EXPERIMENT")
    print("=" * 70)
    print()
    print("Duration: 50,000 cycles (~20-30 minutes)")
    print("Goal: Observe long-term evolutionary dynamics")
    print()

    # Balanced parameters for sustained evolution
    vm = PxVM(
        width=1024,
        height=1024,
        mutation_rate=0.0015,     # 0.15% - moderate
        energy_per_cycle=0.15,    # Moderate hunger
        spawn_energy_cost=0.35    # 35% spawn cost
    )

    # Moderate food supply
    print("Seeding world with food...")
    vm.seed_food(count=400, amount=60.0)
    print(f"  Initial food: {vm.food.sum():.1f}")
    print()

    # Compile
    code = assemble(optimized_forager)
    print(f"Organism genome: {len(code)} bytes")

    # Start with 3 founders
    print("Spawning 3 founder organisms...")
    for i in range(3):
        pid = vm.spawn_kernel(code, color=0x00FF00)
        vm.kernels[i].regs[0] = 250 + i * 250
        vm.kernels[i].regs[1] = 250 + i * 250
        vm.kernels[i].energy = 1500.0  # Good starting energy

    # Data collection
    data_file = "evolution_ultra_longrun.csv"
    with open(data_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'cycle', 'alive', 'total_births', 'total_deaths',
            'total_food', 'avg_energy', 'max_energy', 'min_energy',
            'avg_children', 'max_children'
        ])

    print()
    print("=" * 70)
    print("RUNNING EVOLUTION...")
    print("=" * 70)
    print()

    start_time = time.time()
    last_status = 0

    max_cycles = 50000

    for cycle in range(max_cycles):
        vm.step()

        # Data collection every 200 cycles
        if cycle % 200 == 0:
            alive = vm.alive_count()

            if alive > 0:
                live = [k for k in vm.kernels if not k.halted]
                energies = [k.energy for k in live]
                avg_energy = sum(energies) / alive
                max_energy = max(energies)
                min_energy = min(energies)
                children_counts = [k.children_count for k in live]
                avg_children = sum(children_counts) / alive
                max_children = max(children_counts)
            else:
                avg_energy = max_energy = min_energy = 0
                avg_children = max_children = 0

            with open(data_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    vm.cycle, alive, vm.total_births, vm.total_deaths,
                    vm.food.sum(), avg_energy, max_energy, min_energy,
                    avg_children, max_children
                ])

        # Status every 2000 cycles
        if cycle - last_status >= 2000:
            alive = vm.alive_count()
            elapsed = time.time() - start_time
            rate = vm.cycle / elapsed if elapsed > 0 else 0
            eta = (max_cycles - vm.cycle) / rate if rate > 0 else 0

            if alive > 0:
                live = [k for k in vm.kernels if not k.halted]
                avg_e = sum(k.energy for k in live) / alive
                max_c = max(k.children_count for k in live)
            else:
                avg_e = 0
                max_c = 0

            print(f"[{vm.cycle:5d}] Pop:{alive:3d} | "
                  f"B:{vm.total_births:4d} D:{vm.total_deaths:4d} | "
                  f"Food:{vm.food.sum():10.0f} | "
                  f"AvgE:{avg_e:6.1f} | "
                  f"MaxKids:{max_c:2d} | "
                  f"Rate:{rate:4.0f}c/s | "
                  f"ETA:{eta/60:.1f}min")
            last_status = cycle

        # Check extinction
        if vm.alive_count() == 0:
            print()
            print(f"⚰️  EXTINCTION at cycle {vm.cycle}")
            break

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print()

    print(f"Total cycles: {vm.cycle:,}")
    print(f"Real time: {elapsed/60:.1f} minutes")
    print(f"Simulation rate: {vm.cycle/elapsed:.1f} cycles/sec")
    print()

    print(f"Final population: {vm.alive_count()}")
    print(f"Total organisms: {vm.next_pid - 1}")
    print(f"Births: {vm.total_births}")
    print(f"Deaths: {vm.total_deaths}")
    print(f"Survival rate: {(vm.total_births - vm.total_deaths) / vm.total_births * 100:.1f}%")
    print()

    if vm.alive_count() > 0:
        print("✅ POPULATION SURVIVED")
        print()

        live = sorted([k for k in vm.kernels if not k.halted],
                     key=lambda k: k.total_energy_gained, reverse=True)

        print("Top 5 organisms by total energy gained:")
        for k in live[:5]:
            eff = k.total_energy_gained / k.total_energy_consumed if k.total_energy_consumed > 0 else 0
            print(f"  PID {k.pid:3d}: "
                  f"Age {k.cycles:6d} | "
                  f"Energy {k.energy:7.1f} | "
                  f"Kids {k.children_count:2d} | "
                  f"Efficiency {eff:.3f}")

        print()
        print("Lineage analysis:")
        founders = [k for k in live if k.pid <= 3]
        descendants = [k for k in live if k.pid > 3]
        print(f"  Founders still alive: {len(founders)}/3")
        print(f"  Descendant organisms: {len(descendants)}")

    print()
    print(f"Data saved to: {data_file}")
    print(f"Analyze with: python analyze_evolution.py")
    print()

if __name__ == '__main__':
    main()
