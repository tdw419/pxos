#!/usr/bin/env python3
"""
experiment_paradise.py - Evolution in resource-rich environment

Ultra-low selection pressure to allow sustained evolution.
Goal: Observe mutation accumulation and drift over 50K+ cycles.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pxvm.vm import PxVM
from pxvm.assembler import assemble
from pxvm.glyphs import *
import csv
import time

# Simple efficient forager
paradise_forager = f"""
loop:
    SYS_EAT
    MOV R2, 0x00FF00
    PLOT
    MOV R3, 2
    ADD R0, R3
    ADD R1, R3
    SYS_EAT
    MOV R1, 510
    MOV R2, 505
    SYS_SPAWN
    JMP loop
"""

def main():
    print("=" * 70)
    print("PARADISE MODE — Sustained Evolution Experiment")
    print("=" * 70)
    print()
    print("Environment: Resource-rich, low selection pressure")
    print("Goal: Observe drift, mutation accumulation, lineage divergence")
    print("Duration: 50,000 cycles")
    print()

    # PARADISE PARAMETERS - Easy survival
    vm = PxVM(
        width=1024,
        height=1024,
        mutation_rate=0.002,       # 0.2% - visible mutations
        energy_per_cycle=0.02,     # VERY low hunger (50x less than v1)
        spawn_energy_cost=0.2      # 20% spawn cost (cheap reproduction)
    )

    # MASSIVE food abundance
    print("Creating paradise world...")
    vm.seed_food(count=800, amount=100.0)
    # Increase food regeneration significantly
    vm.food_regen_rate = 0.05  # 5× higher regen
    print(f"  Initial food: {vm.food.sum():.1f}")
    print(f"  Food regeneration: {vm.food_regen_rate} per pixel per cycle")
    print()

    code = assemble(paradise_forager)
    print(f"Genome: {len(code)} bytes")

    # 5 founders for genetic diversity
    print("Spawning 5 founder organisms...")
    for i in range(5):
        pid = vm.spawn_kernel(code, color=0x00FF00)
        vm.kernels[i].regs[0] = 200 + i * 150
        vm.kernels[i].regs[1] = 200 + i * 150
        vm.kernels[i].energy = 3000.0  # Lots of starting energy

    data_file = "evolution_paradise.csv"
    with open(data_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'cycle', 'alive', 'total_births', 'total_deaths',
            'total_food', 'avg_energy', 'max_energy',
            'avg_age', 'max_children', 'genome_diversity'
        ])

    print()
    print("=" * 70)
    print("EVOLUTION IN PARADISE")
    print("=" * 70)
    print()

    start_time = time.time()
    last_report = 0
    max_cycles = 50000

    for cycle in range(max_cycles):
        vm.step()

        # Data every 250 cycles
        if cycle % 250 == 0:
            alive = vm.alive_count()

            if alive > 0:
                live = [k for k in vm.kernels if not k.halted]
                avg_energy = sum(k.energy for k in live) / alive
                max_energy = max(k.energy for k in live)
                avg_age = sum(k.cycles for k in live) / alive
                max_children = max(k.children_count for k in live)

                # Simple genome diversity: count unique first 10 bytes
                genomes = set()
                for k in live:
                    genome_sample = bytes(k.memory[:10])
                    genomes.add(genome_sample)
                genome_diversity = len(genomes)
            else:
                avg_energy = max_energy = avg_age = max_children = genome_diversity = 0

            with open(data_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    vm.cycle, alive, vm.total_births, vm.total_deaths,
                    vm.food.sum(), avg_energy, max_energy,
                    avg_age, max_children, genome_diversity
                ])

        # Report every 2500 cycles
        if cycle - last_report >= 2500:
            alive = vm.alive_count()
            elapsed = time.time() - start_time
            rate = vm.cycle / elapsed if elapsed > 0 else 0
            eta = (max_cycles - vm.cycle) / rate / 60 if rate > 0 else 0

            if alive > 0:
                live = [k for k in vm.kernels if not k.halted]
                avg_e = sum(k.energy for k in live) / alive
                max_kids = max(k.children_count for k in live)

                # Check genome diversity
                genomes = set(bytes(k.memory[:10]) for k in live)
                diversity = len(genomes)
            else:
                avg_e = max_kids = diversity = 0

            print(f"[{vm.cycle:6d}] Pop:{alive:3d} | "
                  f"B:{vm.total_births:5d} D:{vm.total_deaths:5d} | "
                  f"AvgE:{avg_e:7.0f} | "
                  f"MaxKids:{max_kids:3d} | "
                  f"Diversity:{diversity:2d} | "
                  f"{rate:4.0f}c/s | "
                  f"ETA:{eta:.1f}m")
            last_report = cycle

        if vm.alive_count() == 0:
            print(f"\n⚰️  EXTINCTION at cycle {vm.cycle}\n")
            break

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("PARADISE EXPERIMENT COMPLETE")
    print("=" * 70)
    print()
    print(f"Cycles: {vm.cycle:,} in {elapsed/60:.1f} min ({vm.cycle/elapsed:.0f} c/s)")
    print(f"Population: {vm.alive_count()} alive")
    print(f"Organisms created: {vm.next_pid - 1}")
    print(f"Births: {vm.total_births} | Deaths: {vm.total_deaths}")
    print()

    if vm.alive_count() > 0:
        print("✅ SUSTAINED EVOLUTION ACHIEVED")
        print()

        live = sorted([k for k in vm.kernels if not k.halted],
                     key=lambda k: k.children_count, reverse=True)

        print("Most successful lineages:")
        for k in live[:10]:
            eff = k.total_energy_gained / k.total_energy_consumed if k.total_energy_consumed > 0 else 0
            generation = "Founder" if k.pid <= 5 else f"Gen-{(k.pid-5)//12+1}"
            print(f"  PID {k.pid:4d} ({generation:8s}): "
                  f"{k.children_count:3d} children | "
                  f"Age {k.cycles:7d} | "
                  f"E={k.energy:7.0f} | "
                  f"Eff={eff:.3f}")

        print()
        print("Genetic diversity:")
        genomes = {}
        for k in live:
            signature = bytes(k.memory[:20])
            if signature in genomes:
                genomes[signature] += 1
            else:
                genomes[signature] = 1

        print(f"  Unique genomes (first 20 bytes): {len(genomes)}")
        print(f"  Population: {vm.alive_count()}")
        print(f"  Diversity ratio: {len(genomes)/vm.alive_count():.2%}")

    print()
    print(f"Data: {data_file}")
    print()

if __name__ == '__main__':
    main()
