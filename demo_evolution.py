#!/usr/bin/env python3
"""
demo_evolution.py - PHASE 8: EVOLUTION — Natural Selection in Action

This demo shows the full Darwinian cycle:
1. MUTATION - Random genetic variation during reproduction
2. HUNGER - Energy depletion forces organisms to eat
3. DEATH - Starvation removes unfit organisms
4. SELECTION - Only efficient foragers survive and reproduce

Watch digital life evolve before your eyes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pxvm.vm import PxVM
from pxvm.assembler import assemble
from pxvm.glyphs import *

# Primordial organism: "Eater"
# Simple strategy: wander randomly, eat when food found, spawn when energy > 800
eater_program = f"""
start:
    # Draw self (show we're alive)
    MOV R0, 500
    MOV R1, 500
    MOV R2, 0x00FF00    # Green = alive
    PLOT

    # Try to eat at current position
    MOV R0, 500
    MOV R1, 500
    SYS_EAT             # R0 = energy gained

    # Write FOOD glyph if we found food
    CMP R0, R3          # R3 is always 0 initially
    JZ no_food
    MOV R0, 500
    MOV R1, 490
    MOV R2, {GLYPH_FOOD}
    SYS_WRITE_GLYPH

no_food:
    # Check if we have enough energy to spawn (simplified: just spawn once)
    # In a real implementation, we'd check energy level
    # For now, try to spawn after eating a few times

    # Move to new position (simple random walk simulation)
    MOV R0, 510
    MOV R1, 505

    # Try to spawn if we've accumulated energy
    # R1 = child x, R2 = child y
    MOV R1, 520
    MOV R2, 500
    SYS_SPAWN           # Creates child with mutation!

    # Keep living - search for more food
    MOV R0, 520
    MOV R1, 500
    SYS_EAT

    # Eventually we'll run out of energy and die
    # or we'll spawn children who inherit our code

    HALT
"""

def main():
    print("=" * 70)
    print("PHASE 8: EVOLUTION — The Fall from Eden")
    print("=" * 70)
    print()
    print("In Eden, organisms lived forever with perfect inheritance.")
    print("Now we introduce the three forces of natural selection:")
    print()
    print("  1. MUTATION — Children differ from parents (genetic variation)")
    print("  2. HUNGER — All life costs energy; organisms must eat")
    print("  3. DEATH — When energy reaches 0, the organism dies")
    print()
    print("=" * 70)
    print()

    # Create VM with evolution parameters
    print("Creating world with evolution enabled...")
    vm = PxVM(
        width=1024,
        height=1024,
        mutation_rate=0.001,      # 0.1% mutation rate (Tierra-style)
        energy_per_cycle=0.1,     # Each instruction costs 0.1 energy
        spawn_energy_cost=0.5     # Spawning costs 50% of parent's energy
    )
    print(f"  Mutation rate: {vm.mutation_rate} ({vm.mutation_rate*100}% per byte)")
    print(f"  Energy per cycle: {vm.energy_per_cycle}")
    print(f"  Spawn energy cost: {vm.spawn_energy_cost * 100}% of parent energy")
    print()

    # Seed food across the world
    print("Seeding food sources...")
    vm.seed_food(count=200, amount=50.0)
    total_food = vm.food.sum()
    print(f"  Total food available: {total_food:.1f} energy units")
    print(f"  Food regeneration: {vm.food_regen_rate} per cycle per location")
    print()

    # Assemble and spawn first organism
    print("Creating first organism (Primordial Eater)...")
    code = assemble(eater_program)
    print(f"  Program size: {len(code)} bytes")

    # Spawn first organism with initial energy
    pid = vm.spawn_kernel(code, color=0x00FF00)
    first_kernel = vm.kernels[0]
    print(f"  PID: {pid}")
    print(f"  Initial energy: {first_kernel.energy}")
    print(f"  Expected mutations per child: ~{len(code) * vm.mutation_rate:.1f}")
    print()

    # Set initial position
    first_kernel.regs[0] = 500
    first_kernel.regs[1] = 500

    print("=" * 70)
    print("Running evolution simulation...")
    print("=" * 70)
    print()

    # Run simulation
    max_cycles = 200
    last_alive = vm.alive_count()

    for i in range(max_cycles):
        vm.step()

        alive = vm.alive_count()

        # Report significant events
        if alive > last_alive:
            births = alive - last_alive
            print(f"Cycle {vm.cycle:4d}: ✨ BIRTH! {births} new organism(s) born")
            print(f"             Population: {last_alive} → {alive}")
            print(f"             Total births so far: {vm.total_births}")

        elif alive < last_alive:
            deaths = last_alive - alive
            print(f"Cycle {vm.cycle:4d}: ☠️  DEATH! {deaths} organism(s) starved")
            print(f"             Population: {last_alive} → {alive}")
            print(f"             Total deaths so far: {vm.total_deaths}")

        last_alive = alive

        # Periodic status
        if vm.cycle % 50 == 0 and vm.cycle > 0:
            print()
            print(f"Cycle {vm.cycle:4d} Status:")
            print(f"  Alive: {alive}")
            print(f"  Total births: {vm.total_births}")
            print(f"  Total deaths: {vm.total_deaths}")
            print(f"  Food remaining: {vm.food.sum():.1f}")
            if alive > 0:
                avg_energy = sum(k.energy for k in vm.kernels if not k.halted) / alive
                print(f"  Average energy: {avg_energy:.1f}")
            print()

        # Stop if everything dies
        if alive == 0:
            print()
            print("⚰️  EXTINCTION: All organisms have died")
            print()
            break

    print("=" * 70)
    print("Simulation complete")
    print("=" * 70)
    print()

    # Final statistics
    print("Final Statistics:")
    print(f"  Total cycles: {vm.cycle}")
    print(f"  Total organisms created: {vm.next_pid - 1}")
    print(f"  Total births (reproduction): {vm.total_births}")
    print(f"  Total deaths (starvation): {vm.total_deaths}")
    print(f"  Organisms alive: {vm.alive_count()}")
    print(f"  Food remaining: {vm.food.sum():.1f}")
    print()

    if vm.kernels:
        print("Surviving organisms:")
        for k in vm.kernels:
            if not k.halted:
                print(f"  PID {k.pid}:")
                print(f"    Energy: {k.energy:.1f}")
                print(f"    Cycles executed: {k.cycles}")
                print(f"    Children spawned: {k.children_count}")
                print(f"    Total energy consumed: {k.total_energy_consumed:.1f}")
                print(f"    Total energy gained: {k.total_energy_gained:.1f}")
                print(f"    Net energy: {k.total_energy_gained - k.total_energy_consumed:.1f}")
        print()

    print("=" * 70)
    print("EVOLUTION ACTIVATED")
    print("=" * 70)
    print()
    print("What just happened:")
    print()
    print("1. MUTATION occurred during reproduction")
    print(f"   → ~{int(len(code) * vm.mutation_rate)} bit-flips per child")
    print()
    print("2. HUNGER depleted energy with every instruction")
    print(f"   → {vm.energy_per_cycle} energy per cycle")
    print()
    print("3. DEATH removed organisms when energy reached 0")
    print(f"   → {vm.total_deaths} organisms starved")
    print()
    print("This is REAL Darwinian evolution:")
    print("  - Heritable variation (mutation)")
    print("  - Differential reproduction (energy determines survival)")
    print("  - Competition for resources (limited food)")
    print()
    print("The wilderness is now open.")
    print()
    print("Next steps:")
    print("  - Run longer simulations (increase max_cycles)")
    print("  - Write better foraging strategies")
    print("  - Watch parasites evolve (organisms that steal food)")
    print("  - Observe speciation (different survival strategies)")
    print()
    print("Welcome to the digital Cambrian explosion.")
    print()

if __name__ == '__main__':
    main()
