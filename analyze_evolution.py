#!/usr/bin/env python3
"""
analyze_evolution.py - Analyze evolution experiment data

Generates plots showing:
- Population dynamics over time
- Birth/death rates
- Energy trends
- Food availability
- Reproductive success
"""

import csv
import sys

def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        sys.exit(1)

    # Load data
    data_file = "evolution_data.csv"
    try:
        with open(data_file, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except FileNotFoundError:
        print(f"Error: {data_file} not found")
        print("Run experiment_evolution_longrun.py first")
        sys.exit(1)

    # Parse data
    cycles = [int(row['cycle']) for row in data]
    alive = [int(row['alive']) for row in data]
    births = [int(row['total_births']) for row in data]
    deaths = [int(row['total_deaths']) for row in data]
    food = [float(row['total_food']) for row in data]
    avg_energy = [float(row['avg_energy']) for row in data]
    avg_age = [float(row['avg_age']) for row in data]
    avg_children = [float(row['avg_children']) for row in data]

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Digital Evolution Experiment Results', fontsize=16)

    # 1. Population over time
    ax = axes[0, 0]
    ax.plot(cycles, alive, 'g-', linewidth=2)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Population')
    ax.set_title('Population Dynamics')
    ax.grid(True, alpha=0.3)

    # 2. Births and deaths
    ax = axes[0, 1]
    ax.plot(cycles, births, 'b-', label='Total Births', linewidth=2)
    ax.plot(cycles, deaths, 'r-', label='Total Deaths', linewidth=2)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Count')
    ax.set_title('Cumulative Births & Deaths')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Food availability
    ax = axes[1, 0]
    ax.plot(cycles, food, 'orange', linewidth=2)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Total Food')
    ax.set_title('Food Availability')
    ax.grid(True, alpha=0.3)

    # 4. Average energy
    ax = axes[1, 1]
    ax.plot(cycles, avg_energy, 'purple', linewidth=2)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Average Energy')
    ax.set_title('Population Energy Level')
    ax.grid(True, alpha=0.3)

    # 5. Average age
    ax = axes[2, 0]
    ax.plot(cycles, avg_age, 'brown', linewidth=2)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Average Age (cycles)')
    ax.set_title('Population Age')
    ax.grid(True, alpha=0.3)

    # 6. Reproductive success
    ax = axes[2, 1]
    ax.plot(cycles, avg_children, 'cyan', linewidth=2)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Avg Children per Organism')
    ax.set_title('Reproductive Success')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = "evolution_analysis.png"
    plt.savefig(output_file, dpi=150)
    print(f"Analysis saved to: {output_file}")

    # Show plot
    plt.show()

    # Print summary statistics
    print()
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print()
    print(f"Cycles simulated: {cycles[-1]}")
    print(f"Peak population: {max(alive)}")
    print(f"Final population: {alive[-1]}")
    print(f"Total births: {births[-1]}")
    print(f"Total deaths: {deaths[-1]}")
    print()

    if alive[-1] > 0:
        print("Final state:")
        print(f"  Average energy: {avg_energy[-1]:.1f}")
        print(f"  Average age: {avg_age[-1]:.1f} cycles")
        print(f"  Average children: {avg_children[-1]:.2f}")
        print()
        print("✅ Population survived")
    else:
        print("⚰️  Population went extinct")
        # Find extinction point
        for i, pop in enumerate(alive):
            if pop == 0:
                print(f"  Extinction at cycle {cycles[i]}")
                break
    print()

if __name__ == '__main__':
    main()
