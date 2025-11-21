#!/usr/bin/env python3
"""
demo_vram_substrate.py

Complete demonstration of the simulated VRAM substrate approach.

This script shows:
1. Building VRAM OS from a roadmap
2. Evaluating the result
3. Launching the improvement loop

Usage:
    python demo_vram_substrate.py
"""

import sys
from pathlib import Path

from pxos.vram_sim import SimulatedVRAM
from pxos.agent.roadmap_loader import load_roadmap
from pxos.agent.roadmap_agent import RoadmapAgent
from pxos.eval.vram_os_evaluator import evaluate_and_save
from pxos.loops.vram_os_improvement_loop import VRAMOSImprovementLoop


def demo_basic_build():
    """Demo 1: Basic VRAM OS build."""
    print("\n" + "="*60)
    print("DEMO 1: Basic VRAM OS Build")
    print("="*60 + "\n")

    # Load roadmap
    roadmap = load_roadmap("pxos/roadmaps/ROADMAP_VRAM_OS.yaml")

    # Create VRAM
    vram = SimulatedVRAM(
        roadmap.metadata.vram_width,
        roadmap.metadata.vram_height
    )

    # Execute roadmap
    agent = RoadmapAgent(vram, roadmap)
    agent.run(verbose=True)

    # Save
    output_path = "artifacts/vram_os_demo.png"
    agent.save_snapshot(output_path)

    print(f"\n✓ Demo 1 complete. Output: {output_path}\n")
    return output_path


def demo_evaluation(png_path):
    """Demo 2: Evaluate VRAM OS."""
    print("\n" + "="*60)
    print("DEMO 2: VRAM OS Evaluation")
    print("="*60 + "\n")

    eval_path = "eval/vram_os_demo.json"
    metrics = evaluate_and_save(png_path, eval_path, verbose=True)

    print(f"\n✓ Demo 2 complete. Metrics: {eval_path}\n")
    return metrics


def demo_improvement_loop():
    """Demo 3: Run improvement loop."""
    print("\n" + "="*60)
    print("DEMO 3: Improvement Loop (3 generations)")
    print("="*60 + "\n")

    loop = VRAMOSImprovementLoop(
        base_roadmap_path="pxos/roadmaps/ROADMAP_VRAM_OS.yaml"
    )

    results = loop.run_loop(start_gen=1, num_iterations=3, verbose=True)

    print("\n✓ Demo 3 complete.\n")
    print("Generated files:")
    for i in range(1, 4):
        paths = loop.get_paths_for_gen(i)
        print(f"  Gen {i}:")
        print(f"    Roadmap: {paths['roadmap']}")
        print(f"    PNG:     {paths['png']}")
        print(f"    Eval:    {paths['eval_json']}")

    return results


def main():
    print("\n" + "#"*60)
    print("# pxOS Simulated VRAM Substrate Demo")
    print("#"*60)

    # Ensure directories exist
    Path("artifacts").mkdir(exist_ok=True)
    Path("eval").mkdir(exist_ok=True)

    # Run demos
    png_path = demo_basic_build()
    metrics = demo_evaluation(png_path)
    results = demo_improvement_loop()

    print("\n" + "#"*60)
    print("# All Demos Complete!")
    print("#"*60)
    print("\nKey files created:")
    print("  - artifacts/vram_os_demo.png       (single build)")
    print("  - artifacts/vram_os_stage1.gen*.png (improvement loop)")
    print("  - eval/*.json                       (metrics)")
    print("\nNext steps:")
    print("  1. View the PNG files to see the VRAM OS layout")
    print("  2. Run with viewport: python run_vram_os_build_with_view.py")
    print("  3. Implement LLM-based roadmap improvement in the loop")
    print("  4. Add more sophisticated evaluation metrics")
    print("  5. Create a pixel interpreter to actually execute the programs")
    print()


if __name__ == "__main__":
    main()
