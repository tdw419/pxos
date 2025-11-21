#!/usr/bin/env python3
"""
pxos/loops/vram_os_improvement_loop.py

The improvement loop that makes both the roadmap and VRAM program evolve.

Tight feedback loop with 4 phases:
1. BUILD: Execute roadmap → produce VRAM snapshot
2. EVAL: Measure VRAM quality → produce metrics
3. REFLECT: Analyze roadmap + metrics → identify improvements
4. PROPOSE: Generate new roadmap → repeat

Each iteration gets a generation ID:
- ROADMAP_VRAM_OS.gen001.yaml
- artifacts/vram_os_stage1.gen001.png
- eval/vram_os_stage1.gen001.json

The loop uses all three as input to produce the next generation.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any

from pxos.vram_sim import SimulatedVRAM
from pxos.agent.roadmap_loader import load_roadmap
from pxos.agent.roadmap_agent import RoadmapAgent
from pxos.eval.vram_os_evaluator import evaluate_and_save


class VRAMOSImprovementLoop:
    """
    Driver for the VRAM OS improvement loop.

    Iteratively builds, evaluates, and improves the VRAM OS.
    """

    def __init__(self,
                 base_roadmap_path: str,
                 output_dir: str = "artifacts",
                 eval_dir: str = "eval",
                 roadmap_dir: str = "pxos/roadmaps"):
        self.base_roadmap_path = base_roadmap_path
        self.output_dir = Path(output_dir)
        self.eval_dir = Path(eval_dir)
        self.roadmap_dir = Path(roadmap_dir)

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.roadmap_dir.mkdir(parents=True, exist_ok=True)

    def get_paths_for_gen(self, gen: int) -> Dict[str, str]:
        """Get file paths for a specific generation."""
        return {
            "roadmap": str(self.roadmap_dir / f"ROADMAP_VRAM_OS.gen{gen:03d}.yaml"),
            "png": str(self.output_dir / f"vram_os_stage1.gen{gen:03d}.png"),
            "eval_json": str(self.eval_dir / f"vram_os_stage1.gen{gen:03d}.json"),
        }

    def run_build(self, gen: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Phase 1: BUILD
        Load roadmap and execute it to produce VRAM snapshot.
        """
        paths = self.get_paths_for_gen(gen)

        if verbose:
            print(f"\n{'='*60}")
            print(f"GENERATION {gen:03d} - BUILD PHASE")
            print(f"{'='*60}")
            print(f"Roadmap: {paths['roadmap']}")
            print(f"Output:  {paths['png']}")

        # Load roadmap
        roadmap = load_roadmap(paths["roadmap"])

        # Create VRAM
        vram = SimulatedVRAM(
            roadmap.metadata.vram_width,
            roadmap.metadata.vram_height
        )

        # Execute roadmap
        agent = RoadmapAgent(vram, roadmap)
        ctx = agent.run(verbose=verbose)

        # Save snapshot
        agent.save_snapshot(paths["png"], verbose=verbose)

        return {
            "generation": gen,
            "roadmap_path": paths["roadmap"],
            "png_path": paths["png"],
            "context": ctx,
        }

    def run_eval(self, gen: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Phase 2: EVAL
        Evaluate the VRAM snapshot and produce metrics.
        """
        paths = self.get_paths_for_gen(gen)

        if verbose:
            print(f"\n{'='*60}")
            print(f"GENERATION {gen:03d} - EVAL PHASE")
            print(f"{'='*60}")

        # Run evaluation
        metrics = evaluate_and_save(
            paths["png"],
            paths["eval_json"],
            verbose=verbose
        )

        return metrics

    def run_reflect_and_propose(self, gen: int, verbose: bool = True) -> str:
        """
        Phase 3 & 4: REFLECT + PROPOSE

        This is where you'd hook in an LLM to:
        - Read the roadmap
        - Read the metrics
        - Propose changes to the roadmap

        For now, this is a stub that just copies the roadmap forward.
        Replace this with a call to LM Studio / Claude / Gemini.
        """
        current_paths = self.get_paths_for_gen(gen)
        next_paths = self.get_paths_for_gen(gen + 1)

        if verbose:
            print(f"\n{'='*60}")
            print(f"GENERATION {gen:03d} - REFLECT & PROPOSE PHASE")
            print(f"{'='*60}")
            print(f"[planner] Reading roadmap: {current_paths['roadmap']}")
            print(f"[planner] Reading metrics: {current_paths['eval_json']}")

        # TODO: Replace this with actual LLM call
        # For now, just copy roadmap forward
        shutil.copyfile(current_paths["roadmap"], next_paths["roadmap"])

        if verbose:
            print(f"[planner] (stub) Copied roadmap to {next_paths['roadmap']}")
            print(f"[planner] TODO: Replace with LLM-based roadmap improvement")

        return next_paths["roadmap"]

    def run_iteration(self, gen: int, verbose: bool = True) -> Dict[str, Any]:
        """Run one complete iteration of the loop."""
        # Phase 1: Build
        build_result = self.run_build(gen, verbose=verbose)

        # Phase 2: Eval
        metrics = self.run_eval(gen, verbose=verbose)

        # Phase 3 & 4: Reflect + Propose
        next_roadmap = self.run_reflect_and_propose(gen, verbose=verbose)

        return {
            "generation": gen,
            "build_result": build_result,
            "metrics": metrics,
            "next_roadmap": next_roadmap,
        }

    def run_loop(self, start_gen: int, num_iterations: int, verbose: bool = True):
        """Run the improvement loop for N iterations."""
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# VRAM OS IMPROVEMENT LOOP")
            print(f"# Starting generation: {start_gen}")
            print(f"# Iterations: {num_iterations}")
            print(f"{'#'*60}\n")

        # Ensure gen 1 roadmap exists (copy from base if needed)
        gen1_path = self.get_paths_for_gen(start_gen)["roadmap"]
        if not Path(gen1_path).exists():
            if verbose:
                print(f"[loop] Initializing gen {start_gen} roadmap from {self.base_roadmap_path}")
            shutil.copyfile(self.base_roadmap_path, gen1_path)

        # Run iterations
        results = []
        for i in range(num_iterations):
            gen = start_gen + i
            result = self.run_iteration(gen, verbose=verbose)
            results.append(result)

            if verbose:
                score = result["metrics"]["score"]
                max_score = result["metrics"]["max_score"]
                print(f"\n[loop] Generation {gen} complete: score {score}/{max_score}")

        if verbose:
            print(f"\n{'#'*60}")
            print(f"# LOOP COMPLETE")
            print(f"# Generations: {start_gen} → {start_gen + num_iterations - 1}")
            print(f"{'#'*60}\n")

        return results


def main():
    """Run the improvement loop from command line."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pxos/loops/vram_os_improvement_loop.py <base_roadmap.yaml> [num_iterations]")
        sys.exit(1)

    base_roadmap = sys.argv[1]
    num_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    loop = VRAMOSImprovementLoop(base_roadmap)
    loop.run_loop(start_gen=1, num_iterations=num_iters)


if __name__ == "__main__":
    main()
