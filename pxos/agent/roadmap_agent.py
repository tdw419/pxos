"""
pxos/agent/roadmap_agent.py

The RoadmapAgent executes a sequence of build steps against SimulatedVRAM.

This is the "builder" that takes a roadmap and actually writes the OS into VRAM.
"""

from typing import Dict, Any
from pxos.vram_sim import SimulatedVRAM
from pxos.agent.roadmap_types import Roadmap, RoadmapStep


class RoadmapAgent:
    """
    Agent that executes a roadmap to build pxOS in VRAM.

    The agent maintains:
    - A reference to SimulatedVRAM (the substrate)
    - A list of steps to execute
    - A context dict that carries state between steps
    """

    def __init__(self, vram: SimulatedVRAM, roadmap: Roadmap):
        self.vram = vram
        self.roadmap = roadmap
        self.ctx: Dict[str, Any] = {
            "vram_width": vram.width,
            "vram_height": vram.height,
            "roadmap_name": roadmap.metadata.name,
            "roadmap_version": roadmap.metadata.version,
            "generation": roadmap.metadata.generation,
        }
        self.step_results = []

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """Execute all steps in sequence."""
        if verbose:
            print(f"[roadmap-agent] Starting roadmap: {self.roadmap.metadata.name}")
            print(f"[roadmap-agent] Generation: {self.roadmap.metadata.generation}")
            print(f"[roadmap-agent] VRAM: {self.vram.width}x{self.vram.height}")
            print(f"[roadmap-agent] Steps: {len(self.roadmap.steps)}")
            print()

        for i, step in enumerate(self.roadmap.steps, 1):
            if verbose:
                print(f"[roadmap-agent] [{i}/{len(self.roadmap.steps)}] {step.name}")
                if step.description:
                    print(f"                {step.description}")

            # Execute step
            try:
                self.ctx = step.execute(self.vram, self.ctx)
                self.step_results.append({
                    "step": step.name,
                    "status": "success",
                })
            except Exception as e:
                if verbose:
                    print(f"[roadmap-agent] ERROR in step '{step.name}': {e}")
                self.step_results.append({
                    "step": step.name,
                    "status": "error",
                    "error": str(e),
                })
                raise

        if verbose:
            print()
            print(f"[roadmap-agent] ✓ All {len(self.roadmap.steps)} steps completed")

        return self.ctx

    def save_snapshot(self, path: str = None, verbose: bool = True):
        """Save VRAM to PNG."""
        if path is None:
            path = self.roadmap.get_output_path()

        self.vram.save_png(path)
        if verbose:
            print(f"[roadmap-agent] Saved VRAM snapshot → {path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        return {
            "roadmap": self.roadmap.metadata.name,
            "version": self.roadmap.metadata.version,
            "generation": self.roadmap.metadata.generation,
            "vram_size": f"{self.vram.width}x{self.vram.height}",
            "steps_executed": len(self.step_results),
            "steps_failed": sum(1 for r in self.step_results if r["status"] == "error"),
            "context": self.ctx,
        }
