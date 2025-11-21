#!/usr/bin/env python3
"""
run_vram_os_build.py

Build pxOS VRAM image from a roadmap.

Usage:
    python run_vram_os_build.py [roadmap.yaml] [output.png]

Example:
    python run_vram_os_build.py pxos/roadmaps/ROADMAP_VRAM_OS.yaml artifacts/vram_os.png
"""

import sys
from pathlib import Path

from pxos.vram_sim import SimulatedVRAM
from pxos.agent.roadmap_loader import load_roadmap
from pxos.agent.roadmap_agent import RoadmapAgent


def main():
    # Parse arguments
    roadmap_path = sys.argv[1] if len(sys.argv) > 1 else "pxos/roadmaps/ROADMAP_VRAM_OS.yaml"
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"[build] Loading roadmap from {roadmap_path}")

    # Load roadmap
    roadmap = load_roadmap(roadmap_path)

    # Create VRAM
    vram = SimulatedVRAM(
        roadmap.metadata.vram_width,
        roadmap.metadata.vram_height
    )

    # Execute roadmap
    agent = RoadmapAgent(vram, roadmap)
    ctx = agent.run(verbose=True)

    # Save snapshot
    if output_path is None:
        output_path = roadmap.get_output_path()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save_snapshot(output_path, verbose=True)

    print("\n[build] ✓ Build complete")
    print(f"[build] Output: {output_path}")


if __name__ == "__main__":
    main()
