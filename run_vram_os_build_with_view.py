#!/usr/bin/env python3
"""
run_vram_os_build_with_view.py

Build pxOS VRAM image from a roadmap WITH live viewport visualization.

This launches a viewport window that shows the VRAM being built in real-time.

Usage:
    python run_vram_os_build_with_view.py [roadmap.yaml] [output.png]

Example:
    python run_vram_os_build_with_view.py pxos/roadmaps/ROADMAP_VRAM_OS.yaml

Note: The viewport window must be closed before the program exits.
"""

import sys
import time
from pathlib import Path

from pxos.vram_sim import SimulatedVRAM
from pxos.agent.roadmap_loader import load_roadmap
from pxos.agent.roadmap_agent import RoadmapAgent
from pxos.viewport.vram_viewport import launch_vram_viewer_thread


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

    # Launch viewport in background thread
    print("[build] Launching viewport...")
    viewer_thread = launch_vram_viewer_thread(
        vram,
        refresh_ms=100,
        title=f"pxOS - {roadmap.metadata.name}"
    )

    # Give viewport time to initialize
    time.sleep(0.5)

    # Execute roadmap
    print("\n[build] Executing roadmap steps...\n")
    agent = RoadmapAgent(vram, roadmap)
    ctx = agent.run(verbose=True)

    # Save snapshot
    if output_path is None:
        output_path = roadmap.get_output_path()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save_snapshot(output_path, verbose=True)

    print("\n[build] ✓ Build complete")
    print(f"[build] Output: {output_path}")
    print("\n[build] Viewport is still running. Close the window to exit.")

    # Keep main thread alive so viewport keeps running
    try:
        viewer_thread.join()
    except KeyboardInterrupt:
        print("\n[build] Interrupted")


if __name__ == "__main__":
    main()
