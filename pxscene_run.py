#!/usr/bin/env python3
"""
PXSCENE Runner - Compile and execute PXSCENE files in one step
"""
from __future__ import annotations
import sys
import json
from pxscene_compile import compile_pxscene
from pxos_llm_terminal import run_pxterm_file
import tempfile
import os


def run_pxscene(scene_file: str):
    """Compile and run a PXSCENE file"""
    # Load scene to get output filename
    with open(scene_file, 'r') as f:
        scene = json.load(f)

    output_file = scene.get("output", {}).get("file", "output.png")

    # Compile to PXTERM
    instructions = compile_pxscene(scene)

    # Write to temporary PXTERM file
    pxterm_file = scene_file.replace(".json", ".pxterm")
    with open(pxterm_file, 'w') as f:
        f.write("\n".join(instructions) + "\n")

    print(f"Compiled {scene_file} -> {pxterm_file}")
    print(f"Generated {len(instructions)} instructions")
    print()

    # Execute PXTERM
    print(f"Executing {pxterm_file}...")
    run_pxterm_file(pxterm_file, output_file)

    print()
    print(f"✓ Scene rendered successfully: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python pxscene_run.py <scene.json>")
        sys.exit(1)

    scene_file = sys.argv[1]

    if not os.path.exists(scene_file):
        print(f"Error: File not found: {scene_file}")
        sys.exit(1)

    run_pxscene(scene_file)


if __name__ == "__main__":
    main()
