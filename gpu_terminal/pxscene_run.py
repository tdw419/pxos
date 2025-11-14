#!/usr/bin/env python3
"""
pxscene_run.py - One-shot runner for PXSCENE

Compiles a PXSCENE JSON and runs it through the PXTERM terminal in a single command.

This is the easiest way for LLMs (or humans) to go from:
  PXSCENE JSON → Visual output

Usage:
  python pxscene_run.py scene.json

What it does:
  1. Compiles scene.json → scene.pxterm (using pxscene_compile.py)
  2. Executes scene.pxterm (using pxos_llm_terminal.py)
  3. Shows result in window
  4. Saves PNG if specified in JSON

Perfect for LLM automation:
  LLM generates JSON → Save as scene.json → python pxscene_run.py scene.json
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """Main entry point"""

    if len(sys.argv) != 2:
        print("pxscene_run.py - One-shot PXSCENE runner")
        print()
        print("Usage:")
        print("  python pxscene_run.py <scene.json>")
        print()
        print("Examples:")
        print("  python pxscene_run.py examples/scene1_basic.json")
        print("  python pxscene_run.py my_scene.json")
        print()
        print("This will:")
        print("  1. Compile JSON → PXTERM")
        print("  2. Execute PXTERM → GPU")
        print("  3. Show window with result")
        print("  4. Save PNG (if specified in JSON)")
        sys.exit(1)

    scene_path = Path(sys.argv[1])

    # Validate input file exists
    if not scene_path.exists():
        print(f"[ERROR] File not found: {scene_path}", file=sys.stderr)
        sys.exit(1)

    if not scene_path.suffix == ".json":
        print(f"[ERROR] Input must be a .json file, got: {scene_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output PXTERM path
    pxterm_path = scene_path.with_suffix(".pxterm")

    print("=" * 60)
    print("pxscene_run.py - One-shot PXSCENE runner")
    print("=" * 60)
    print(f"Input:  {scene_path}")
    print(f"Output: {pxterm_path}")
    print()

    # Step 1: Compile JSON → PXTERM
    print("[1/2] Compiling PXSCENE JSON → PXTERM...")
    try:
        subprocess.check_call(
            [sys.executable, "pxscene_compile.py", str(scene_path), str(pxterm_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("[1/2] ✓ Compilation successful")
    except subprocess.CalledProcessError as e:
        print(f"[1/2] ✗ Compilation failed", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Execute PXTERM → GPU
    print("[2/2] Executing PXTERM...")
    try:
        subprocess.check_call(
            [sys.executable, "pxos_llm_terminal.py", str(pxterm_path)]
        )
        print("[2/2] ✓ Execution complete")
    except subprocess.CalledProcessError as e:
        print(f"[2/2] ✗ Execution failed", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[2/2] Interrupted by user")
        sys.exit(0)

    print()
    print("=" * 60)
    print("✓ Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
