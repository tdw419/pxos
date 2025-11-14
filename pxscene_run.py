#!/usr/bin/env python3
"""
pxscene_run.py - compile a PXSCENE JSON and run it through the PXTERM terminal.
Usage:
  python pxscene_run.py scene1.json
"""

import os
import sys
import subprocess

def main():
    if len(sys.argv) != 2:
        print("Usage: pxscene_run.py <scene.json>")
        sys.exit(1)

    scene_path = sys.argv[1]
    base, _ = os.path.splitext(scene_path)
    pxterm_path = base + ".pxterm"

    # 1) compile JSON -> PXTERM
    subprocess.check_call([sys.executable, "pxscene_compile.py", scene_path, pxterm_path])

    # 2) run PXTERM
    subprocess.check_call([sys.executable, "pxos_llm_terminal.py", pxterm_path])

if __name__ == "__main__":
    main()
