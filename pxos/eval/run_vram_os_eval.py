#!/usr/bin/env python3
"""
pxos/eval/run_vram_os_eval.py

Command-line tool to evaluate a VRAM OS snapshot.

Usage:
    python pxos/eval/run_vram_os_eval.py <png_path> <output_json>
"""

import sys
from pxos.eval.vram_os_evaluator import evaluate_and_save


def main():
    if len(sys.argv) < 3:
        print("Usage: python pxos/eval/run_vram_os_eval.py <png_path> <output_json>")
        sys.exit(1)

    png_path = sys.argv[1]
    output_json = sys.argv[2]

    evaluate_and_save(png_path, output_json, verbose=True)


if __name__ == "__main__":
    main()
