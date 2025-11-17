#!/usr/bin/env python3
"""
pxvm/examples/compare_cpu_gpu.py

Execute dot_test.pxi on both CPU and GPU interpreters and verify
that they produce identical results.

This is the definitive proof that the Pixel Protocol is executor-agnostic:
same .pxi file, same results, different executors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pxvm.core.interpreter import run_program as run_program_cpu
from pxvm.gpu import run_program_gpu
from pxvm.examples.make_dot_test import make_dot_test_image, save_pxi

def load_pxi(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    return np.array(img, dtype=np.uint8)

def main() -> None:
    base = Path(__file__).parent
    pxi_path = base / "dot_test.pxi"

    if not pxi_path.exists():
        img = make_dot_test_image()
        save_pxi(pxi_path, img)
        print(f"Generated {pxi_path}")

    img_cpu = load_pxi(pxi_path)
    img_gpu = load_pxi(pxi_path)

    # Run on CPU
    print("Running on CPU...")
    img_cpu = run_program_cpu(img_cpu)
    low_cpu = int(img_cpu[3, 0, 0])
    high_cpu = int(img_cpu[3, 0, 1])
    dot_cpu = low_cpu + (high_cpu << 8)

    # Run on GPU
    print("Running on GPU...")
    try:
        img_gpu = run_program_gpu(img_gpu)
        low_gpu = int(img_gpu[3, 0, 0])
        high_gpu = int(img_gpu[3, 0, 1])
        dot_gpu = low_gpu + (high_gpu << 8)
    except Exception as e:
        print(f"GPU execution failed: {e}")
        dot_gpu = -1

    print("\n============================================================")
    print("COMPARISON")
    print("============================================================")
    print(f"CPU result:  {dot_cpu}")
    print(f"GPU result:  {dot_gpu}")

    if np.array_equal(img_cpu, img_gpu):
        print("\n✅ IDENTICAL RESULTS")
        print("\nThe Pixel Protocol is executor-agnostic!")
        print("Same .pxi file, same results, different executors.")
    else:
        print("\n❌ RESULTS DIFFER")

if __name__ == "__main__":
    main()
