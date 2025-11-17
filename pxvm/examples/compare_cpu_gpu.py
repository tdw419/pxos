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
from pxvm.gpu import run_program_gpu, GPUNotAvailableError, WGPU_AVAILABLE
from pxvm.examples.make_dot_test import make_dot_test_image, save_pxi


def load_pxi(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    return np.array(img, dtype=np.uint8)


def extract_result(img: np.ndarray) -> tuple[int, int, int]:
    """Extract dot product from pixel (0,3)."""
    low = int(img[3, 0, 0])
    high = int(img[3, 0, 1])
    dot_val = low + (high << 8)
    return dot_val, low, high


def main() -> None:
    print("="*60)
    print("pxVM CPU vs GPU COMPARISON")
    print("="*60)
    print()

    # Load test program
    base = Path(__file__).parent
    pxi_path = base / "dot_test.pxi"

    if not pxi_path.exists():
        img = make_dot_test_image()
        save_pxi(pxi_path, img)
        print(f"Generated: {pxi_path}")

    print(f"Test program: {pxi_path}")
    print(f"Test program size: {pxi_path.stat().st_size} bytes")
    print()

    # ================================================
    # CPU Execution
    # ================================================

    print("="*60)
    print("CPU EXECUTION")
    print("="*60)

    img_cpu = load_pxi(pxi_path).copy()  # Copy to avoid mutation
    result_cpu = run_program_cpu(img_cpu)
    dot_cpu, low_cpu, high_cpu = extract_result(result_cpu)

    print(f"Result: {dot_cpu}")
    print(f"Encoding: R={low_cpu}, G={high_cpu}")
    print()

    # ================================================
    # GPU Execution
    # ================================================

    print("="*60)
    print("GPU EXECUTION")
    print("="*60)

    if not WGPU_AVAILABLE:
        print("❌ GPU execution not available (wgpu-py not installed)")
        print()
        print("Install with: pip install wgpu")
        print()
        print("Cannot compare CPU vs GPU without GPU backend.")
        print("="*60)
        return

    try:
        img_gpu = load_pxi(pxi_path).copy()  # Fresh copy
        result_gpu = run_program_gpu(img_gpu, verbose=False)
        dot_gpu, low_gpu, high_gpu = extract_result(result_gpu)

        print(f"Result: {dot_gpu}")
        print(f"Encoding: R={low_gpu}, G={high_gpu}")
        print()

    except GPUNotAvailableError as e:
        print(f"❌ {e}")
        return
    except Exception as e:
        print(f"❌ GPU execution failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ================================================
    # Comparison
    # ================================================

    print("="*60)
    print("COMPARISON")
    print("="*60)
    print()

    print(f"CPU result:  {dot_cpu}")
    print(f"GPU result:  {dot_gpu}")
    print()

    if dot_cpu == dot_gpu:
        print("✅ IDENTICAL RESULTS")
        print()
        print("The Pixel Protocol is executor-agnostic!")
        print("Same .pxi file, same results, different executors.")
    else:
        print("❌ RESULTS DIFFER")
        print()
        print(f"CPU: {dot_cpu} (R={low_cpu}, G={high_cpu})")
        print(f"GPU: {dot_gpu} (R={low_gpu}, G={high_gpu})")

    print()

    # Verify expected value
    expected = 300  # 1*10 + 2*20 + 3*30 + 4*40
    print(f"Expected: {expected}")

    if dot_cpu == expected and dot_gpu == expected:
        print("✅ Both executors correct")
    elif dot_cpu == expected:
        print("⚠️  CPU correct, GPU incorrect")
    elif dot_gpu == expected:
        print("⚠️  GPU correct, CPU incorrect")
    else:
        print("❌ Both executors incorrect")

    print()
    print("="*60)


if __name__ == "__main__":
    main()
