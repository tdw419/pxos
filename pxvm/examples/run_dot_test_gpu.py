#!/usr/bin/env python3
"""
pxvm/examples/run_dot_test_gpu.py

Run dot_test.pxi through the GPU interpreter and verify results.

This proves that the same .pxi file runs identically on CPU and GPU,
demonstrating that the Pixel Protocol is executor-agnostic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pxvm.gpu import run_program_gpu, GPUNotAvailableError, WGPU_AVAILABLE
from pxvm.examples.make_dot_test import make_dot_test_image, save_pxi


def load_pxi(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    return np.array(img, dtype=np.uint8)


def main() -> None:
    print("="*60)
    print("pxVM GPU EXECUTION TEST")
    print("="*60)
    print()

    # Check GPU availability
    if not WGPU_AVAILABLE:
        print("❌ GPU execution not available")
        print()
        print("wgpu-py is not installed.")
        print("Install with: pip install wgpu")
        print()
        print("Note: Requires compatible GPU and drivers.")
        print("See: https://github.com/pygfx/wgpu-py")
        print()
        return

    print("✅ wgpu-py is available")
    print()

    # Load or generate test program
    base = Path(__file__).parent
    pxi_path = base / "dot_test.pxi"

    if not pxi_path.exists():
        img = make_dot_test_image()
        save_pxi(pxi_path, img)
        print(f"Generated: {pxi_path}")
    else:
        print(f"Loading: {pxi_path}")

    img = load_pxi(pxi_path)
    print(f"Image size: {img.shape[1]}×{img.shape[0]} RGBA")
    print()

    # Execute on GPU
    print("Executing on GPU...")
    try:
        result_img = run_program_gpu(img, verbose=True)
    except GPUNotAvailableError as e:
        print(f"❌ {e}")
        return
    except Exception as e:
        print(f"❌ GPU execution failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # Read result
    low = int(result_img[3, 0, 0])
    high = int(result_img[3, 0, 1])
    dot_val = low + (high << 8)

    print("="*60)
    print("RESULT")
    print("="*60)
    print(f"Dot product read from pixel (0,3): {dot_val}")
    print(f"Encoded as bytes: R={low}, G={high}")
    print()

    # Verify expected result
    expected = 300  # 1*10 + 2*20 + 3*30 + 4*40
    if dot_val == expected:
        print(f"✅ CORRECT (expected {expected})")
    else:
        print(f"❌ INCORRECT (expected {expected}, got {dot_val})")

    print("="*60)


if __name__ == "__main__":
    main()
