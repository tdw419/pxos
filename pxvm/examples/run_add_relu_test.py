#!/usr/bin/env python3
"""
pxvm/examples/run_add_relu_test.py

Execute add_relu_test.pxi and verify the results.

Expected:
- Row 3 should contain [15, 35, 55] after execution
- Demonstrates OP_ADD and OP_RELU working together
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pxvm.core.interpreter import run_program
from .make_add_relu_test import make_add_relu_test_image, save_pxi


def load_pxi(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    return np.array(img, dtype=np.uint8)


def main() -> None:
    print("="*60)
    print("pxVM ADD + RELU TEST")
    print("="*60)
    print()

    base = Path(__file__).parent
    pxi_path = base / "add_relu_test.pxi"

    if not pxi_path.exists():
        img = make_add_relu_test_image()
        save_pxi(pxi_path, img)
        print(f"Generated: {pxi_path}")
    else:
        print(f"Loading: {pxi_path}")

    img = load_pxi(pxi_path)

    print(f"Image size: {img.shape[1]}×{img.shape[0]} RGBA")
    print()

    print("Input vectors:")
    print(f"  Row 1 (A): [{img[1,0,0]}, {img[1,1,0]}, {img[1,2,0]}]")
    print(f"  Row 2 (B): [{img[2,0,0]}, {img[2,1,0]}, {img[2,2,0]}]")
    print()

    # Execute
    print("Executing program...")
    result_img = run_program(img)
    print()

    # Read results from row 3
    result_values = [
        int(result_img[3, 0, 0]),
        int(result_img[3, 1, 0]),
        int(result_img[3, 2, 0]),
    ]

    print("="*60)
    print("RESULT")
    print("="*60)
    print(f"Row 3 (output): {result_values}")
    print()

    # Verify
    expected = [15, 35, 55]  # [10+5, 20+15, 30+25]
    if result_values == expected:
        print(f"✅ CORRECT (expected {expected})")
    else:
        print(f"❌ INCORRECT (expected {expected}, got {result_values})")

    print("="*60)


if __name__ == "__main__":
    main()
