#!/usr/bin/env python3
"""
pxvm/examples/make_matmul_test.py

Generate a test program demonstrating OP_MATMUL.

Matrices:
A (2×3):
  [[1, 2, 3],
   [4, 5, 6]]

B (3×2):
  [[7,  8],
   [9, 10],
   [11, 12]]

Expected C = A @ B (2×2):
  [[58,  64],
   [139, 154]]

Layout:
- Row 0: [OP_MATMUL, 1, 4, 8] [OP_HALT]
- Row 1: A header (cols=3, rows=2)
- Row 2-3: A data [1,2,3,4,5,6]
- Row 4: B header (cols=2, rows=3)
- Row 5-6: B data [7,8,9,10,11,12]
- Row 8: C header (cols=2, rows=2)
- Row 9: C data (computed) [58,64,139,154]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pxvm.core.opcodes import OP_MATMUL, OP_HALT


def make_matmul_test_image(width: int = 16, height: int = 16) -> np.ndarray:
    img = np.zeros((height, width, 4), dtype=np.uint8)

    # Row 0: Instructions
    # (0,0): OP_MATMUL - multiply A @ B → C
    # A at row 1, B at row 4, C at row 8
    img[0, 0] = np.array([OP_MATMUL, 1, 4, 8], dtype=np.uint8)

    # (1,0): OP_HALT
    img[0, 1] = np.array([OP_HALT, 0, 0, 0], dtype=np.uint8)

    # Matrix A (2×3) at row 1
    # Header: (cols=3, rows=2)
    img[1, 0] = [3, 0, 2, 0]  # (cols_low, cols_high, rows_low, rows_high)

    # Data: [[1,2,3], [4,5,6]] row-major starting at column 1
    img[1, 1] = [1, 0, 0, 0]
    img[1, 2] = [2, 0, 0, 0]
    img[1, 3] = [3, 0, 0, 0]
    img[1, 4] = [4, 0, 0, 0]
    img[1, 5] = [5, 0, 0, 0]
    img[1, 6] = [6, 0, 0, 0]

    # Matrix B (3×2) at row 4
    # Header: (cols=2, rows=3)
    img[4, 0] = [2, 0, 3, 0]  # (cols_low, cols_high, rows_low, rows_high)

    # Data: [[7,8], [9,10], [11,12]] row-major starting at column 1
    img[4, 1] = [7, 0, 0, 0]
    img[4, 2] = [8, 0, 0, 0]
    img[4, 3] = [9, 0, 0, 0]
    img[4, 4] = [10, 0, 0, 0]
    img[4, 5] = [11, 0, 0, 0]
    img[4, 6] = [12, 0, 0, 0]

    # Matrix C (2×2) at row 8 - will be computed
    # Header will be written by OP_MATMUL
    # Data will be computed

    return img


def save_pxi(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="RGBA").save(path, format="PNG")


def main() -> None:
    out_path = Path(__file__).parent / "matmul_test.pxi"
    img = make_matmul_test_image()
    save_pxi(out_path, img)
    print(f"matmul_test.pxi written to: {out_path}")


if __name__ == "__main__":
    main()
