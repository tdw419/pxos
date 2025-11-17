#!/usr/bin/env python3
"""
pxvm/examples/run_matmul_test.py

Execute matmul_test.pxi and verify the results.

Expected C = A @ B:
A (2×3) @ B (3×2) = C (2×2)
  [[58,  64],
   [139, 154]]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pxvm.core.interpreter import run_program, _read_shape, _get_matrix_val
from .make_matmul_test import make_matmul_test_image, save_pxi


def load_pxi(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    return np.array(img, dtype=np.uint8)


def main() -> None:
    print("="*60)
    print("pxVM MATMUL TEST")
    print("="*60)
    print()

    base = Path(__file__).parent
    pxi_path = base / "matmul_test.pxi"

    if not pxi_path.exists():
        img = make_matmul_test_image()
        save_pxi(pxi_path, img)
        print(f"Generated: {pxi_path}")
    else:
        print(f"Loading: {pxi_path}")

    img = load_pxi(pxi_path)

    print(f"Image size: {img.shape[1]}×{img.shape[0]} RGBA")
    print()

    # Show input matrices
    print("Input matrices:")

    # Matrix A at row 1
    cols_a, rows_a = _read_shape(img, 1)
    print(f"  A ({rows_a}×{cols_a}): ", end="")
    a_vals = []
    for i in range(rows_a * cols_a):
        a_vals.append(_get_matrix_val(img, 1, cols_a, rows_a, i))
    print(a_vals)

    # Matrix B at row 4
    cols_b, rows_b = _read_shape(img, 4)
    print(f"  B ({rows_b}×{cols_b}): ", end="")
    b_vals = []
    for i in range(rows_b * cols_b):
        b_vals.append(_get_matrix_val(img, 4, cols_b, rows_b, i))
    print(b_vals)
    print()

    # Execute
    print("Executing program...")
    result_img = run_program(img)
    print()

    # Read results from row 8 (C matrix)
    cols_c, rows_c = _read_shape(result_img, 8)
    print("="*60)
    print("RESULT")
    print("="*60)
    print(f"C ({rows_c}×{cols_c}): ", end="")
    c_vals = []
    for i in range(rows_c * cols_c):
        c_vals.append(_get_matrix_val(result_img, 8, cols_c, rows_c, i))
    print(c_vals)
    print()

    # Reshape and display as matrix
    C = np.array(c_vals).reshape(rows_c, cols_c)
    print("C as matrix:")
    print(C)
    print()

    # Verify
    # A = [[1,2,3], [4,5,6]]
    # B = [[7,8], [9,10], [11,12]]
    # C = [[58,64], [139,154]]
    expected = np.array([[58, 64], [139, 154]], dtype=np.int32)

    if np.array_equal(C, expected):
        print(f"✅ CORRECT")
    else:
        print(f"❌ INCORRECT")
        print(f"Expected:\n{expected}")

    print("="*60)


if __name__ == "__main__":
    main()
