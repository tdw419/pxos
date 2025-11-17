#!/usr/bin/env python3
"""
pxvm/examples/make_font_code_test.py

T016: Generate a test program using the new Font-Code ASCII opcodes (M, A, R, H).

This program verifies that:
1. The new opcodes are correctly encoded (e.g., OP_MATMUL = ord('M') = 77).
2. The interpreter executes the program identically to the old numeric opcodes.

Program: C = A(2x2) x B(2x2) -> C(2x2) using Font-Codes.
A = [[1, 2], [3, 4]], B = [[1, 0], [0, 1]] (Identity)
Expected Result Matrix C: A @ I = A ([[1, 2], [3, 4]])
"""

from pathlib import Path
import numpy as np
from PIL import Image

# Import ASCII opcodes and quantization utilities
from pxvm.core.opcodes import OP_MATMUL, OP_HALT
from pxvm.utils.layout import write_quantized_matrix


def make_font_code_test_image(width: int = 16, height: int = 16) -> np.ndarray:
    """Create test program using ASCII opcodes with proper quantization."""
    img = np.zeros((height, width, 4), dtype=np.uint8)

    # Matrix row positions
    R_A_START = 1
    R_B_START = 2
    R_C_START = 3

    # --- Instruction Row 0 (Font-Codes) ---
    # Opcode 'M' (MATMUL) = 77, Opcode 'H' (HALT) = 72
    img[0, 0] = [OP_MATMUL, R_A_START, R_B_START, R_C_START]  # M(A, B, C)
    img[0, 1] = [OP_HALT, 0, 0, 0]  # H

    # --- Matrix A: 2x2 ---
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    write_quantized_matrix(img, R_A_START, A)

    # --- Matrix B: 2x2 Identity ---
    B = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    write_quantized_matrix(img, R_B_START, B)

    # --- Matrix C: Output placeholder (will be filled by interpreter) ---
    # No need to initialize

    return img


def save_pxi(path: Path, img: np.ndarray) -> None:
    """Save program as .pxi file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="RGBA").save(path, format="PNG")


def main() -> None:
    """Generate Font-Code test program."""
    out_path = Path(__file__).parent / "font_code_test.pxi"
    img = make_font_code_test_image()
    save_pxi(out_path, img)

    print("=" * 70)
    print(" FONT-CODE TEST PROGRAM GENERATED")
    print("=" * 70)
    print()
    print(f"Output: {out_path}")
    print()
    print("Program:")
    print(f"  Instruction 0: M({img[0, 0, 1]}, {img[0, 0, 2]}, {img[0, 0, 3]})  # MatMul")
    print(f"  Instruction 1: H  # Halt")
    print()
    print("Matrices:")
    print("  A (row 1): [[1, 2], [3, 4]]")
    print("  B (row 3): [[1, 0], [0, 1]]  # Identity")
    print("  C (row 5): [output]")
    print()
    print("Expected: C = A @ I = [[1, 2], [3, 4]]")
    print()


if __name__ == "__main__":
    main()
