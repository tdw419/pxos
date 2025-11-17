#!/usr/bin/env python3
"""
pxvm/examples/make_add_relu_test.py

Generate a test program demonstrating OP_ADD and OP_RELU opcodes.

Program:
1. Add two vectors: [10, 20, 30] + [5, 15, 25] = [15, 35, 55]
2. Apply ReLU (no-op for uint8, but demonstrates the pattern)

Layout:
- Row 0: [OP_ADD, 1, 2, 3] [OP_RELU, 3, 0, 0] [OP_HALT]
- Row 1: [10, 20, 30]
- Row 2: [5, 15, 25]
- Row 3: (output) [0, 0, 0] → [15, 35, 55] after execution
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pxvm.core.opcodes import OP_ADD, OP_RELU, OP_HALT


def make_add_relu_test_image(width: int = 16, height: int = 16) -> np.ndarray:
    img = np.zeros((height, width, 4), dtype=np.uint8)

    # Row 0: Instructions
    # (0,0): OP_ADD - add row 1 + row 2 → row 3
    img[0, 0] = np.array([OP_ADD, 1, 2, 3], dtype=np.uint8)

    # (1,0): OP_RELU - apply ReLU to row 3 (in-place)
    img[0, 1] = np.array([OP_RELU, 3, 0, 0], dtype=np.uint8)

    # (2,0): OP_HALT
    img[0, 2] = np.array([OP_HALT, 0, 0, 0], dtype=np.uint8)

    # Row 1: Vector A = [10, 20, 30]
    img[1, 0] = [10, 0, 0, 0]
    img[1, 1] = [20, 0, 0, 0]
    img[1, 2] = [30, 0, 0, 0]

    # Row 2: Vector B = [5, 15, 25]
    img[2, 0] = [5, 0, 0, 0]
    img[2, 1] = [15, 0, 0, 0]
    img[2, 2] = [25, 0, 0, 0]

    # Row 3: Output (initially zeros, will contain [15, 35, 55])

    return img


def save_pxi(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="RGBA").save(path, format="PNG")


def main() -> None:
    out_path = Path(__file__).parent / "add_relu_test.pxi"
    img = make_add_relu_test_image()
    save_pxi(out_path, img)
    print(f"add_relu_test.pxi written to: {out_path}")


if __name__ == "__main__":
    main()
