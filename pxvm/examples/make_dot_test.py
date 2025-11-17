#!/usr/bin/env python3
"""
pxvm/examples/make_dot_test.py

Generate a 16x16 RGBA .pxi image that encodes a single OP_DOT_RGB program.

Layout:

- Row 0:
    (0,0) = OP_DOT_RGB  (opcode=1, rowA=1, rowB=2, rowOut=3)
    (1,0) = OP_HALT
- Row 1: vector A = [1,2,3,4] (R channel)
- Row 2: vector B = [10,20,30,40] (R channel)
- Row 3: output row (initially zeros; result written to (0,3))

Expected dot product:

    dot = 1*10 + 2*20 + 3*30 + 4*40 = 300
    300 decimal = 0x012C
    → (R,G) = (0x2C, 0x01) = (44, 1)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pxvm.core.opcodes import OP_DOT_RGB, OP_HALT


def make_dot_test_image(width: int = 16, height: int = 16) -> np.ndarray:
    img = np.zeros((height, width, 4), dtype=np.uint8)

    # Instruction: OP_DOT_RGB at (0,0)
    #   opcode = 1
    #   rowA   = 1
    #   rowB   = 2
    #   rowOut = 3
    img[0, 0] = np.array([OP_DOT_RGB, 1, 2, 3], dtype=np.uint8)

    # Instruction: HALT at (1,0)
    img[0, 1] = np.array([OP_HALT, 0, 0, 0], dtype=np.uint8)

    # Row 1: vector A = [1,2,3,4]
    img[1, 0] = [1, 0, 0, 0]
    img[1, 1] = [2, 0, 0, 0]
    img[1, 2] = [3, 0, 0, 0]
    img[1, 3] = [4, 0, 0, 0]

    # Row 2: vector B = [10,20,30,40]
    img[2, 0] = [10, 0, 0, 0]
    img[2, 1] = [20, 0, 0, 0]
    img[2, 2] = [30, 0, 0, 0]
    img[2, 3] = [40, 0, 0, 0]

    # Row 3: output row (all zeros initially)
    #   result will be written to (0,3) by interpreter

    return img


def save_pxi(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="RGBA").save(path, format="PNG")


def main() -> None:
    out_path = Path(__file__).parent / "dot_test.pxi"
    img = make_dot_test_image()
    save_pxi(out_path, img)
    print(f"dot_test.pxi written to: {out_path}")


if __name__ == "__main__":
    main()
