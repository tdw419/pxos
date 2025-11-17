#!/usr/bin/env python3
"""
pxvm/core/interpreter.py

Minimal pxVM interpreter implementing OP_DOT_RGB.

Image layout assumptions for v0 test:

- RGBA uint8 image, shape (H, W, 4)
- Row 0: instruction pixels
- Row 1: vector A (values in R channel)
- Row 2: vector B (values in R channel)
- Row 3: output row (result written to (0, 3) as low/high bytes in R/G)

Instruction encoding (per-pixel):

    R = OPCODE
    G = ARG0
    B = ARG1
    A = ARG2

OP_DOT_RGB (opcode = 1):

    R = 1           # opcode
    G = row index of vector A
    B = row index of vector B
    A = row index of output row

Semantics:

    - Treat row G as vector A, row B as vector B
    - Infer vector length by scanning across columns until both A[x] and B[x]
      are all zeros or we hit the image width.
    - Compute integer dot product using R channel only:
          dot = sum_i (A[i].R * B[i].R)
    - Store result in row A (output row) at column 0:
          low  byte -> R
          high byte -> G
      B and A channels can remain 0.

OP_HALT (opcode = 0):

    Stops execution.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .opcodes import OP_HALT, OP_DOT_RGB, OP_ADD, OP_RELU, OP_MATMUL


def _infer_vector_length(
    img: np.ndarray,
    row_a: int,
    row_b: int,
) -> int:
    """
    Infer vector length by scanning row_a and row_b until we hit:
      - both pixels fully zero, or
      - the end of the row
    """
    height, width, _ = img.shape
    assert 0 <= row_a < height
    assert 0 <= row_b < height

    length = 0
    for x in range(width):
        px_a = img[row_a, x]
        px_b = img[row_b, x]
        if (px_a == 0).all() and (px_b == 0).all():
            break
        length += 1
    return length


def _exec_dot_rgb(
    img: np.ndarray,
    instr: np.ndarray,
) -> None:
    """
    Execute OP_DOT_RGB on the given image in-place.

    instr: single RGBA uint8 pixel from row 0.
    """
    height, width, _ = img.shape

    row_a = int(instr[1])  # G
    row_b = int(instr[2])  # B
    row_out = int(instr[3])  # A

    if not (0 <= row_a < height and 0 <= row_b < height and 0 <= row_out < height):
        # Out-of-bounds rows: do nothing for now (could log/flag later)
        return

    length = _infer_vector_length(img, row_a, row_b)

    dot_val = 0
    for x in range(length):
        a_r = int(img[row_a, x, 0])  # R channel only
        b_r = int(img[row_b, x, 0])
        dot_val += a_r * b_r

    # Encode dot product as little-endian 16-bit integer in (R,G)
    low = dot_val & 0xFF
    high = (dot_val >> 8) & 0xFF

    img[row_out, 0, 0] = low
    img[row_out, 0, 1] = high
    # B and A can remain 0 for now


def _exec_add(
    img: np.ndarray,
    instr: np.ndarray,
) -> None:
    """
    Execute OP_ADD: element-wise addition.

    instr: single RGBA uint8 pixel from row 0.
    Format: (OP_ADD, row_a, row_b, row_out)
    Semantics: row_out[i] = row_a[i] + row_b[i] for all i (R channel)
    """
    height, width, _ = img.shape

    row_a = int(instr[1])    # G
    row_b = int(instr[2])    # B
    row_out = int(instr[3])  # A

    if not (0 <= row_a < height and 0 <= row_b < height and 0 <= row_out < height):
        return  # Out-of-bounds, skip

    # Infer vector length
    length = _infer_vector_length(img, row_a, row_b)

    # Element-wise addition (R channel only)
    for x in range(length):
        a_r = int(img[row_a, x, 0])
        b_r = int(img[row_b, x, 0])
        sum_val = a_r + b_r

        # Clamp to uint8 range
        sum_val = min(255, max(0, sum_val))

        img[row_out, x, 0] = sum_val
        # G, B, A channels remain 0


def _exec_relu(
    img: np.ndarray,
    instr: np.ndarray,
) -> None:
    """
    Execute OP_RELU: in-place ReLU activation.

    instr: single RGBA uint8 pixel from row 0.
    Format: (OP_RELU, row_data, 0, 0)
    Semantics: row_data[i] = max(row_data[i], 0) for all i (R channel, in-place)
    """
    height, width, _ = img.shape

    row_data = int(instr[1])  # G

    if not (0 <= row_data < height):
        return  # Out-of-bounds, skip

    # Apply ReLU to all non-zero pixels in row (R channel)
    # Since we're using uint8, negative values don't exist, so this is a no-op
    # But we keep the structure for future float support
    for x in range(width):
        val = int(img[row_data, x, 0])
        # For uint8: ReLU(x) = max(x, 0) = x (since x >= 0)
        # This becomes meaningful when we support signed/float values
        img[row_data, x, 0] = max(0, val)


def run_program(
    img: np.ndarray,
    max_steps: int = 1024,
) -> np.ndarray:
    """
    Run a pxVM program encoded in an RGBA uint8 image.

    - Instructions are read from row 0, columns increasing from x=0.
    - Each instruction is one pixel.
    - Execution stops when:
        - OP_HALT is encountered, or
        - max_steps is exceeded, or
        - we reach the end of the row.

    Returns the modified image (same object, also mutated in-place).
    """
    assert img.ndim == 3 and img.shape[2] == 4, "Expected RGBA image"

    height, width, _ = img.shape

    pc_x = 0
    pc_y = 0  # instruction row

    steps = 0

    while steps < max_steps and pc_x < width:
        instr = img[pc_y, pc_x]
        opcode = int(instr[0])

        if opcode == OP_HALT:
            break

        if opcode == OP_DOT_RGB:
            _exec_dot_rgb(img, instr)
        elif opcode == OP_ADD:
            _exec_add(img, instr)
        elif opcode == OP_RELU:
            _exec_relu(img, instr)
        # OP_MATMUL deferred to later

        # NEXT instruction
        pc_x += 1
        steps += 1

    return img


__all__ = ["run_program"]
