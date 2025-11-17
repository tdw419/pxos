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

    # Infer vector length (includes header at column 0)
    length = _infer_vector_length(img, row_a, row_b)

    # Compute dot product over data columns (skip column 0 header)
    dot_val = 0
    for x in range(1, length):
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
    Execute OP_ADD: element-wise addition with quantization.

    instr: single RGBA uint8 pixel from row 0.
    Format: (OP_ADD, row_a, row_b, row_out)
    Semantics: row_out = row_a + row_b (element-wise)

    Process:
    1. Read quantized matrices A, B and dequantize to float32
    2. Compute C = A + B in float32 (prevents saturation)
    3. Quantize result C with new scale/offset
    4. Write quantized C to output row
    """
    from pxvm.utils.layout import read_quantized_matrix, write_quantized_matrix

    height, width, _ = img.shape

    row_a = int(instr[1])
    row_b = int(instr[2])
    row_out = int(instr[3])

    if not (0 <= row_a < height and 0 <= row_b < height and 0 <= row_out < height):
        return  # Out-of-bounds, skip

    # Read and dequantize matrices to float32
    A = read_quantized_matrix(img, row_a)
    B = read_quantized_matrix(img, row_b)

    # Validate shapes for element-wise addition
    if A.shape != B.shape:
        return  # Invalid shapes: must be identical

    # Compute C = A + B in float32 (prevents saturation)
    C = A + B

    # Write quantized result with new scale/offset
    write_quantized_matrix(img, row_out, C)


def _exec_relu(
    img: np.ndarray,
    instr: np.ndarray,
) -> None:
    """
    Execute OP_RELU: in-place ReLU activation with quantization.

    instr: single RGBA uint8 pixel from row 0.
    Format: (OP_RELU, row_data, 0, 0)
    Semantics: row_data = max(row_data, 0) (in-place, element-wise)

    Process:
    1. Read quantized matrix and dequantize to float32
    2. Apply ReLU: clamp negative values to zero
    3. Quantize result with new scale/offset
    4. Write quantized result back to same row
    """
    from pxvm.utils.layout import read_quantized_matrix, write_quantized_matrix

    height, width, _ = img.shape

    row_data = int(instr[1])

    if not (0 <= row_data < height):
        return  # Out-of-bounds, skip

    # Read and dequantize matrix to float32
    data = read_quantized_matrix(img, row_data)

    # Apply ReLU: max(data, 0)
    data_relu = np.maximum(data, 0.0)

    # Write quantized result back (in-place)
    write_quantized_matrix(img, row_data, data_relu)


def _read_shape(img: np.ndarray, row_start: int) -> tuple[int, int]:
    """
    Read matrix shape from header pixel at (0, row_start).

    Header format: (cols_low, cols_high, rows_low, rows_high)
    Returns: (cols, rows)
    """
    hdr = img[row_start, 0]
    cols = int(hdr[0]) + (int(hdr[1]) << 8)
    rows = int(hdr[2]) + (int(hdr[3]) << 8)
    return cols, rows


def _get_matrix_val(
    img: np.ndarray,
    row_start: int,
    cols: int,
    rows: int,
    idx: int,
) -> int:
    """
    Get value from flattened matrix data.

    Data begins at column 1, row-major layout.
    Wraps across image width, continues on subsequent rows.
    """
    width = img.shape[1]
    stride = width - 1  # Column 0 is header

    x = 1 + (idx % stride)
    y = row_start + (idx // stride)

    return int(img[y, x, 0])


def _set_matrix_val(
    img: np.ndarray,
    row_start: int,
    cols: int,
    rows: int,
    idx: int,
    val: int,
) -> None:
    """
    Set value in flattened matrix data.

    Data begins at column 1, row-major layout.
    Wraps across image width, continues on subsequent rows.
    """
    width = img.shape[1]
    stride = width - 1  # Column 0 is header

    x = 1 + (idx % stride)
    y = row_start + (idx // stride)

    img[y, x, 0] = val


def _exec_matmul(
    img: np.ndarray,
    instr: np.ndarray,
) -> None:
    """
    Execute OP_MATMUL: matrix multiply with quantization.

    instr: single RGBA uint8 pixel from row 0.
    Format: (OP_MATMUL, row_A_start, row_B_start, row_C_start)
    Semantics: C = A @ B (matrix multiplication)

    Quantized matrix encoding:
    - Header at (0, row_start): (cols_low, cols_high, rows_low, rows_high)
    - Metadata at (1-2, row_start): (scale, offset) as float32
    - Data starts at column 3, row-major flattened

    Process:
    1. Read quantized matrices A, B and dequantize to float32
    2. Compute C = A @ B in float32 (prevents saturation)
    3. Quantize result C with new scale/offset
    4. Write quantized C to output row
    """
    from pxvm.utils.layout import read_quantized_matrix, write_quantized_matrix

    height, width, _ = img.shape

    row_a = int(instr[1])
    row_b = int(instr[2])
    row_c = int(instr[3])

    if not (0 <= row_a < height and 0 <= row_b < height and 0 <= row_c < height):
        return  # Out-of-bounds, skip

    # Read and dequantize matrices to float32
    A = read_quantized_matrix(img, row_a)  # M×K
    B = read_quantized_matrix(img, row_b)  # K×N

    # Validate shapes for matrix multiplication
    if A.shape[1] != B.shape[0]:
        return  # Invalid shapes: A.cols != B.rows

    # Compute C = A @ B in float32 (prevents saturation)
    C = A @ B  # M×N

    # Write quantized result with new scale/offset
    write_quantized_matrix(img, row_c, C)


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
        elif opcode == OP_MATMUL:
            _exec_matmul(img, instr)

        # NEXT instruction
        pc_x += 1
        steps += 1

    return img


__all__ = ["run_program"]
