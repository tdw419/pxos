#!/usr/bin/env python3
"""
pxvm/debug/constraints.py

Validation utilities for pixel program constraints.

Philosophy:
- Validate addressing constraints (uint8 limits)
- Validate operation requirements (MATMUL shapes, etc.)
- Provide clear error messages for encoding issues
- Read-only - never modifies pixels
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

from pxvm.core.opcodes import OP_HALT, OP_MATMUL, OP_ADD, OP_RELU, OP_DOT

# Legacy alias
OP_DOT_RGB = OP_DOT
from pxvm.core.interpreter import _read_shape


def validate_addressing_constraints(img: np.ndarray) -> List[str]:
    """
    Validate uint8 addressing constraints for pixel program.

    Constraints:
    - All instruction arguments must be <= 255 (uint8 limit)
    - All row references must be < image height

    Args:
        img: RGBA pixel array (H, W, 4)

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    height, width, channels = img.shape

    # Extract instructions from row 0
    instructions = []
    for x in range(width):
        instr = img[0, x]
        opcode = int(instr[0])

        if opcode == OP_HALT:
            instructions.append((opcode, int(instr[1]), int(instr[2]), int(instr[3])))
            break
        elif opcode in [OP_MATMUL, OP_ADD, OP_RELU, OP_DOT_RGB]:
            instructions.append((opcode, int(instr[1]), int(instr[2]), int(instr[3])))
        elif opcode == 0:
            # Padding, continue
            continue
        else:
            # Unknown opcode, stop
            break

    # Validate each instruction
    for i, (opcode, arg0, arg1, arg2) in enumerate(instructions):
        if opcode == OP_HALT:
            continue

        # Check uint8 range (0-255)
        for arg_num, arg_val in enumerate([arg0, arg1, arg2]):
            if arg_val > 255:
                errors.append(
                    f"Instruction {i}: arg{arg_num}={arg_val} exceeds uint8 limit (255)"
                )

        # Check image height bounds
        if opcode in [OP_MATMUL, OP_ADD, OP_DOT_RGB]:
            # These opcodes use all three arguments as row references
            for arg_num, arg_val in enumerate([arg0, arg1, arg2]):
                if arg_val > 0 and arg_val >= height:
                    errors.append(
                        f"Instruction {i}: arg{arg_num}={arg_val} exceeds image height ({height})"
                    )
        elif opcode == OP_RELU:
            # RELU uses only arg0
            if arg0 > 0 and arg0 >= height:
                errors.append(
                    f"Instruction {i}: arg0={arg0} exceeds image height ({height})"
                )

    return errors


def validate_matmul_structure(
    img: np.ndarray,
    instr_idx: int,
    instr_pixel: np.ndarray
) -> Optional[str]:
    """
    Validate MATMUL instruction shape compatibility.

    For A @ B = C, requires:
    - A.shape = (m, k)
    - B.shape = (k, n)
    - C.shape = (m, n)

    In pixel encoding:
    - A has cols=k, rows=m
    - B has cols=n, rows=k
    - Inner dimensions must match: A.cols == B.rows

    Args:
        img: RGBA pixel array (H, W, 4)
        instr_idx: Instruction index for error messages
        instr_pixel: Instruction pixel (opcode, arg0, arg1, arg2)

    Returns:
        Error message if invalid, None if valid
    """
    opcode, row_a, row_b, row_c = instr_pixel

    if opcode != OP_MATMUL:
        return f"Instruction {instr_idx}: Not a MATMUL opcode ({opcode})"

    # Read matrix shapes
    try:
        cols_a, rows_a = _read_shape(img, row_a)
    except Exception as e:
        return f"Instruction {instr_idx} (MATMUL): Cannot read matrix A at row {row_a}: {e}"

    try:
        cols_b, rows_b = _read_shape(img, row_b)
    except Exception as e:
        return f"Instruction {instr_idx} (MATMUL): Cannot read matrix B at row {row_b}: {e}"

    # Validate shape compatibility
    # For A @ B:
    #   A: (rows_a, cols_a) in numpy convention
    #   B: (rows_b, cols_b) in numpy convention
    #   Need: cols_a == rows_b

    if cols_a != rows_b:
        return (
            f"Instruction {instr_idx} (MATMUL): Shape mismatch - "
            f"A[row{row_a}] has cols={cols_a}, rows={rows_a}; "
            f"B[row{row_b}] has cols={cols_b}, rows={rows_b}; "
            f"incompatible (need A.cols={cols_a} == B.rows={rows_b})"
        )

    # Check output matrix if it exists
    try:
        cols_c, rows_c = _read_shape(img, row_c)

        # Expected: C.shape = (rows_a, cols_b)
        expected_cols = cols_b
        expected_rows = rows_a

        if cols_c != expected_cols or rows_c != expected_rows:
            return (
                f"Instruction {instr_idx} (MATMUL): Output matrix C[row{row_c}] "
                f"has shape {cols_c}×{rows_c}, expected {expected_cols}×{expected_rows}"
            )
    except:
        # Output matrix not yet written, this is fine
        pass

    return None
