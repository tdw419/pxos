#!/usr/bin/env python3
"""
pxvm/utils/validation.py

Validation utilities for pixel programs.

These help verify that pixel programs are well-formed before execution.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pxvm.core.opcodes import OP_HALT, OP_DOT_RGB, OP_ADD, OP_RELU, OP_MATMUL


def validate_matrix_header(
    img: np.ndarray,
    row: int,
    expected_shape: Optional[tuple[int, int]] = None,
) -> tuple[int, int]:
    """
    Validate and read matrix header.

    Args:
        img: RGBA image array
        row: Row to check
        expected_shape: Optional (cols, rows) to verify

    Returns:
        (cols, rows) from header

    Raises:
        ValueError: If header is invalid or doesn't match expected shape
    """
    if row < 0 or row >= img.shape[0]:
        raise ValueError(f"Row {row} out of bounds (height={img.shape[0]})")

    hdr = img[row, 0]
    cols = int(hdr[0]) + (int(hdr[1]) << 8)
    rows = int(hdr[2]) + (int(hdr[3]) << 8)

    if cols == 0 or rows == 0:
        raise ValueError(f"Invalid matrix shape at row {row}: {cols}×{rows}")

    if expected_shape is not None:
        expected_cols, expected_rows = expected_shape
        if (cols, rows) != (expected_cols, expected_rows):
            raise ValueError(
                f"Shape mismatch at row {row}: "
                f"expected {expected_cols}×{expected_rows}, "
                f"got {cols}×{rows}"
            )

    return cols, rows


def validate_instruction(
    img: np.ndarray,
    pc_x: int,
) -> tuple[int, int, int, int]:
    """
    Validate and read instruction.

    Args:
        img: RGBA image array
        pc_x: Instruction column (x coordinate in row 0)

    Returns:
        (opcode, arg0, arg1, arg2)

    Raises:
        ValueError: If instruction is invalid
    """
    if pc_x < 0 or pc_x >= img.shape[1]:
        raise ValueError(f"PC {pc_x} out of bounds (width={img.shape[1]})")

    instr = img[0, pc_x]
    opcode = int(instr[0])
    arg0 = int(instr[1])
    arg1 = int(instr[2])
    arg2 = int(instr[3])

    # Validate opcode
    valid_opcodes = {OP_HALT, OP_DOT_RGB, OP_ADD, OP_RELU, OP_MATMUL}
    if opcode not in valid_opcodes:
        raise ValueError(f"Invalid opcode {opcode} at PC {pc_x}")

    return opcode, arg0, arg1, arg2


def validate_program_structure(
    img: np.ndarray,
    expected_instructions: Optional[list[tuple[int, ...]]] = None,
) -> dict:
    """
    Validate overall pixel program structure.

    Args:
        img: RGBA image array
        expected_instructions: Optional list of (opcode, arg0, arg1, arg2) tuples

    Returns:
        Dict with validation results: {
            'valid': bool,
            'instruction_count': int,
            'errors': list[str],
        }

    Raises:
        ValueError: If program is fundamentally malformed
    """
    errors = []
    height, width, channels = img.shape

    # Check image format
    if channels != 4:
        raise ValueError(f"Image must be RGBA (4 channels), got {channels}")

    # Check for HALT instruction
    halt_found = False
    instruction_count = 0

    for x in range(width):
        instr = img[0, x]
        opcode = int(instr[0])

        if opcode == OP_HALT:
            halt_found = True
            instruction_count = x
            break
        elif opcode in {OP_DOT_RGB, OP_ADD, OP_RELU, OP_MATMUL}:
            instruction_count += 1
        else:
            # Could be end of instructions (all zeros)
            if opcode == 0 and x > 0:
                errors.append(f"Missing HALT instruction before PC {x}")
                break

    if not halt_found:
        errors.append("No HALT instruction found in row 0")

    # Validate expected instructions if provided
    if expected_instructions is not None:
        for i, expected in enumerate(expected_instructions):
            if i >= width:
                errors.append(f"Expected instruction {i} beyond image width")
                continue

            instr = img[0, i]
            actual = tuple(int(instr[j]) for j in range(4))

            if actual != expected:
                errors.append(
                    f"Instruction {i} mismatch: "
                    f"expected {expected}, got {actual}"
                )

    return {
        'valid': len(errors) == 0,
        'instruction_count': instruction_count,
        'errors': errors,
    }


def validate_matmul_compatibility(
    img: np.ndarray,
    row_a: int,
    row_b: int,
) -> bool:
    """
    Check if two matrices are compatible for multiplication.

    Args:
        img: RGBA image array
        row_a: Row of matrix A
        row_b: Row of matrix B

    Returns:
        True if A @ B is valid (i.e., A.cols == B.rows)

    Raises:
        ValueError: If headers are invalid
    """
    cols_a, rows_a = validate_matrix_header(img, row_a)
    cols_b, rows_b = validate_matrix_header(img, row_b)

    # For A @ B: A is M×K, B is K×N
    # So: cols_a must equal rows_b
    return cols_a == rows_b


__all__ = [
    "validate_matrix_header",
    "validate_instruction",
    "validate_program_structure",
    "validate_matmul_compatibility",
]
