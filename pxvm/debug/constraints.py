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

from pxvm.core.opcodes import OP_HALT, OP_MATMUL, OP_ADD, OP_RELU, OP_DOT_RGB

def validate_addressing_constraints(img: np.ndarray) -> List[str]:
    """
    Checks if all instruction arguments (row indices) fit in the uint8 space (0-255).
    """
    errors = []
    # Check rows used in instruction row 0
    instructions = img[0, :, :]
    max_row_used = 0

    for x, instr in enumerate(instructions):
        opcode = int(instr[0])
        if opcode == OP_HALT: break

        args = instr[1:]

        for arg in args:
            if arg > 255:
                errors.append(f"Instruction {x} ({opcode}): Row index {arg} exceeds uint8 limit (255).")
            if arg > max_row_used:
                max_row_used = arg

    # Check against image height
    if max_row_used >= img.shape[0]:
        errors.append(f"Program height ({img.shape[0]}) is insufficient for max row index ({max_row_used}) used in instructions.")

    return errors

def validate_matmul_structure(img: np.ndarray, x: int, instr: np.ndarray) -> List[str]:
    """
    Checks if MATMUL arguments refer to matrices with compatible inner dimensions.
    """
    # This is a stub. A real implementation would read the matrix headers
    # and verify that the inner dimensions match.
    return []
