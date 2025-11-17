#!/usr/bin/env python3
"""
pxvm/tests/test_matmul.py

Unit tests for OP_MATMUL opcode.
"""

from __future__ import annotations

import numpy as np

from pxvm.core.interpreter import run_program, _read_shape, _get_matrix_val
from pxvm.examples.make_matmul_test import make_matmul_test_image


def test_op_matmul_basic():
    """Test OP_MATMUL with 2×3 @ 3×2 matrices."""
    img = make_matmul_test_image()

    # Run program
    out = run_program(img)

    # Read result matrix C at row 8
    cols_c, rows_c = _read_shape(out, 8)
    assert cols_c == 2
    assert rows_c == 2

    # Extract values
    c_vals = []
    for i in range(rows_c * cols_c):
        c_vals.append(_get_matrix_val(out, 8, cols_c, rows_c, i))

    # Expected: [[58,64], [139,154]]
    expected = [58, 64, 139, 154]
    assert c_vals == expected


def test_op_matmul_identity():
    """Test OP_MATMUL with identity matrix."""
    from pxvm.core.opcodes import OP_MATMUL, OP_HALT

    img = np.zeros((16, 16, 4), dtype=np.uint8)

    # Instruction: MATMUL A @ I → C
    img[0, 0] = [OP_MATMUL, 1, 4, 8]
    img[0, 1] = [OP_HALT, 0, 0, 0]

    # Matrix A (2×2): [[5,6], [7,8]]
    img[1, 0] = [2, 0, 2, 0]  # Header: cols=2, rows=2
    img[1, 1] = [5, 0, 0, 0]
    img[1, 2] = [6, 0, 0, 0]
    img[1, 3] = [7, 0, 0, 0]
    img[1, 4] = [8, 0, 0, 0]

    # Matrix I (2×2): [[1,0], [0,1]]
    img[4, 0] = [2, 0, 2, 0]  # Header: cols=2, rows=2
    img[4, 1] = [1, 0, 0, 0]
    img[4, 2] = [0, 0, 0, 0]
    img[4, 3] = [0, 0, 0, 0]
    img[4, 4] = [1, 0, 0, 0]

    # Run
    out = run_program(img)

    # C should equal A
    c_vals = []
    for i in range(4):
        c_vals.append(_get_matrix_val(out, 8, 2, 2, i))

    assert c_vals == [5, 6, 7, 8]


def test_op_matmul_clamping():
    """Test that OP_MATMUL clamps to uint8 range."""
    from pxvm.core.opcodes import OP_MATMUL, OP_HALT

    img = np.zeros((16, 16, 4), dtype=np.uint8)

    # Instruction
    img[0, 0] = [OP_MATMUL, 1, 4, 8]
    img[0, 1] = [OP_HALT, 0, 0, 0]

    # Matrix A (1×2): [[100, 100]]
    img[1, 0] = [2, 0, 1, 0]  # Header: cols=2, rows=1
    img[1, 1] = [100, 0, 0, 0]
    img[1, 2] = [100, 0, 0, 0]

    # Matrix B (2×1): [[2], [2]]
    img[4, 0] = [1, 0, 2, 0]  # Header: cols=1, rows=2
    img[4, 1] = [2, 0, 0, 0]
    img[4, 2] = [2, 0, 0, 0]

    # Run: C = A @ B = [[100*2 + 100*2]] = [[400]] → clamp to 255
    out = run_program(img)

    c_val = _get_matrix_val(out, 8, 1, 1, 0)
    assert c_val == 255  # Clamped


def test_op_matmul_zero():
    """Test OP_MATMUL with zero matrix."""
    from pxvm.core.opcodes import OP_MATMUL, OP_HALT

    img = np.zeros((16, 16, 4), dtype=np.uint8)

    # Instruction
    img[0, 0] = [OP_MATMUL, 1, 4, 8]
    img[0, 1] = [OP_HALT, 0, 0, 0]

    # Matrix A (2×2): [[1,2], [3,4]]
    img[1, 0] = [2, 0, 2, 0]
    img[1, 1] = [1, 0, 0, 0]
    img[1, 2] = [2, 0, 0, 0]
    img[1, 3] = [3, 0, 0, 0]
    img[1, 4] = [4, 0, 0, 0]

    # Matrix B (2×2): all zeros
    img[4, 0] = [2, 0, 2, 0]
    # Data already all zeros

    # Run: C should be all zeros
    out = run_program(img)

    c_vals = []
    for i in range(4):
        c_vals.append(_get_matrix_val(out, 8, 2, 2, i))

    assert c_vals == [0, 0, 0, 0]
