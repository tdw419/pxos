#!/usr/bin/env python3
"""
pxvm/tests/test_add_relu.py

Unit tests for OP_ADD and OP_RELU opcodes.
"""

from __future__ import annotations

import numpy as np

from pxvm.core.interpreter import run_program
from pxvm.examples.make_add_relu_test import make_add_relu_test_image


def test_op_add_basic():
    """Test OP_ADD with simple vectors."""
    img = make_add_relu_test_image()
    height, width, _ = img.shape
    assert height >= 4 and width >= 16

    # Run program
    out = run_program(img)

    # Result should be at row 3: [15, 35, 55]
    # (10+5, 20+15, 30+25)
    assert int(out[3, 0, 0]) == 15
    assert int(out[3, 1, 0]) == 35
    assert int(out[3, 2, 0]) == 55


def test_op_relu_basic():
    """Test OP_RELU (no-op for uint8, but tests the structure)."""
    img = make_add_relu_test_image()

    # Run program (includes RELU after ADD)
    out = run_program(img)

    # After RELU, values should be unchanged (uint8 is always >= 0)
    assert int(out[3, 0, 0]) == 15
    assert int(out[3, 1, 0]) == 35
    assert int(out[3, 2, 0]) == 55


def test_op_add_clamping():
    """Test that OP_ADD clamps to uint8 range."""
    from pxvm.core.opcodes import OP_ADD, OP_HALT

    img = np.zeros((16, 16, 4), dtype=np.uint8)

    # Instruction: ADD row 1 + row 2 → row 3
    img[0, 0] = [OP_ADD, 1, 2, 3]
    img[0, 1] = [OP_HALT, 0, 0, 0]

    # Row 1: [200, 100, 50]
    img[1, 0] = [200, 0, 0, 0]
    img[1, 1] = [100, 0, 0, 0]
    img[1, 2] = [50, 0, 0, 0]

    # Row 2: [100, 200, 50]
    img[2, 0] = [100, 0, 0, 0]
    img[2, 1] = [200, 0, 0, 0]
    img[2, 2] = [50, 0, 0, 0]

    # Run
    out = run_program(img)

    # Row 3 should be: [255 (clamped), 255 (clamped), 100]
    assert int(out[3, 0, 0]) == 255  # 200+100=300 → 255 (clamped)
    assert int(out[3, 1, 0]) == 255  # 100+200=300 → 255 (clamped)
    assert int(out[3, 2, 0]) == 100  # 50+50=100
