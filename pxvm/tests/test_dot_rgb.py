#!/usr/bin/env python3
"""
pxvm/tests/test_dot_rgb.py

Unit test for OP_DOT_RGB execution in pxVM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pxvm.core.interpreter import run_program
from pxvm.examples.make_dot_test import make_dot_test_image


def test_op_dot_rgb_basic():
    img = make_dot_test_image()
    height, width, _ = img.shape
    assert height >= 4 and width >= 4

    # Run program
    out = run_program(img)

    # Result should be at (0,3)
    low = int(out[3, 0, 0])
    high = int(out[3, 0, 1])
    dot_val = low + (high << 8)

    assert dot_val == 300, f"Expected dot=300, got {dot_val}"
