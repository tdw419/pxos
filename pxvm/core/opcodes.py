#!/usr/bin/env python3
"""
pxvm/core/opcodes.py

Opcode definitions for pxVM.

v0.0.1: OP_HALT, OP_DOT_RGB
v0.0.2: OP_ADD, OP_RELU, OP_MATMUL
"""

# Core control flow
OP_HALT: int = 0

# Vector operations (v0.0.1)
OP_DOT_RGB: int = 1  # Integer dot product over R-channel of two rows

# Neural network primitives (v0.0.2)
OP_ADD: int = 2      # Element-wise addition: row_out = row_a + row_b
OP_RELU: int = 3     # In-place ReLU: row[i] = max(row[i], 0)
OP_MATMUL: int = 4   # Matrix multiply: out_row = in_row @ weight_matrix

__all__ = [
    "OP_HALT",
    "OP_DOT_RGB",
    "OP_ADD",
    "OP_RELU",
    "OP_MATMUL",
]
