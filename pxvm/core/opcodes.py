#!/usr/bin/env python3
"""
pxvm/core/opcodes.py

Opcode definitions for the pxVM.
"""

# Halt execution
OP_HALT: int = 0

# Integer dot-product over RGB row data (we use R channel only in v0)
OP_DOT_RGB: int = 1

# Element-wise addition
OP_ADD: int = 2

# ReLU activation
OP_RELU: int = 3

# Matrix multiplication
OP_MATMUL: int = 4


__all__ = [
    "OP_HALT",
    "OP_DOT_RGB",
    "OP_ADD",
    "OP_RELU",
    "OP_MATMUL",
]
