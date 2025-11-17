#!/usr/bin/env python3
"""
pxvm/core/opcodes.py

Opcode definitions for the minimal pxVM.

This first version only defines:

- OP_HALT    (0)
- OP_DOT_RGB (1)  # integer dot product over R-channel of two rows
"""

# Halt execution
OP_HALT: int = 0

# Integer dot-product over RGB row data (we use R channel only in v0)
OP_DOT_RGB: int = 1

__all__ = [
    "OP_HALT",
    "OP_DOT_RGB",
]
