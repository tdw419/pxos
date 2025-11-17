"""
pxvm.core - Core pxVM components

Includes:
- Opcode definitions (v0.2.0: ASCII opcodes)
- Interpreter
"""

from .opcodes import (
    OP_HALT,
    OP_MATMUL,
    OP_ADD,
    OP_RELU,
    OP_DOT,
    OP_BLIT_GLYPH,
)
from .interpreter import run_program

# Legacy alias for backward compatibility
OP_DOT_RGB = OP_DOT

__all__ = [
    "OP_HALT",
    "OP_MATMUL",
    "OP_ADD",
    "OP_RELU",
    "OP_DOT",
    "OP_DOT_RGB",  # Legacy alias
    "OP_BLIT_GLYPH",
    "run_program",
]
