"""
pxvm.core - Core pxVM components

Includes:
- Opcode definitions
- Interpreter
"""

from .opcodes import OP_HALT, OP_DOT_RGB
from .interpreter import run_program

__all__ = [
    "OP_HALT",
    "OP_DOT_RGB",
    "run_program",
]
