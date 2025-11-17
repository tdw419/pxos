#!/usr/bin/env python3
"""
pxvm/core/opcodes.py

Opcode definitions for pxVM.

v0.0.1: OP_HALT, OP_DOT_RGB (numeric opcodes)
v0.0.2: OP_ADD, OP_RELU, OP_MATMUL (numeric opcodes)
v0.2.0: ASCII opcodes - self-documenting machine code

Design Philosophy (v0.2.0):
--------------------------
Opcodes are printable ASCII characters, making programs self-documenting
and enabling visual inspection via font rendering.

Each opcode is both:
  - A machine instruction (executable byte)
  - A visual glyph (renderable character)

This unifies the machine layer and visual layer:
  Program bytes = Readable text = Renderable glyphs

Instruction Format:
------------------
Instructions are stored in row 0, pixel format:
  R channel: Opcode (ASCII character)
  G channel: Argument 1 (typically row address)
  B channel: Argument 2 (typically row address)
  A channel: Argument 3 (typically row address)

Example:
  Pixel [77, 5, 7, 9, 255] = 'M' @ row5 @ row7 @ row9
  → MatMul(row5, row7, row9)

When viewed as text: "M@5@7@9"
When executed: Matrix multiply instruction

ASCII Opcode Map:
----------------
Control:
  H (72)  = Halt

Neural Network Ops:
  M (77)  = MatMul  - Matrix multiply
  A (65)  = Add     - Element-wise add
  R (82)  = ReLU    - ReLU activation
  D (68)  = Dot     - Dot product

Visual Ops:
  B (66)  = Blit    - Blit glyph (future)

Special:
  @ (64)  = Separator (for visual formatting)
  0-9     = Numeric literals
"""

# ============================================================================
# ASCII Opcode Definitions (v0.2.0)
# ============================================================================

# Control flow
OP_HALT: int = ord('H')  # 72 - Halt execution

# Neural network primitives
OP_MATMUL: int = ord('M')  # 77 - Matrix multiply
OP_ADD: int = ord('A')     # 65 - Element-wise addition
OP_RELU: int = ord('R')    # 82 - ReLU activation
OP_DOT: int = ord('D')     # 68 - Dot product

# Visual operations (future)
OP_BLIT_GLYPH: int = ord('B')  # 66 - Blit glyph to framebuffer

# Special markers
ARG_SEPARATOR: int = ord('@')  # 64 - Visual separator

# ============================================================================
# Legacy Opcode Support (v0.0.x compatibility)
# ============================================================================

# Map old numeric opcodes to new ASCII opcodes for migration
LEGACY_OPCODE_MAP = {
    0: OP_HALT,      # 0 → 'H' (72)
    1: OP_DOT,       # 1 → 'D' (68)
    2: OP_ADD,       # 2 → 'A' (65)
    3: OP_RELU,      # 3 → 'R' (82)
    4: OP_MATMUL,    # 4 → 'M' (77)
}

# Reverse map for converting back
ASCII_TO_LEGACY = {v: k for k, v in LEGACY_OPCODE_MAP.items()}

# ============================================================================
# Opcode Metadata
# ============================================================================

# Map opcodes to human-readable names
OPCODE_NAMES = {
    OP_HALT: "Halt",
    OP_MATMUL: "MatMul",
    OP_ADD: "Add",
    OP_RELU: "ReLU",
    OP_DOT: "Dot",
    OP_BLIT_GLYPH: "BlitGlyph",
}

# Map opcodes to single characters
OPCODE_CHARS = {
    OP_HALT: 'H',
    OP_MATMUL: 'M',
    OP_ADD: 'A',
    OP_RELU: 'R',
    OP_DOT: 'D',
    OP_BLIT_GLYPH: 'B',
}

# Reverse lookup: character to opcode
CHAR_TO_OPCODE = {v: k for k, v in OPCODE_CHARS.items()}

# ============================================================================
# Utility Functions
# ============================================================================

def opcode_to_char(opcode: int) -> str:
    """
    Convert opcode to its character representation.

    Args:
        opcode: Opcode byte value

    Returns:
        Character representation (or '?' if unknown)
    """
    return OPCODE_CHARS.get(opcode, '?')


def char_to_opcode(char: str) -> int:
    """
    Convert character to opcode byte value.

    Args:
        char: Character ('M', 'A', 'R', etc.)

    Returns:
        Opcode byte value

    Raises:
        KeyError: If character is not a valid opcode
    """
    return CHAR_TO_OPCODE[char]


def opcode_name(opcode: int) -> str:
    """
    Get human-readable name for opcode.

    Args:
        opcode: Opcode byte value

    Returns:
        Name like "MatMul", "Add", etc.
    """
    return OPCODE_NAMES.get(opcode, f"Unknown({opcode})")


def is_legacy_opcode(opcode: int) -> bool:
    """Check if opcode is from legacy numeric format."""
    return opcode in LEGACY_OPCODE_MAP


def migrate_legacy_opcode(opcode: int) -> int:
    """
    Convert legacy numeric opcode to ASCII opcode.

    Args:
        opcode: Legacy opcode (0-4)

    Returns:
        ASCII opcode (65-82)
    """
    return LEGACY_OPCODE_MAP.get(opcode, opcode)


def format_instruction(opcode: int, arg1: int, arg2: int, arg3: int) -> str:
    """
    Format instruction as human-readable string.

    Args:
        opcode: Opcode byte
        arg1, arg2, arg3: Instruction arguments

    Returns:
        Formatted string like "M@5@7@9" or "MatMul(5, 7, 9)"
    """
    char = opcode_to_char(opcode)
    name = opcode_name(opcode)

    # Visual format (for display)
    visual = f"{char}@{arg1}@{arg2}@{arg3}"

    # Readable format
    readable = f"{name}({arg1}, {arg2}, {arg3})"

    return f"{visual}  # {readable}"


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Opcodes
    "OP_HALT",
    "OP_MATMUL",
    "OP_ADD",
    "OP_RELU",
    "OP_DOT",
    "OP_BLIT_GLYPH",

    # Special
    "ARG_SEPARATOR",

    # Metadata
    "OPCODE_NAMES",
    "OPCODE_CHARS",
    "CHAR_TO_OPCODE",

    # Utilities
    "opcode_to_char",
    "char_to_opcode",
    "opcode_name",
    "format_instruction",

    # Legacy support
    "LEGACY_OPCODE_MAP",
    "is_legacy_opcode",
    "migrate_legacy_opcode",
]
