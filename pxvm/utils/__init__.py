"""
pxvm.utils - Utilities for working with pixel programs.

These utilities help construct and validate pixel programs
while maintaining pixels as the primary representation.

DO NOT use these to abstract away pixels. Use them to work
WITH pixels more reliably.
"""

from .layout import (
    calculate_image_size,
    allocate_rows,
    write_matrix,
    read_matrix,
)

from .validation import (
    validate_matrix_header,
    validate_program_structure,
)

__all__ = [
    "calculate_image_size",
    "allocate_rows",
    "write_matrix",
    "read_matrix",
    "validate_matrix_header",
    "validate_program_structure",
]
