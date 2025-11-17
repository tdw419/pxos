"""
pxvm.debug - Validation and Debugging Utilities

Philosophy:
- Validate pixel programs for correctness
- Detect common encoding errors
- Provide clear error messages
- Never modify pixels (read-only)

Tools:
- constraints: Validate addressing, shape compatibility, bounds
"""

from .constraints import validate_addressing_constraints, validate_matmul_structure

__all__ = ["validate_addressing_constraints", "validate_matmul_structure"]
