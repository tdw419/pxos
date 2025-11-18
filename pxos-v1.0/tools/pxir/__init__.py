"""
pxIR package - High-level typed IR for pxOS

This package provides a proper compiler IR with:
- SSA (Static Single Assignment) form
- Explicit typing (i8, i16, i32, f32, mat, vec, ptr)
- Matrix operations as first-class citizens
- Block-structured control flow
- Optimization passes

Usage:
    from pxir import IRBuilder, Type, Program
    from pxir.python_frontend import PythonToPxIR
    from pxir.codegen_pxi import PxIRToPXI

    # Build IR manually
    builder = IRBuilder()
    entry = builder.create_block("entry")
    builder.set_insert_point(entry)

    # Or compile from Python
    compiler = PythonToPxIR()
    program = compiler.compile("def main(): print('Hello')")

    # Generate code
    codegen = PxIRToPXI()
    pxi_assembly = codegen.generate(program)
"""

from .spec import (
    PXIR_TYPES,
    PXIR_OPS,
    MEM_SPACES,
    MAT_ELEM_TYPES,
    QUANT_SCHEMES,
)

from .ir import (
    Type,
    TypeKind,
    Value,
    Op,
    Block,
    Program,
    IRBuilder,
)

__all__ = [
    # Spec
    'PXIR_TYPES',
    'PXIR_OPS',
    'MEM_SPACES',
    'MAT_ELEM_TYPES',
    'QUANT_SCHEMES',

    # IR
    'Type',
    'TypeKind',
    'Value',
    'Op',
    'Block',
    'Program',
    'IRBuilder',
]

__version__ = '1.0.0'
