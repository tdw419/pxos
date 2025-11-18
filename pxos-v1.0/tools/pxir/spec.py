"""
pxIR - pxOS Intermediate Representation

High-level, typed, SSA-based IR for pxOS compilation.

This is the **semantic** IR layer that sits above PXI Assembly.
It supports:
- Static Single Assignment (SSA) form
- Explicit typing (i8, i16, i32, f32, mat, vec, ptr)
- Matrix operations as first-class citizens
- Block-structured control flow
- Optimization passes

Architecture:
    Python/C → pxIR (this layer) → PXI Assembly → Primitives → Binary
                 ↑
           Typed, SSA, optimizable
"""

# Core IR types
PXIR_TYPES = [
    "i8",         # 8-bit integer
    "i16",        # 16-bit integer
    "i32",        # 32-bit integer
    "f32",        # 32-bit float
    "bool",       # Boolean
    "vec",        # Vector (homogeneous array)
    "mat",        # Matrix (2D array)
    "ptr",        # Pointer with address space
]

# Core operations
PXIR_OPS = [
    # Arithmetic
    "ADD", "SUB", "MUL", "DIV", "MOD",
    "NEG", "ABS",

    # Matrix operations
    "MATMUL",     # Matrix multiply
    "RELU",       # ReLU activation
    "SOFTMAX",    # Softmax
    "ARGMAX",     # Argmax
    "SAMPLE",     # Sample from distribution

    # Memory operations
    "LOAD",       # Load from memory
    "STORE",      # Store to memory
    "ALLOC",      # Allocate memory

    # Data movement
    "CAST",       # Type cast
    "PACK",       # Pack into struct/tuple
    "UNPACK",     # Unpack from struct/tuple

    # Control flow
    "BR",         # Conditional branch
    "JMP",        # Unconditional jump
    "RET",        # Return from function
    "CALL",       # Function call

    # Comparison
    "EQ", "NE", "LT", "LE", "GT", "GE",

    # Logical
    "AND", "OR", "NOT", "XOR",

    # System calls
    "SYSCALL",    # Generic syscall

    # pxOS-specific I/O
    "PRINT_STR",  # Print string
    "DRAW_GLYPH", # Draw glyph to framebuffer
    "BLIT",       # Blit pixels
    "READ_KEY",   # Read keyboard input
]

# Memory address spaces
MEM_SPACES = [
    "mem",         # General RAM
    "framebuffer", # Video memory
    "io",          # I/O ports
]

# Matrix element types
MAT_ELEM_TYPES = ["i8", "i16", "f32"]

# Quantization schemes
QUANT_SCHEMES = [
    "none",        # No quantization
    "linear",      # Linear quantization (scale, zero_point)
    "symmetric",   # Symmetric quantization
]
