#!/usr/bin/env python3
"""
pxvm/dev/assembler.py

Pixel Program Assembler - Symbolic DSL for writing .pxi programs.

Philosophy:
- Development is done SYMBOLICALLY (readable, maintainable)
- Canonical storage is VISUAL (.pxi pixels)
- This tool COMPILES TO pixels, never REPLACES them
- Assembler is a LENS to write pixels, not a layer

The Development Lifecycle:
1. Write: Assembler (symbolic DSL → pixels)
2. Store: .pxi file (canonical format)
3. Debug: Inspector (pixels → human-readable)
4. Execute: pxVM (pixels → computation)

Usage:
    from pxvm.dev import PixelAssembler

    asm = PixelAssembler(image_width=1024)
    asm.define_matrix("W_hidden", W_hidden)
    asm.define_matrix("b_hidden", b_hidden)
    asm.MATMUL("h", "h_in", "W_hidden")
    asm.ADD("h", "h", "b_hidden")
    asm.RELU("h")
    asm.HALT()

    program = asm.compile()
    Image.fromarray(program).save("program.pxi")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

from pxvm.core.opcodes import OP_HALT, OP_MATMUL, OP_ADD, OP_RELU, OP_DOT_RGB
from pxvm.utils.layout import calculate_matrix_rows, write_matrix
from pxvm.debug.constraints import validate_addressing_constraints, validate_matmul_structure


class PixelAssembler:
    """
    Symbolic assembler for pixel programs.

    Two-pass compilation:
    1. Layout pass: Calculate row allocations for all matrices
    2. Encoding pass: Write instructions and matrices to pixels
    """

    def __init__(self, image_width: int = 1024):
        """
        Initialize assembler.

        Args:
            image_width: Width of output pixel image (default 1024)
        """
        self.image_width = image_width

        # Symbolic state
        self.matrices: Dict[str, np.ndarray] = {}  # name → data
        self.instructions: List[Tuple] = []  # (opcode, arg0, arg1, arg2)

        # Layout state (computed during compile)
        self.layout: Dict[str, int] = {}  # name → row_start
        self.total_height: int = 0

    def define_matrix(self, name: str, data: np.ndarray, quantize: bool = True):
        """
        Register a matrix/vector for the program.

        Args:
            name: Symbolic name (e.g., "W_hidden", "h_in")
            data: NumPy array (1D or 2D)
            quantize: If True, quantize float32 → uint8
        """
        if name in self.matrices:
            raise ValueError(f"Matrix '{name}' already defined")

        # Ensure 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim != 2:
            raise ValueError(f"Matrix '{name}' must be 1D or 2D, got {data.ndim}D")

        self.matrices[name] = data

    def MATMUL(self, output_name: str, input_a: str, input_b: str):
        """
        Symbolic MATMUL: output = input_a @ input_b

        Args:
            output_name: Name for output matrix (will be allocated)
            input_a: Name of first input matrix
            input_b: Name of second input matrix
        """
        # Validate inputs exist
        if input_a not in self.matrices:
            raise ValueError(f"Matrix '{input_a}' not defined")
        if input_b not in self.matrices:
            raise ValueError(f"Matrix '{input_b}' not defined")

        # Compute output shape
        A = self.matrices[input_a]
        B = self.matrices[input_b]

        rows_a, cols_a = A.shape
        rows_b, cols_b = B.shape

        if cols_a != rows_b:
            raise ValueError(
                f"MATMUL shape mismatch: {input_a}={rows_a}×{cols_a}, "
                f"{input_b}={rows_b}×{cols_b} (need cols_a == rows_b)"
            )

        # Allocate output matrix (will be computed by pxVM)
        # Initialize as zeros for layout calculation
        output = np.zeros((rows_a, cols_b), dtype=np.uint8)
        self.matrices[output_name] = output

        # Record instruction (args will be resolved during layout pass)
        self.instructions.append((OP_MATMUL, input_a, input_b, output_name))

    def ADD(self, output_name: str, input_a: str, input_b: str):
        """
        Symbolic ADD: output = input_a + input_b

        Args:
            output_name: Name for output matrix (will be allocated if new)
            input_a: Name of first input matrix
            input_b: Name of second input matrix
        """
        if input_a not in self.matrices:
            raise ValueError(f"Matrix '{input_a}' not defined")
        if input_b not in self.matrices:
            raise ValueError(f"Matrix '{input_b}' not defined")

        A = self.matrices[input_a]
        B = self.matrices[input_b]

        if A.shape != B.shape:
            raise ValueError(
                f"ADD shape mismatch: {input_a}={A.shape}, {input_b}={B.shape}"
            )

        # If output doesn't exist, allocate it
        if output_name not in self.matrices:
            self.matrices[output_name] = np.zeros_like(A, dtype=np.uint8)

        self.instructions.append((OP_ADD, input_a, input_b, output_name))

    def RELU(self, target_name: str):
        """
        Symbolic RELU: target = relu(target) (in-place)

        Args:
            target_name: Name of matrix to apply ReLU to
        """
        if target_name not in self.matrices:
            raise ValueError(f"Matrix '{target_name}' not defined")

        # RELU uses only arg0, rest are unused (set to 0)
        self.instructions.append((OP_RELU, target_name, None, None))

    def DOT_RGB(self, output_name: str, input_a: str, input_b: str):
        """
        Symbolic DOT_RGB: output = dot(input_a, input_b)

        Args:
            output_name: Name for output vector
            input_a: Name of first input vector
            input_b: Name of second input vector
        """
        if input_a not in self.matrices:
            raise ValueError(f"Matrix '{input_a}' not defined")
        if input_b not in self.matrices:
            raise ValueError(f"Matrix '{input_b}' not defined")

        A = self.matrices[input_a]
        B = self.matrices[input_b]

        # Both must be vectors (1 row)
        if A.shape[0] != 1 or B.shape[0] != 1:
            raise ValueError(
                f"DOT_RGB requires vectors: {input_a}={A.shape}, {input_b}={B.shape}"
            )

        if A.shape[1] != B.shape[1]:
            raise ValueError(
                f"DOT_RGB length mismatch: {input_a}={A.shape[1]}, {input_b}={B.shape[1]}"
            )

        # Output is scalar (1×1)
        if output_name not in self.matrices:
            self.matrices[output_name] = np.zeros((1, 1), dtype=np.uint8)

        self.instructions.append((OP_DOT_RGB, input_a, input_b, output_name))

    def HALT(self):
        """Add HALT instruction (terminates program)."""
        self.instructions.append((OP_HALT, None, None, None))

    def compile(self, quantize: bool = True) -> np.ndarray:
        """
        Compile symbolic program to pixel array.

        Two-pass compilation:
        1. Layout pass: Allocate row positions for all matrices
        2. Encoding pass: Write instructions and data to pixels

        Args:
            quantize: If True, quantize float32 matrices to uint8

        Returns:
            RGBA pixel array ready to save as .pxi
        """
        # === PASS 1: Layout Calculation ===

        # Row 0 is always instructions
        current_row = 1

        # Allocate rows for each matrix in dependency order
        # (Matrices are added to self.matrices as they're referenced)
        for name, data in self.matrices.items():
            if name not in self.layout:
                # data.shape = (rows, cols) in numpy convention
                rows_needed = calculate_matrix_rows(data.shape[1], data.shape[0], self.image_width)
                self.layout[name] = current_row
                current_row += rows_needed

        self.total_height = current_row

        # === Validate Addressing Constraints ===

        # Check uint8 limit (max row = 255)
        max_row = max(self.layout.values())
        if max_row > 255:
            raise ValueError(
                f"Program exceeds uint8 addressing limit: "
                f"max row = {max_row}, limit = 255"
            )

        # === PASS 2: Pixel Encoding ===

        # Allocate pixel array
        img = np.zeros((self.total_height, self.image_width, 4), dtype=np.uint8)

        # Encode instructions (row 0)
        for i, instr in enumerate(self.instructions):
            if i >= self.image_width:
                raise ValueError(
                    f"Too many instructions ({len(self.instructions)}), "
                    f"max width = {self.image_width}"
                )

            opcode = instr[0]

            if opcode == OP_HALT:
                img[0, i] = [OP_HALT, 0, 0, 0]
            elif opcode == OP_RELU:
                # RELU: arg0 = target row, rest unused
                target_name = instr[1]
                row = self.layout[target_name]
                img[0, i] = [OP_RELU, row, 0, 0]
            else:
                # MATMUL, ADD, DOT_RGB: resolve symbolic names to rows
                arg0_name, arg1_name, arg2_name = instr[1], instr[2], instr[3]

                row_a = self.layout[arg0_name]
                row_b = self.layout[arg1_name]
                row_c = self.layout[arg2_name]

                img[0, i] = [opcode, row_a, row_b, row_c]

        # Encode matrices
        for name, row_start in self.layout.items():
            data = self.matrices[name]
            write_matrix(img, row_start, data, quantize=quantize)

        # === Validation ===

        # Validate addressing constraints
        errors = validate_addressing_constraints(img)
        if errors:
            raise ValueError(
                f"Program validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        # Validate MATMUL structures
        for i, instr in enumerate(self.instructions):
            if instr[0] == OP_MATMUL:
                # Resolve to actual pixel instruction
                arg0_name, arg1_name, arg2_name = instr[1], instr[2], instr[3]
                row_a = self.layout[arg0_name]
                row_b = self.layout[arg1_name]
                row_c = self.layout[arg2_name]

                instr_pixel = np.array([OP_MATMUL, row_a, row_b, row_c], dtype=np.uint8)
                error = validate_matmul_structure(img, i, instr_pixel)
                if error:
                    raise ValueError(error)

        return img


def compile_pixellm_program(weights_path: Path, output_path: Path, image_width: int = 1024):
    """
    Example: Compile Pixel-LLM forward pass using the assembler.

    This demonstrates how to use the assembler for a real neural network.

    Args:
        weights_path: Path to pixellm_v0.npz weights
        output_path: Path to save compiled .pxi program
        image_width: Width of output image
    """
    print("=" * 70)
    print(" PIXEL ASSEMBLER - Compiling Pixel-LLM Forward Pass")
    print("=" * 70)
    print()

    # Load weights
    print(f"Loading weights: {weights_path}")
    weights = np.load(weights_path)

    W_hidden = weights["hidden"]  # [128, 128]
    b_hidden = weights["b_hidden"]  # [128]
    W_out = weights["out"]  # [128, 1024]
    b_out = weights["b_out"]  # [1024]

    print(f"  W_hidden: {W_hidden.shape}")
    print(f"  b_hidden: {b_hidden.shape}")
    print(f"  W_out: {W_out.shape}")
    print(f"  b_out: {b_out.shape}")
    print()

    # Create assembler
    print("Initializing assembler...")
    asm = PixelAssembler(image_width=image_width)

    # Define matrices
    print("Defining matrices...")
    asm.define_matrix("h_in", np.zeros((1, 128)))  # Input (will be overwritten at runtime)
    asm.define_matrix("W_hidden", W_hidden)
    asm.define_matrix("b_hidden", b_hidden.reshape(1, -1))
    asm.define_matrix("W_out", W_out)
    asm.define_matrix("b_out", b_out.reshape(1, -1))

    # Define computation
    print("Assembling instructions...")
    asm.MATMUL("h", "h_in", "W_hidden")      # h = h_in @ W_hidden
    asm.ADD("h", "h", "b_hidden")             # h = h + b_hidden
    asm.RELU("h")                              # h = relu(h)
    asm.MATMUL("logits", "h", "W_out")       # logits = h @ W_out
    asm.ADD("logits", "logits", "b_out")     # logits = logits + b_out
    asm.HALT()

    print(f"  Total instructions: {len(asm.instructions)}")
    print(f"  Total matrices: {len(asm.matrices)}")
    print()

    # Compile
    print("Compiling to pixels...")
    program = asm.compile(quantize=True)

    print(f"  Program dimensions: {program.shape[1]}×{program.shape[0]} RGBA")
    print(f"  Total size: {program.nbytes:,} bytes")
    print()

    # Show layout
    print("Memory layout:")
    for name, row in sorted(asm.layout.items(), key=lambda x: x[1]):
        shape = asm.matrices[name].shape
        print(f"  row {row:3d}: {name:12s} {shape[1]:4d}×{shape[0]:4d}")
    print()

    # Save
    print(f"Saving: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(program).save(output_path, format="PNG")

    file_size = output_path.stat().st_size
    print(f"  File size: {file_size:,} bytes")
    print()

    print("=" * 70)
    print(" COMPILATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Next steps:")
    print(f"  1. Inspect: python3 -m pxvm.dev.inspector {output_path}")
    print(f"  2. Validate: python3 -m pxvm.examples.validate_pixellm_accuracy")
    print()


def main():
    """CLI interface for assembler."""
    import sys

    root = Path(__file__).resolve().parents[2]
    weights_path = root / "pixel_llm" / "models" / "pixellm_v0.npz"
    output_path = root / "pixel_llm" / "programs" / "pixellm_forward_compiled.pxi"

    if not weights_path.exists():
        print(f"ERROR: Weights not found: {weights_path}")
        print("Run: python3 pixel_llm/models/pixellm_v0_train.py")
        return

    compile_pixellm_program(weights_path, output_path)


if __name__ == "__main__":
    main()
