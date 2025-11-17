#!/usr/bin/env python3
"""
pxvm/dev/assembler.py

The Pixel Assembler: A Development-Time DSL for creating compliant pxVM programs.

This tool serves as the 'write' layer, translating human-readable symbolic assembly
into the canonical, executable .pxi pixel format. The compilation step enforces
all protocol constraints.

Philosophy:
- Programs are written symbolically for convenience.
- Programs are compiled lossily to the pixel format.
- The compiled .pxi file remains the SOVEREIGN executable.
"""
from __future__ import annotations

import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from pxvm.core.opcodes import OP_HALT, OP_DOT_RGB, OP_ADD, OP_RELU, OP_MATMUL
from pxvm.utils.layout import calculate_image_size, allocate_rows, write_matrix
from pxvm.debug.constraints import validate_addressing_constraints, validate_matmul_structure
from pxvm.dev.inspector import OPCODE_MAP # Reuse opcode names

# Define the instruction format for the Assembler's internal queue
# (opcode, arg0_name, arg1_name, arg2_name)
SymbolicInstruction = Tuple[int, str, str, str]

class PixelAssembler:
    """
    Manages the symbolic definition of a pxVM program and compiles it to pixels.
    """
    def __init__(self, image_width: int = 1024):
        # Symbolic State: {name: (rows, cols, data)}
        self.matrices: Dict[str, Tuple[int, int, Optional[np.ndarray]]] = {}
        # Symbolic Instructions: (OP, arg0, arg1, arg2)
        self.instructions: List[SymbolicInstruction] = []
        self.image_width = image_width
        self.row_allocations: Optional[Dict[str, int]] = None

    # --- Matrix Definition DSL ---

    def define_matrix(self, name: str, data: np.ndarray, is_input: bool = False):
        """
        Registers a matrix (weights, bias, or placeholder) for layout calculation.
        """
        if name in self.matrices:
            raise ValueError(f"Matrix '{name}' already defined.")

        if data.ndim != 2:
            raise ValueError(f"Matrix data must be 2D (got {data.ndim}D for '{name}').")

        rows, cols = data.shape
        self.matrices[name] = (rows, cols, data)

    # --- Instruction Assembly DSL ---

    def MATMUL(self, output_name: str, input_a_name: str, input_b_name: str):
        """C = A @ B instruction."""
        self.instructions.append((OP_MATMUL, input_a_name, input_b_name, output_name))

    def ADD(self, output_name: str, input_a_name: str, input_b_name: str):
        """Output = A + B instruction."""
        self.instructions.append((OP_ADD, input_a_name, input_b_name, output_name))

    def RELU(self, target_name: str):
        """Target = ReLU(Target) instruction (in-place)."""
        self.instructions.append((OP_RELU, target_name, "0", "0"))

    def HALT(self):
        """Stop execution."""
        self.instructions.append((OP_HALT, "0", "0", "0"))

    # --- Compilation Logic ---

    def compile(self) -> np.ndarray:
        """
        Performs the two-pass compilation: layout calculation and pixel generation.
        Returns the canonical, executable pixel array.
        """

        # --- PASS 1: Layout and Validation ---

        # 1. Prepare matrix dimensions for layout calculation
        dims = {name: (c, r) for r, c, _ in self.matrices.values()}

        # 2. Calculate minimum image dimensions
        width, height = calculate_image_size(
            matrices=dims,
            instruction_count=len(self.instructions),
            width=self.image_width
        )
        print(f"Compilation Dims: {width}x{height} (Requires {height} rows)")

        if height > 256:
             raise ValueError(f"CRITICAL ERROR: Program requires {height} rows. Cannot be addressed by uint8 (max 256). Try increasing image_width.")

        # 3. Allocate final row indices
        self.row_allocations = allocate_rows(dims, width=width, start_row=1)

        # 4. Generate the empty pixel image
        img = np.zeros((height, width, 4), dtype=np.uint8)

        # 5. Compile instructions into Row 0 and perform MatMul shape validation
        if not self._compile_instructions(img):
            raise ValueError("Compilation failed due to instruction/shape validation.")

        # --- PASS 2: Data Writing ---

        # 6. Write all matrix data into allocated rows
        for name, (rows, cols, data) in self.matrices.items():
            if data is not None:
                row_start = self.row_allocations[name]
                print(f"Writing data for {name} ({rows}x{cols}) to row R{row_start}")
                write_matrix(img, row_start, data, image_width=width)

        # The resulting numpy array is the canonical .pxi executable
        return img

    def _compile_instructions(self, img: np.ndarray) -> bool:
        """Converts symbolic instructions to pixels and performs validation checks."""
        row_alloc = self.row_allocations
        if row_alloc is None: return False

        validation_img = img.copy() # Use a copy for pre-execution validation checks
        validation_errors: List[str] = []

        for x, instr in enumerate(self.instructions):
            opcode, arg0_name, arg1_name, arg2_name = instr

            # Resolve symbolic names to concrete row indices
            arg0 = row_alloc.get(arg0_name, int(arg0_name) if arg0_name.isdigit() else 0)
            arg1 = row_alloc.get(arg1_name, int(arg1_name) if arg1_name.isdigit() else 0)
            arg2 = row_alloc.get(arg2_name, int(arg2_name) if arg2_name.isdigit() else 0)

            # Write pixel
            img[0, x] = np.array([opcode, arg0, arg1, arg2], dtype=np.uint8)

            # MatMul specific validation (must check structure)
            if opcode == OP_MATMUL:
                matmul_instr_pixel = img[0, x] # Use the newly compiled pixel
                errors = validate_matmul_structure(validation_img, x, matmul_instr_pixel)
                validation_errors.extend(errors)

        # Check final addressing constraints (uint8, height bounds)
        addressing_errors = validate_addressing_constraints(img)
        validation_errors.extend(addressing_errors)

        if validation_errors:
            print("\n--- COMPILATION FAILED: Protocol Violations ---")
            for err in validation_errors:
                print(f"❌ {err}")
            return False

        return True


# --- Example Integration (LLM Forward Pass) ---

def compile_pixellm_program(weights_path: Path) -> Optional[np.ndarray]:
    """Compiles the Pixel-LLM forward pass using the Assembler."""
    print("--- Starting Pixel-LLM Assembly ---")

    # 1. Load weights
    try:
        weights = np.load(weights_path)
    except FileNotFoundError:
        print(f"Error: Weights not found at {weights_path}")
        return None

    W_hidden = weights["hidden"]     # [128, 128]
    b_hidden = weights["hidden_bias"].reshape(1, -1) # [1, 128]
    W_out = weights["output"]        # [128, 1024]
    b_out = weights["output_bias"].reshape(1, -1)     # [1, 1024]

    assembler = PixelAssembler(image_width=1024)

    # 2. Define all matrices and placeholders

    # Hidden Layer Inputs/Weights
    assembler.define_matrix("h_in", np.zeros((1, 128), dtype=np.uint8)) # 1x128 Input placeholder
    assembler.define_matrix("W_hidden", W_hidden)                       # 128x128 Weights
    assembler.define_matrix("b_hidden", b_hidden)                       # 1x128 Bias
    assembler.define_matrix("h", np.zeros((1, 128), dtype=np.uint8))    # 1x128 Hidden buffer

    # Output Layer Weights/Outputs
    assembler.define_matrix("W_out", W_out)                             # 128x1024 Weights
    assembler.define_matrix("b_out", b_out)                             # 1x1024 Bias
    assembler.define_matrix("logits", np.zeros((1, 1024), dtype=np.uint8)) # 1x1024 Logits output

    # 3. Assemble Instructions

    # Hidden Layer: h = ReLU(h_in @ W_hidden + b_hidden)
    assembler.MATMUL("h", "h_in", "W_hidden")
    assembler.ADD("h", "h", "b_hidden")
    assembler.RELU("h")

    # Output Layer: logits = h @ W_out + b_out
    assembler.MATMUL("logits", "h", "W_out")
    assembler.ADD("logits", "logits", "b_out")

    # End Program
    assembler.HALT()

    # 4. Compile to Pixels
    try:
        program_pixels = assembler.compile()
        print("--- Assembly Successful ---")
        return program_pixels
    except ValueError as e:
        print(f"\n--- FATAL COMPILATION ERROR ---")
        print(e)
        return None


if __name__ == "__main__":
    from pathlib import Path
    from PIL import Image

    # NOTE: Requires weights to be generated by the training script
    WEIGHTS = Path("pixel_llm/models/pixellm_v0.npz")
    OUTPUT = Path("pixel_llm/programs/pixellm_forward_compiled.pxi")

    pixels = compile_pixellm_program(WEIGHTS)

    if pixels is not None:
        OUTPUT.parent.mkdir(exist_ok=True)
        Image.fromarray(pixels, mode="RGBA").save(OUTPUT, format="PNG")

        print(f"\n✅ Program compiled and saved to: {OUTPUT}")
        print("   Use PixelInspector to verify.")
