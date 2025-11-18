#!/usr/bin/env python3
"""
pixel_llm/export_to_pixels.py

Export trained model weights to pixel program format.
"""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# This is a placeholder for the real assembler.
# I'll need to create this module next.
from pxvm.dev.assembler import PixelAssembler

def main():
    parser = argparse.ArgumentParser(description="Export trained model weights to a pixel program.")
    parser.add_argument("--input", type=Path, default="pixel_llm/models/tiny_ref_v2.npz", help="Path to the trained model weights.")
    parser.add_argument("--output", type=Path, default="pixel_llm/programs/tiny_ref_v2.pxi", help="Path to save the compiled pixel program.")
    args = parser.parse_args()

    print(f"--- Exporting {args.input} to {args.output} ---")

    # 1. Load weights
    try:
        weights = np.load(args.input)
    except FileNotFoundError:
        print(f"Error: Weights not found at {args.input}")
        return

    W_hidden = weights["hidden"]
    b_hidden = weights["hidden_bias"].reshape(1, -1)
    W_out = weights["output"]
    b_out = weights["output_bias"].reshape(1, -1)

    assembler = PixelAssembler(image_width=1024)

    # 2. Define all matrices and placeholders
    assembler.define_matrix("h_in", np.zeros((1, HIDDEN_DIM), dtype=np.uint8))
    assembler.define_matrix("W_hidden", W_hidden)
    assembler.define_matrix("b_hidden", b_hidden)
    assembler.define_matrix("h", np.zeros((1, HIDDEN_DIM), dtype=np.uint8))
    assembler.define_matrix("W_out", W_out)
    assembler.define_matrix("b_out", b_out)
    assembler.define_matrix("logits", np.zeros((1, weights["output"].shape[1]), dtype=np.uint8))

    # 3. Assemble Instructions
    assembler.MATMUL("h", "h_in", "W_hidden")
    assembler.ADD("h", "h", "b_hidden")
    assembler.RELU("h")
    assembler.MATMUL("logits", "h", "W_out")
    assembler.ADD("logits", "logits", "b_out")
    assembler.HALT()

    # 4. Compile to Pixels
    try:
        program_pixels = assembler.compile()
        print("--- Assembly Successful ---")
    except ValueError as e:
        print(f"\n--- FATAL COMPILATION ERROR ---")
        print(e)
        return

    # 5. Save the program
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(program_pixels, mode="RGBA").save(args.output, format="PNG")
    print(f"\n✅ Program compiled and saved to: {args.output}")

if __name__ == "__main__":
    # A bit of a hack to get the HIDDEN_DIM from the training script
    from pixel_llm.train_tiny_ref_model import HIDDEN_DIM
    main()
