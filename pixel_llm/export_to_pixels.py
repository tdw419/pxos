#!/usr/bin/env python3
"""
pixel_llm/export_to_pixels.py

Export trained model weights to pixel program format.

This script takes trained weights (from train_tiny_ref_model.py) and compiles
them into a .pxi pixel program using the quantized matrix assembler.

The compiled program can then be used with generate_text.py for semantic
text generation.

Usage:
    python3 pixel_llm/export_to_pixels.py
    python3 pixel_llm/export_to_pixels.py --input models/custom.npz --output programs/custom.pxi
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from PIL import Image

from pxvm.dev.assembler import PixelAssembler


def export_weights_to_pixels(
    weights_path: Path,
    output_path: Path,
    image_width: int = 1024
):
    """
    Export trained weights to pixel program.

    Args:
        weights_path: Path to .npz weights file
        output_path: Path to save compiled .pxi program
        image_width: Width of output image
    """
    print("=" * 70)
    print(" EXPORTING TRAINED WEIGHTS TO PIXEL PROGRAM")
    print("=" * 70)
    print()

    # Load weights
    print(f"Loading weights: {weights_path}")
    weights = np.load(weights_path)

    # The training script saves: embed, hidden, b_hidden, out, b_out
    # But the pixel program only uses: hidden, b_hidden, out, b_out
    # (Embedding is handled by generate_text.py's encode_hidden_state)

    W_hidden = weights["hidden"]  # [128, 128]
    b_hidden = weights["b_hidden"]  # [128]
    W_out = weights["out"]  # [128, vocab_size]
    b_out = weights["b_out"]  # [vocab_size]

    print(f"  W_hidden: {W_hidden.shape}")
    print(f"  b_hidden: {b_hidden.shape}")
    print(f"  W_out: {W_out.shape}")
    print(f"  b_out: {b_out.shape}")
    print()

    # Verify shapes match expected architecture
    hidden_dim = W_hidden.shape[0]
    vocab_size = W_out.shape[1]

    if W_hidden.shape != (hidden_dim, hidden_dim):
        raise ValueError(f"W_hidden shape mismatch: expected ({hidden_dim}, {hidden_dim}), got {W_hidden.shape}")
    if W_out.shape != (hidden_dim, vocab_size):
        raise ValueError(f"W_out shape mismatch: expected ({hidden_dim}, {vocab_size}), got {W_out.shape}")

    print(f"Architecture validated: {hidden_dim}-dim hidden, {vocab_size}-token vocab")
    print()

    # Create assembler
    print("Initializing assembler...")
    asm = PixelAssembler(image_width=image_width)

    # Define matrices
    print("Defining matrices...")
    asm.define_matrix("h_in", np.zeros((1, hidden_dim)))  # Input (will be overwritten at runtime)
    asm.define_matrix("W_hidden", W_hidden)
    asm.define_matrix("b_hidden", b_hidden.reshape(1, -1))
    asm.define_matrix("W_out", W_out)
    asm.define_matrix("b_out", b_out.reshape(1, -1))

    # Define computation (same as pixellm_forward)
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
    print("Compiling to pixels with quantization...")
    program = asm.compile()

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
    print(" EXPORT COMPLETE")
    print("=" * 70)
    print()
    print(f"Next step:")
    print(f"  Generate text: python3 -m pxvm.examples.generate_text --program {output_path.relative_to(Path.cwd())}")
    print()


def main():
    """CLI interface for weight export."""
    parser = argparse.ArgumentParser(
        description="Export trained weights to pixel program"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("pixel_llm/models/tiny_ref.npz"),
        help="Path to .npz weights file (default: tiny_ref.npz)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pixel_llm/programs/tiny_ref.pxi"),
        help="Path to save .pxi program (default: tiny_ref.pxi)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width in pixels (default: 1024)"
    )

    args = parser.parse_args()

    # Resolve paths (go up one level from pixel_llm/ to pxos/)
    root = Path(__file__).resolve().parents[1]
    weights_path = root / args.input
    output_path = root / args.output

    if not weights_path.exists():
        print(f"ERROR: Weights not found: {weights_path}")
        print("Run: python3 pixel_llm/train_tiny_ref_model.py")
        return 1

    export_weights_to_pixels(weights_path, output_path, args.width)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
