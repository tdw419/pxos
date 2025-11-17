#!/usr/bin/env python3
"""
pxvm/examples/encode_pixellm_forward.py

Encode Pixel-LLM forward pass as a pxVM program.

Takes trained Pixel-LLM weights and creates a .pxi program that:
1. Loads input embedding vector
2. Executes hidden layer: h = relu(input @ W_hidden + b_hidden)
3. Executes output layer: logits = h @ W_out + b_out

Architecture (from pixellm_v0_train.py):
- Vocab: 1024 tokens
- Model dim: 128
- Embedding: [1024, 128]
- Hidden: [128, 128] + bias [128]
- Output: [128, 1024] + bias [1024]

Program layout:
- Row 0: Instructions [MATMUL, ADD, RELU, MATMUL, ADD, HALT]
- Row 1: h_in [1×128] (input embedding vector)
- Row 2: W_hidden [128×128] (hidden weights as matrix)
- Row 3: b_hidden [1×128] (hidden bias as vector)
- Row 4: h [1×128] (hidden activations, output)
- Row 5: W_out [128×1024] (output weights as matrix)
- Row 6: b_out [1×1024] (output bias as vector)
- Row 7: logits [1×1024] (final output)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pxvm.core.opcodes import OP_MATMUL, OP_ADD, OP_RELU, OP_HALT
from pxvm.utils.layout import write_matrix, calculate_image_size, allocate_rows


def create_pixellm_forward_program(
    weights_path: Path,
    output_path: Path,
    image_width: int = 256,
) -> None:
    """
    Create pxVM program for Pixel-LLM forward pass.

    Image height is calculated automatically based on weight sizes.

    Args:
        weights_path: Path to pixellm_v0.npz
        output_path: Path to output .pxi program
        image_width: Width of output image (height auto-calculated)
    """
    # Load weights
    print(f"Loading weights from: {weights_path}")
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

    # Calculate required image size
    print("Calculating image size...")
    matrices = {
        'h_in': (128, 1),
        'W_hidden': tuple(reversed(W_hidden.shape)),  # (cols, rows)
        'b_hidden': (128, 1),
        'h': (128, 1),
        'W_out': tuple(reversed(W_out.shape)),
        'b_out': (1024, 1),
        'logits': (1024, 1),
    }

    width, height = calculate_image_size(matrices, instruction_count=6, width=image_width)
    print(f"  Image size: {width}×{height}")

    # Allocate row positions for each matrix
    rows = allocate_rows(matrices, width=width, start_row=1)
    print(f"  Row allocation:")
    for name, row in rows.items():
        print(f"    {name:12s} @ row {row}")
    print()

    # Create image
    img = np.zeros((height, width, 4), dtype=np.uint8)

    # Row 0: Instructions
    print("Encoding instructions (row 0)...")

    # Instruction sequence using allocated rows:
    # 1. MATMUL: h = h_in @ W_hidden
    # 2. ADD: h = h + b_hidden
    # 3. RELU: h = relu(h)
    # 4. MATMUL: logits = h @ W_out
    # 5. ADD: logits = logits + b_out
    # 6. HALT

    r_h_in = rows['h_in']
    r_W_hidden = rows['W_hidden']
    r_b_hidden = rows['b_hidden']
    r_h = rows['h']
    r_W_out = rows['W_out']
    r_b_out = rows['b_out']
    r_logits = rows['logits']

    img[0, 0] = [OP_MATMUL, r_h_in, r_W_hidden, r_h]  # h = h_in @ W_hidden
    img[0, 1] = [OP_ADD, r_h, r_b_hidden, r_h]        # h += b_hidden
    img[0, 2] = [OP_RELU, r_h, 0, 0]                  # h = relu(h)
    img[0, 3] = [OP_MATMUL, r_h, r_W_out, r_logits]   # logits = h @ W_out
    img[0, 4] = [OP_ADD, r_logits, r_b_out, r_logits] # logits += b_out
    img[0, 5] = [OP_HALT, 0, 0, 0]

    # Encode matrices using allocated row positions
    print(f"Encoding h_in placeholder (row {r_h_in})...")
    write_matrix(img, row_start=r_h_in, data=np.zeros(128, dtype=np.float32), quantize=False)

    print(f"Encoding W_hidden matrix (row {r_W_hidden})...")
    write_matrix(img, row_start=r_W_hidden, data=W_hidden, quantize=True)

    print(f"Encoding b_hidden vector (row {r_b_hidden})...")
    write_matrix(img, row_start=r_b_hidden, data=b_hidden, quantize=True)

    print(f"Encoding h placeholder (row {r_h})...")
    write_matrix(img, row_start=r_h, data=np.zeros(128, dtype=np.float32), quantize=False)

    print(f"Encoding W_out matrix (row {r_W_out})...")
    write_matrix(img, row_start=r_W_out, data=W_out, quantize=True)

    print(f"Encoding b_out vector (row {r_b_out})...")
    write_matrix(img, row_start=r_b_out, data=b_out, quantize=True)

    print(f"Encoding logits placeholder (row {r_logits})...")
    write_matrix(img, row_start=r_logits, data=np.zeros(1024, dtype=np.float32), quantize=False)

    # Save
    print()
    print(f"Saving program to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="RGBA").save(output_path, format="PNG")

    print(f"✅ Pixel-LLM forward pass encoded ({img.shape[1]}×{img.shape[0]} PNG)")
    print(f"   File size: {output_path.stat().st_size} bytes")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    weights_path = root / "pixel_llm" / "models" / "pixellm_v0.npz"
    output_path = root / "pixel_llm" / "programs" / "pixellm_forward.pxi"

    if not weights_path.exists():
        print(f"❌ Weights not found: {weights_path}")
        print("   Run: python3 pixel_llm/models/pixellm_v0_train.py")
        return

    # Use width=1024 to keep row addresses within uint8 range (0-255)
    create_pixellm_forward_program(weights_path, output_path, image_width=1024)


if __name__ == "__main__":
    main()
