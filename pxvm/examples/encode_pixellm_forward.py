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


def encode_matrix_header(cols: int, rows: int) -> np.ndarray:
    """
    Encode matrix shape as header pixel.

    Format: (cols_low, cols_high, rows_low, rows_high)
    """
    return np.array([
        cols & 0xFF,
        (cols >> 8) & 0xFF,
        rows & 0xFF,
        (rows >> 8) & 0xFF,
    ], dtype=np.uint8)


def encode_matrix_data(
    data: np.ndarray,
    img: np.ndarray,
    row_start: int,
    quantize: bool = True,
) -> None:
    """
    Encode matrix data into image starting at row_start.

    Data is flattened row-major and written starting at column 1.
    Wraps across image width to subsequent rows.

    Args:
        data: 2D numpy array (or 1D for vectors)
        img: RGBA image array to write into
        row_start: Starting row index
        quantize: If True, quantize float32 to uint8 (0-255)
    """
    height, width, _ = img.shape

    # Flatten data (row-major)
    flat = data.flatten()

    if quantize:
        # Quantize float32 to uint8
        # Simple linear mapping: min→0, max→255
        data_min = flat.min()
        data_max = flat.max()

        if data_max - data_min > 0:
            flat_q = ((flat - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        else:
            flat_q = np.zeros_like(flat, dtype=np.uint8)
    else:
        flat_q = flat.astype(np.uint8)

    # Write data starting at column 1, wrapping rows
    stride = width - 1  # Column 0 is header

    for i, val in enumerate(flat_q):
        x = 1 + (i % stride)
        y = row_start + (i // stride)

        if y >= height:
            break  # Out of space

        img[y, x, 0] = val  # R channel only
        # G, B, A remain 0


def create_pixellm_forward_program(
    weights_path: Path,
    output_path: Path,
    image_size: tuple[int, int] = (256, 256),
) -> None:
    """
    Create pxVM program for Pixel-LLM forward pass.

    Args:
        weights_path: Path to pixellm_v0.npz
        output_path: Path to output .pxi program
        image_size: (width, height) of output image
    """
    width, height = image_size

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

    # Create image
    img = np.zeros((height, width, 4), dtype=np.uint8)

    # Row 0: Instructions
    print("Encoding instructions (row 0)...")

    # Instruction sequence:
    # 1. MATMUL: h = h_in @ W_hidden  (row 1 @ row 2 → row 4)
    # 2. ADD: h = h + b_hidden  (row 4 + row 3 → row 4)
    # 3. RELU: h = relu(h)  (row 4 in-place)
    # 4. MATMUL: logits = h @ W_out  (row 4 @ row 5 → row 7)
    # 5. ADD: logits = logits + b_out  (row 7 + row 6 → row 7)
    # 6. HALT

    img[0, 0] = [OP_MATMUL, 1, 2, 4]   # h = h_in @ W_hidden
    img[0, 1] = [OP_ADD, 4, 3, 4]      # h += b_hidden
    img[0, 2] = [OP_RELU, 4, 0, 0]     # h = relu(h)
    img[0, 3] = [OP_MATMUL, 4, 5, 7]   # logits = h @ W_out
    img[0, 4] = [OP_ADD, 7, 6, 7]      # logits += b_out
    img[0, 5] = [OP_HALT, 0, 0, 0]

    # Row 1: h_in placeholder (will be filled at inference time)
    # Header: cols=128, rows=1 (vector)
    print("Encoding h_in placeholder (row 1)...")
    img[1, 0] = encode_matrix_header(128, 1)
    # Data left as zeros (filled at inference)

    # Row 2: W_hidden [128×128]
    print("Encoding W_hidden matrix (row 2)...")
    img[2, 0] = encode_matrix_header(128, 128)
    encode_matrix_data(W_hidden, img, 2, quantize=True)

    # Row 3: b_hidden [128] (as 1×128 vector)
    print("Encoding b_hidden vector (row 3)...")
    img[3, 0] = encode_matrix_header(128, 1)
    encode_matrix_data(b_hidden.reshape(1, -1), img, 3, quantize=True)

    # Row 4: h placeholder (computed)
    print("Encoding h placeholder (row 4)...")
    img[4, 0] = encode_matrix_header(128, 1)

    # Row 5: W_out [128×1024]
    print("Encoding W_out matrix (row 5)...")
    img[5, 0] = encode_matrix_header(1024, 128)
    encode_matrix_data(W_out, img, 5, quantize=True)

    # Row 6: b_out [1024] (as 1×1024 vector)
    print("Encoding b_out vector (row 6)...")
    img[6, 0] = encode_matrix_header(1024, 1)
    encode_matrix_data(b_out.reshape(1, -1), img, 6, quantize=True)

    # Row 7: logits placeholder (computed)
    print("Encoding logits placeholder (row 7)...")
    img[7, 0] = encode_matrix_header(1024, 1)

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

    create_pixellm_forward_program(weights_path, output_path)


if __name__ == "__main__":
    main()
