#!/usr/bin/env python3
"""
pxvm/examples/run_pixellm_forward.py

Execute the Pixel-LLM forward pass as a pxVM program.

This demonstrates the ultimate achievement: a neural network whose
entire forward pass is encoded as pixels and executed by pxVM.

The program:
1. Loads pixellm_forward.pxi (weights + program in one image)
2. Fills in h_in (input embedding vector)
3. Runs the program (MATMUL, ADD, RELU, MATMUL, ADD, HALT)
4. Reads logits output

This is computation as pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pxvm.core.interpreter import run_program, _read_shape, _get_matrix_val


def load_pxi(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    return np.array(img, dtype=np.uint8)


def set_input_vector(img: np.ndarray, row: int, data: np.ndarray) -> None:
    """
    Set input vector values in the image.

    Args:
        img: RGBA image array
        row: Row index where vector is stored
        data: 1D numpy array of values (will be quantized to uint8)
    """
    width = img.shape[1]
    stride = width - 1  # Column 0 is header

    # Quantize to uint8
    data_min = data.min()
    data_max = data.max()

    if data_max - data_min > 0:
        data_q = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
    else:
        data_q = np.zeros_like(data, dtype=np.uint8)

    # Write to R channel starting at column 1
    for i, val in enumerate(data_q):
        x = 1 + (i % stride)
        y = row + (i // stride)
        img[y, x, 0] = val


def extract_output_vector(img: np.ndarray, row: int, length: int) -> np.ndarray:
    """
    Extract output vector from image.

    Args:
        img: RGBA image array
        row: Row index where vector is stored
        length: Expected vector length

    Returns:
        1D numpy array of uint8 values
    """
    width = img.shape[1]
    stride = width - 1

    output = np.zeros(length, dtype=np.uint8)

    for i in range(length):
        x = 1 + (i % stride)
        y = row + (i // stride)
        output[i] = img[y, x, 0]

    return output


def main() -> None:
    print("="*60)
    print("pxVM PIXEL-LLM FORWARD PASS")
    print("="*60)
    print()

    root = Path(__file__).resolve().parents[2]
    program_path = root / "pixel_llm" / "programs" / "pixellm_forward.pxi"

    if not program_path.exists():
        print(f"❌ Program not found: {program_path}")
        print("   Run: python3 pxvm/examples/encode_pixellm_forward.py")
        return

    # Load program
    print(f"Loading program: {program_path}")
    img = load_pxi(program_path)
    print(f"  Image size: {img.shape[1]}×{img.shape[0]} RGBA")
    print(f"  File size: {program_path.stat().st_size:,} bytes")
    print()

    # Create dummy input (simulate embedding lookup + mean pooling)
    print("Creating input vector (h_in)...")
    h_in = np.random.randn(128).astype(np.float32)  # Random embedding
    h_in = h_in / np.linalg.norm(h_in)  # Normalize
    print(f"  Shape: {h_in.shape}")
    print(f"  Range: [{h_in.min():.3f}, {h_in.max():.3f}]")
    print()

    # Set input in program
    print("Setting input vector (row 1)...")
    set_input_vector(img, row=1, data=h_in)
    print()

    # Execute program
    print("Executing pixel program...")
    print("  Instructions:")
    print("    1. MATMUL: h = h_in @ W_hidden")
    print("    2. ADD: h += b_hidden")
    print("    3. RELU: h = relu(h)")
    print("    4. MATMUL: logits = h @ W_out")
    print("    5. ADD: logits += b_out")
    print("    6. HALT")
    print()

    result_img = run_program(img.copy())
    print("  ✅ Execution complete")
    print()

    # Extract output
    print("Reading output (logits from row 7)...")
    logits = extract_output_vector(result_img, row=7, length=1024)
    print(f"  Shape: {logits.shape}")
    print(f"  Range: [{logits.min()}, {logits.max()}]")
    print()

    # Show top predictions
    print("="*60)
    print("RESULT")
    print("="*60)

    # Find top-5 predictions
    top_k = 5
    top_indices = np.argsort(logits)[-top_k:][::-1]

    print(f"Top {top_k} predictions (tokens):")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. Token {idx:4d}: score={logits[idx]:3d}")
    print()

    print("✅ Neural network forward pass executed as pixels")
    print("   The entire computation happened inside the image.")
    print()
    print("="*60)


if __name__ == "__main__":
    main()
