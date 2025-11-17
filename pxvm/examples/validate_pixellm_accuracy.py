#!/usr/bin/env python3
"""
pxvm/examples/validate_pixellm_accuracy.py

Validate pxVM execution accuracy against numpy reference implementation.

This script measures the cost of uint8 quantization and proves that the
pixel program produces correct results.

Metrics:
- Pearson correlation coefficient
- Mean squared error (MSE)
- Maximum absolute error
- Top-5 token overlap

Expected: >0.9 correlation despite quantization
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pxvm.core.interpreter import run_program
from pxvm.utils.layout import read_matrix


def pixellm_forward_numpy(
    h_in: np.ndarray,
    W_hidden: np.ndarray,
    b_hidden: np.ndarray,
    W_out: np.ndarray,
    b_out: np.ndarray,
) -> np.ndarray:
    """
    Reference implementation using numpy float32.

    This is the ground truth for what the pixel program should compute.

    Args:
        h_in: Input embedding [128]
        W_hidden: Hidden layer weights [128, 128]
        b_hidden: Hidden layer bias [128]
        W_out: Output layer weights [128, 1024]
        b_out: Output layer bias [1024]

    Returns:
        logits: Output logits [1024]
    """
    # Hidden layer: h = relu(h_in @ W_hidden + b_hidden)
    h = h_in @ W_hidden + b_hidden
    h = np.maximum(0, h)  # ReLU

    # Output layer: logits = h @ W_out + b_out
    logits = h @ W_out + b_out

    return logits


def pixellm_forward_pixel(
    h_in: np.ndarray,
    program_path: Path,
) -> np.ndarray:
    """
    pxVM implementation using quantized pixel program.

    Args:
        h_in: Input embedding [128]
        program_path: Path to pixellm_forward.pxi

    Returns:
        logits: Output logits [1024] (dequantized to float32)
    """
    # Load program
    img = np.array(Image.open(program_path).convert("RGBA"), dtype=np.uint8)

    # Find h_in row (should be row 1 based on encode_pixellm_forward.py)
    # Read header to confirm
    from pxvm.core.interpreter import _read_shape

    h_in_row = 1
    cols, rows = _read_shape(img, h_in_row)

    if (cols, rows) != (128, 1):
        raise ValueError(f"Expected h_in to be 128×1, got {cols}×{rows} at row {h_in_row}")

    # Quantize h_in to uint8 (same quantization used in program)
    h_in_min = h_in.min()
    h_in_max = h_in.max()

    if h_in_max - h_in_min > 0:
        h_in_q = ((h_in - h_in_min) / (h_in_max - h_in_min) * 255).astype(np.uint8)
    else:
        h_in_q = np.zeros_like(h_in, dtype=np.uint8)

    # Write h_in to row 1
    stride = img.shape[1] - 1  # Column 0 is header

    for i, val in enumerate(h_in_q):
        x = 1 + (i % stride)
        y = h_in_row + (i // stride)
        img[y, x, 0] = val

    # Execute program
    result_img = run_program(img.copy())

    # Extract logits from row 7 (based on encode_pixellm_forward.py)
    # First, find the actual logits row by reading instruction 4 (MATMUL that writes to logits)
    logits_row = int(result_img[0, 3, 3])  # ARG2 of instruction 3 (0-indexed)

    # Read logits matrix
    logits_q = read_matrix(result_img, logits_row)

    # Dequantize logits (they're uint8, map back to approximate float range)
    # Since we don't know the original min/max, we just use the uint8 values directly
    # This is a limitation of the quantization scheme - we lose the scale information
    logits = logits_q.astype(np.float32).flatten()

    return logits


def compute_metrics(
    numpy_logits: np.ndarray,
    pixel_logits: np.ndarray,
) -> dict:
    """
    Compute accuracy metrics comparing numpy vs pixel outputs.

    Args:
        numpy_logits: Reference numpy output [1024]
        pixel_logits: pxVM pixel output [1024]

    Returns:
        Dictionary with metrics
    """
    # Pearson correlation coefficient
    correlation = np.corrcoef(numpy_logits, pixel_logits)[0, 1]

    # Mean squared error
    mse = np.mean((numpy_logits - pixel_logits) ** 2)

    # Maximum absolute error
    max_error = np.max(np.abs(numpy_logits - pixel_logits))

    # Top-5 token overlap
    top5_numpy = set(np.argsort(numpy_logits)[-5:][::-1])
    top5_pixel = set(np.argsort(pixel_logits)[-5:][::-1])
    top5_overlap = len(top5_numpy & top5_pixel)

    # Top-1 accuracy
    top1_match = np.argmax(numpy_logits) == np.argmax(pixel_logits)

    return {
        "correlation": correlation,
        "mse": mse,
        "max_error": max_error,
        "top5_overlap": top5_overlap,
        "top1_match": top1_match,
    }


def main():
    """Main validation routine."""
    print("=" * 70)
    print(" pxVM ACCURACY VALIDATION")
    print("=" * 70)
    print()

    root = Path(__file__).resolve().parents[2]
    weights_path = root / "pixel_llm" / "models" / "pixellm_v0.npz"

    # Try compiled version first (from assembler), then fallback to original
    program_path = root / "pixel_llm" / "programs" / "pixellm_forward_compiled.pxi"
    if not program_path.exists():
        program_path = root / "pixel_llm" / "programs" / "pixellm_forward.pxi"

    # Verify files exist
    if not weights_path.exists():
        print(f"❌ Weights not found: {weights_path}")
        print("   Run: python3 pixel_llm/models/pixellm_v0_train.py")
        return

    if not program_path.exists():
        print(f"❌ Program not found: {program_path}")
        print("   Run: python3 -m pxvm.dev.assembler")
        return

    print(f"Using program: {program_path.name}")

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

    # Generate test input (normalized random vector)
    print("Generating test input...")
    np.random.seed(42)  # Reproducible results
    h_in = np.random.randn(128).astype(np.float32)
    h_in = h_in / np.linalg.norm(h_in)  # L2 normalize

    print(f"  h_in shape: {h_in.shape}")
    print(f"  h_in range: [{h_in.min():.3f}, {h_in.max():.3f}]")
    print(f"  h_in norm: {np.linalg.norm(h_in):.3f}")
    print()

    # Run numpy reference
    print("Running numpy reference implementation...")
    numpy_logits = pixellm_forward_numpy(h_in, W_hidden, b_hidden, W_out, b_out)

    print(f"  numpy logits range: [{numpy_logits.min():.3f}, {numpy_logits.max():.3f}]")
    print(f"  numpy top-1 token: {np.argmax(numpy_logits)}")
    print()

    # Run pxVM pixel program
    print(f"Running pxVM pixel program: {program_path.name}")
    pixel_logits = pixellm_forward_pixel(h_in, program_path)

    print(f"  pixel logits range: [{pixel_logits.min():.3f}, {pixel_logits.max():.3f}]")
    print(f"  pixel top-1 token: {np.argmax(pixel_logits)}")
    print()

    # Compute metrics
    print("Computing accuracy metrics...")
    metrics = compute_metrics(numpy_logits, pixel_logits)

    print()
    print("=" * 70)
    print(" VALIDATION RESULTS")
    print("=" * 70)
    print()
    print(f"Correlation Coefficient: {metrics['correlation']:.6f}")
    print(f"Mean Squared Error:      {metrics['mse']:.6f}")
    print(f"Maximum Absolute Error:  {metrics['max_error']:.6f}")
    print(f"Top-5 Token Overlap:     {metrics['top5_overlap']}/5")
    print(f"Top-1 Token Match:       {'✅ YES' if metrics['top1_match'] else '❌ NO'}")
    print()

    # Interpretation
    print("=" * 70)
    print(" INTERPRETATION")
    print("=" * 70)
    print()

    if metrics["correlation"] > 0.9:
        print("✅ EXCELLENT: Correlation >0.9 despite uint8 quantization")
        print("   The pixel program produces highly accurate results.")
    elif metrics["correlation"] > 0.7:
        print("⚠️  ACCEPTABLE: Correlation >0.7 but <0.9")
        print("   Quantization introduces moderate error.")
    else:
        print("❌ POOR: Correlation <0.7")
        print("   Quantization error is too high, or implementation bug.")

    print()

    if metrics["top5_overlap"] >= 4:
        print(f"✅ GOOD: Top-5 overlap = {metrics['top5_overlap']}/5")
        print("   Pixel program ranks tokens similarly to reference.")
    else:
        print(f"⚠️  WARNING: Top-5 overlap = {metrics['top5_overlap']}/5")
        print("   Ranking differs significantly from reference.")

    print()

    # Show top-5 comparison
    print("=" * 70)
    print(" TOP-5 TOKENS COMPARISON")
    print("=" * 70)
    print()

    top5_numpy = np.argsort(numpy_logits)[-5:][::-1]
    top5_pixel = np.argsort(pixel_logits)[-5:][::-1]

    print("Numpy Reference:")
    for i, token_id in enumerate(top5_numpy, 1):
        score = numpy_logits[token_id]
        print(f"  {i}. Token {token_id:4d}: score={score:7.3f}")

    print()
    print("pxVM Pixel Program:")
    for i, token_id in enumerate(top5_pixel, 1):
        score = pixel_logits[token_id]
        match = "✓" if token_id in top5_numpy else " "
        print(f"  {i}. Token {token_id:4d}: score={score:7.3f} {match}")

    print()
    print("=" * 70)

    # Final verdict
    if metrics["correlation"] > 0.9 and metrics["top5_overlap"] >= 4:
        print()
        print("🎉 VALIDATION PASSED")
        print()
        print("The pxVM pixel program implementation is ACCURATE.")
        print("Production neural networks can execute as PNG files with minimal loss.")
        print()
        print("Next steps:")
        print("  - Tag v0.1.0 release")
        print("  - Optimize GPU parallel MatMul kernel")
        print("  - Implement autoregressive text generation")
        print()
    else:
        print()
        print("⚠️  VALIDATION ISSUES DETECTED")
        print()
        print("Review quantization scheme or check for implementation bugs.")
        print()


if __name__ == "__main__":
    main()
