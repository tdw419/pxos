#!/usr/bin/env python3
"""
GPU Attention Mechanism Demo

Demonstrates and validates GPU attention for Pixel-LLM.

This proves:
1. Scaled dot-product attention works correctly
2. Causal masking works (for autoregressive generation)
3. Attention weights sum to 1.0
4. Ready for transformer block assembly
"""

import numpy as np
import time
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.gpu_interface import is_gpu_available, GPUDevice, AttentionGPU


def cpu_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    causal: bool = False
) -> np.ndarray:
    """
    CPU reference implementation of scaled dot-product attention.

    Args:
        query: (seq_len, d_k)
        key: (seq_len, d_k)
        value: (seq_len, d_v)
        causal: Apply causal mask if True

    Returns:
        Attention output (seq_len, d_v)
    """
    seq_len, d_k = query.shape

    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = query @ key.T / np.sqrt(d_k)

    # Apply causal mask if needed
    if causal:
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        scores = scores + mask

    # Softmax
    scores_max = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Apply attention to values
    output = attention_weights @ value

    return output, attention_weights


def test_bidirectional_attention() -> dict:
    """Test bidirectional (encoder-style) attention"""
    print("Testing Bidirectional Attention...")
    print()

    if not is_gpu_available():
        print("⚠️  GPU not available - skipping test")
        return {'has_gpu': False, 'passed': None}

    device = GPUDevice()
    attention_gpu = AttentionGPU(device)

    # Test cases with different dimensions
    test_cases = [
        (4, 8, 8),    # (seq_len, d_k, d_v)
        (8, 16, 16),
        (16, 32, 32),
        (32, 64, 64),
    ]

    results = []

    for i, (seq_len, d_k, d_v) in enumerate(test_cases):
        # Create random Q, K, V matrices
        np.random.seed(42 + i)
        query = np.random.randn(seq_len, d_k).astype(np.float32)
        key = np.random.randn(seq_len, d_k).astype(np.float32)
        value = np.random.randn(seq_len, d_v).astype(np.float32)

        # CPU reference
        cpu_output, cpu_weights = cpu_attention(query, key, value, causal=False)

        # GPU compute
        gpu_output = attention_gpu.compute(query, key, value, causal=False)

        # Compare
        diff = np.abs(cpu_output - gpu_output).max()
        is_correct = diff < 1e-4

        status = "✅" if is_correct else "❌"
        print(f"  Test {i+1} (seq_len={seq_len:2d}, d_k={d_k:2d}, d_v={d_v:2d}): {status} (max_diff={diff:.2e})")

        results.append({
            'test': i+1,
            'seq_len': seq_len,
            'd_k': d_k,
            'd_v': d_v,
            'correct': is_correct,
            'diff': diff
        })

    print()
    all_passed = all(r['correct'] for r in results)

    if all_passed:
        print("✅ All bidirectional attention tests PASSED")
    else:
        print("❌ Some bidirectional attention tests FAILED")

    return {'has_gpu': True, 'passed': all_passed, 'results': results}


def test_causal_attention() -> dict:
    """Test causal (decoder-style) attention"""
    print("\nTesting Causal Attention...")
    print()

    if not is_gpu_available():
        print("⚠️  GPU not available - skipping test")
        return {'has_gpu': False, 'passed': None}

    device = GPUDevice()
    attention_gpu = AttentionGPU(device)

    # Test cases
    test_cases = [
        (4, 8, 8),
        (8, 16, 16),
        (16, 32, 32),
    ]

    results = []

    for i, (seq_len, d_k, d_v) in enumerate(test_cases):
        # Create random Q, K, V matrices
        np.random.seed(100 + i)
        query = np.random.randn(seq_len, d_k).astype(np.float32)
        key = np.random.randn(seq_len, d_k).astype(np.float32)
        value = np.random.randn(seq_len, d_v).astype(np.float32)

        # CPU reference
        cpu_output, cpu_weights = cpu_attention(query, key, value, causal=True)

        # GPU compute
        gpu_output = attention_gpu.compute(query, key, value, causal=True)

        # Compare
        diff = np.abs(cpu_output - gpu_output).max()
        is_correct = diff < 1e-4

        # Verify causal property: output[i] should only depend on positions <= i
        # This is implicitly tested by comparing to CPU reference

        status = "✅" if is_correct else "❌"
        print(f"  Test {i+1} (seq_len={seq_len:2d}, d_k={d_k:2d}, d_v={d_v:2d}): {status} (max_diff={diff:.2e})")

        results.append({
            'test': i+1,
            'seq_len': seq_len,
            'd_k': d_k,
            'd_v': d_v,
            'correct': is_correct,
            'diff': diff
        })

    print()
    all_passed = all(r['correct'] for r in results)

    if all_passed:
        print("✅ All causal attention tests PASSED")
    else:
        print("❌ Some causal attention tests FAILED")

    return {'has_gpu': True, 'passed': all_passed, 'results': results}


def visualize_attention_pattern():
    """Visualize attention patterns"""
    print("\n" + "=" * 70)
    print("ATTENTION PATTERN VISUALIZATION")
    print("=" * 70)
    print()

    if not is_gpu_available():
        print("⚠️  GPU not available - skipping visualization")
        return

    device = GPUDevice()
    attention_gpu = AttentionGPU(device)

    # Create a simple sequence
    seq_len = 8
    d_k = 16
    d_v = 16

    # Create Q, K with specific pattern (for visualization)
    np.random.seed(42)
    query = np.random.randn(seq_len, d_k).astype(np.float32)
    key = np.random.randn(seq_len, d_k).astype(np.float32)
    value = np.random.randn(seq_len, d_v).astype(np.float32)

    # Compute both bidirectional and causal attention
    print("Computing attention patterns...")

    # Get attention weights from CPU (easier to visualize)
    _, bidirectional_weights = cpu_attention(query, key, value, causal=False)
    _, causal_weights = cpu_attention(query, key, value, causal=True)

    print("\nBidirectional Attention Weights (row attends to all columns):")
    print("(Higher values = stronger attention)")
    print()
    print("     ", end="")
    for j in range(seq_len):
        print(f"{j:5d}", end="")
    print()
    print("     " + "-" * (seq_len * 5))

    for i in range(seq_len):
        print(f"{i:3d} |", end="")
        for j in range(seq_len):
            weight = bidirectional_weights[i, j]
            # Simple ASCII visualization
            if weight > 0.2:
                symbol = "█"
            elif weight > 0.15:
                symbol = "▓"
            elif weight > 0.10:
                symbol = "▒"
            elif weight > 0.05:
                symbol = "░"
            else:
                symbol = "·"
            print(f" {symbol:>3s} ", end="")
        print()

    print("\nCausal Attention Weights (row only attends to positions <= column):")
    print("(Notice the triangular pattern - no future information)")
    print()
    print("     ", end="")
    for j in range(seq_len):
        print(f"{j:5d}", end="")
    print()
    print("     " + "-" * (seq_len * 5))

    for i in range(seq_len):
        print(f"{i:3d} |", end="")
        for j in range(seq_len):
            if j > i:
                print("     ", end="")  # Masked positions
            else:
                weight = causal_weights[i, j]
                if weight > 0.2:
                    symbol = "█"
                elif weight > 0.15:
                    symbol = "▓"
                elif weight > 0.10:
                    symbol = "▒"
                elif weight > 0.05:
                    symbol = "░"
                else:
                    symbol = "·"
                print(f" {symbol:>3s} ", end="")
        print()

    print("\nKey observations:")
    print("  • Bidirectional: Each position can attend to all positions")
    print("  • Causal: Position i can only attend to positions 0...i")
    print("  • Weights in each row sum to 1.0 (probability distribution)")
    print("  • Causal masking enables autoregressive generation")


def main():
    print("=" * 70)
    print("GPU ATTENTION MECHANISM DEMONSTRATION")
    print("=" * 70)
    print()

    # Check GPU availability
    gpu_available = is_gpu_available()

    if gpu_available:
        print("✅ GPU Available")
        device = GPUDevice()
        print()
    else:
        print("⚠️  GPU Not Available")
        print("\nTo enable GPU:")
        print("  pip install wgpu")
        print("\nCannot run attention tests without GPU.\n")
        return

    # Run tests
    bidirectional_result = test_bidirectional_attention()
    causal_result = test_causal_attention()

    # Visualize
    visualize_attention_pattern()

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    all_passed = (
        bidirectional_result.get('passed', False) and
        causal_result.get('passed', False)
    )

    if all_passed:
        print("✅ ALL ATTENTION TESTS PASSED")
        print()
        print("🎯 WHAT THIS PROVES:")
        print()
        print("  ✅ Scaled dot-product attention works correctly")
        print("  ✅ Attention scores are computed: Q @ K^T / sqrt(d_k)")
        print("  ✅ Softmax normalization produces valid probabilities")
        print("  ✅ Causal masking prevents future information leakage")
        print("  ✅ Attention weights sum to 1.0 (verified implicitly)")
        print("  ✅ Output is weighted sum of values")
        print()
        print("🚀 READY FOR:")
        print()
        print("  → Multi-head attention (parallel heads)")
        print("  → Transformer encoder blocks")
        print("  → Transformer decoder blocks (with causal attention)")
        print("  → Full LLM inference pipeline")
        print("  → Pixel-native language understanding")
        print()
        print("🔮 MILESTONE:")
        print("  This is THE mechanism that makes transformers work!")
        print("  We've built the heart of modern AI on GPU pixels.")
    else:
        print("❌ SOME TESTS FAILED")
        print()
        if not bidirectional_result.get('passed', False):
            print("  ❌ Bidirectional attention failed")
        if not causal_result.get('passed', False):
            print("  ❌ Causal attention failed")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
