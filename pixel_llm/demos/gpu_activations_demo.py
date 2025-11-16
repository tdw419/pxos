#!/usr/bin/env python3
"""
GPU Activation Functions Demo

Demonstrates and validates GPU activation functions for Pixel-LLM.

This proves:
1. Softmax works correctly (critical for attention)
2. GELU works correctly (used in GPT feed-forward layers)
3. ReLU works correctly (baseline activation)
4. Ready for attention mechanism implementation
"""

import numpy as np
import time
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.gpu_interface import is_gpu_available, GPUDevice, SoftmaxGPU, GELUGPU, ReLUGPU


def cpu_softmax(x: np.ndarray) -> np.ndarray:
    """CPU reference implementation of softmax"""
    # Numerically stable softmax
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)


def cpu_gelu(x: np.ndarray) -> np.ndarray:
    """CPU reference implementation of GELU (tanh approximation)"""
    # Same approximation as used in GPT-2
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    coeff = sqrt_2_over_pi * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(coeff))


def cpu_relu(x: np.ndarray) -> np.ndarray:
    """CPU reference implementation of ReLU"""
    return np.maximum(0, x)


def test_softmax() -> dict:
    """Test softmax correctness"""
    print("Testing Softmax...")
    print()

    # Test cases
    test_cases = [
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([-1.0, -2.0, -3.0, -4.0]),
        np.array([100.0, 200.0, 300.0]),  # Large values (numerical stability test)
        np.random.randn(64).astype(np.float32),  # Random values
    ]

    results = []

    if is_gpu_available():
        device = GPUDevice()
        softmax_gpu = SoftmaxGPU(device)

        for i, x in enumerate(test_cases):
            # CPU reference
            cpu_result = cpu_softmax(x)

            # GPU compute
            gpu_result = softmax_gpu.compute(x)

            # Compare
            diff = np.abs(cpu_result - gpu_result).max()
            is_correct = diff < 1e-5

            # Verify properties
            sum_valid = abs(np.sum(gpu_result) - 1.0) < 1e-5
            all_positive = np.all(gpu_result >= 0)

            status = "✅" if (is_correct and sum_valid and all_positive) else "❌"

            print(f"  Test {i+1}: {status} (max_diff={diff:.2e}, sum={np.sum(gpu_result):.6f})")

            results.append({
                'test': i+1,
                'correct': is_correct,
                'sum_valid': sum_valid,
                'all_positive': all_positive,
                'diff': diff
            })

        print()
        all_passed = all(r['correct'] and r['sum_valid'] and r['all_positive'] for r in results)

        if all_passed:
            print("✅ All softmax tests PASSED")
        else:
            print("❌ Some softmax tests FAILED")

        return {'has_gpu': True, 'passed': all_passed, 'results': results}
    else:
        print("⚠️  GPU not available - skipping softmax tests")
        return {'has_gpu': False, 'passed': None, 'results': []}


def test_gelu() -> dict:
    """Test GELU correctness"""
    print("\nTesting GELU...")
    print()

    # Test cases
    test_cases = [
        np.array([0.0]),
        np.array([1.0]),
        np.array([-1.0]),
        np.array([0.0, 0.5, 1.0, 2.0, -0.5, -1.0, -2.0]),
        np.random.randn(100).astype(np.float32),
    ]

    results = []

    if is_gpu_available():
        device = GPUDevice()
        gelu_gpu = GELUGPU(device)

        for i, x in enumerate(test_cases):
            # CPU reference
            cpu_result = cpu_gelu(x)

            # GPU compute
            gpu_result = gelu_gpu.compute(x)

            # Compare
            diff = np.abs(cpu_result - gpu_result).max()
            is_correct = diff < 1e-5

            status = "✅" if is_correct else "❌"

            print(f"  Test {i+1}: {status} (max_diff={diff:.2e})")

            results.append({
                'test': i+1,
                'correct': is_correct,
                'diff': diff
            })

        print()
        all_passed = all(r['correct'] for r in results)

        if all_passed:
            print("✅ All GELU tests PASSED")
        else:
            print("❌ Some GELU tests FAILED")

        return {'has_gpu': True, 'passed': all_passed, 'results': results}
    else:
        print("⚠️  GPU not available - skipping GELU tests")
        return {'has_gpu': False, 'passed': None, 'results': []}


def test_relu() -> dict:
    """Test ReLU correctness"""
    print("\nTesting ReLU...")
    print()

    # Test cases
    test_cases = [
        np.array([0.0]),
        np.array([1.0]),
        np.array([-1.0]),
        np.array([0.0, 0.5, 1.0, 2.0, -0.5, -1.0, -2.0]),
        np.random.randn(100).astype(np.float32),
    ]

    results = []

    if is_gpu_available():
        device = GPUDevice()
        relu_gpu = ReLUGPU(device)

        for i, x in enumerate(test_cases):
            # CPU reference
            cpu_result = cpu_relu(x)

            # GPU compute
            gpu_result = relu_gpu.compute(x)

            # Compare
            diff = np.abs(cpu_result - gpu_result).max()
            is_correct = diff < 1e-7

            # Verify ReLU property: all outputs >= 0
            all_positive = np.all(gpu_result >= 0)

            status = "✅" if (is_correct and all_positive) else "❌"

            print(f"  Test {i+1}: {status} (max_diff={diff:.2e})")

            results.append({
                'test': i+1,
                'correct': is_correct,
                'all_positive': all_positive,
                'diff': diff
            })

        print()
        all_passed = all(r['correct'] and r['all_positive'] for r in results)

        if all_passed:
            print("✅ All ReLU tests PASSED")
        else:
            print("❌ Some ReLU tests FAILED")

        return {'has_gpu': True, 'passed': all_passed, 'results': results}
    else:
        print("⚠️  GPU not available - skipping ReLU tests")
        return {'has_gpu': False, 'passed': None, 'results': []}


def visualize_activations():
    """Visualize activation function shapes"""
    print("\n" + "=" * 70)
    print("ACTIVATION FUNCTION VISUALIZATION")
    print("=" * 70)
    print()

    x_range = np.linspace(-3, 3, 61)

    if is_gpu_available():
        device = GPUDevice()
        gelu_gpu = GELUGPU(device)
        relu_gpu = ReLUGPU(device)

        # Compute activations
        gelu_y = gelu_gpu.compute(x_range)
        relu_y = relu_gpu.compute(x_range)

        # Simple ASCII visualization
        print("GELU vs ReLU (x from -3 to 3):")
        print()

        # Print a few sample points
        sample_x = [-2.0, -1.0, 0.0, 1.0, 2.0]
        print(f"{'x':>6} | {'GELU':>10} | {'ReLU':>10}")
        print("-" * 32)

        for x_val in sample_x:
            idx = np.argmin(np.abs(x_range - x_val))
            gelu_val = gelu_y[idx]
            relu_val = relu_y[idx]
            print(f"{x_val:>6.1f} | {gelu_val:>10.4f} | {relu_val:>10.4f}")

        print()
        print("Key observations:")
        print("  • GELU is smooth everywhere (differentiable)")
        print("  • ReLU has sharp cutoff at 0")
        print("  • GELU allows small negative values (better gradients)")
        print("  • Both are used in modern LLMs")

    else:
        print("⚠️  GPU not available - skipping visualization")


def main():
    print("=" * 70)
    print("GPU ACTIVATION FUNCTIONS DEMONSTRATION")
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
        print("\nCannot run activation tests without GPU.\n")
        return

    # Run tests
    softmax_result = test_softmax()
    gelu_result = test_gelu()
    relu_result = test_relu()

    # Visualize
    visualize_activations()

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    all_passed = (
        softmax_result.get('passed', False) and
        gelu_result.get('passed', False) and
        relu_result.get('passed', False)
    )

    if all_passed:
        print("✅ ALL ACTIVATION TESTS PASSED")
        print()
        print("🎯 WHAT THIS PROVES:")
        print()
        print("  ✅ Softmax works correctly (numerically stable)")
        print("  ✅ GELU works correctly (matches GPT-2/3 implementation)")
        print("  ✅ ReLU works correctly (baseline verified)")
        print("  ✅ GPU activation functions ready for production")
        print()
        print("🚀 READY FOR:")
        print()
        print("  → Attention mechanism (uses softmax)")
        print("  → Feed-forward layers (uses GELU)")
        print("  → Full transformer block assembly")
        print("  → Pixel-native LLM inference")
    else:
        print("❌ SOME TESTS FAILED")
        print()
        if not softmax_result.get('passed', False):
            print("  ❌ Softmax failed")
        if not gelu_result.get('passed', False):
            print("  ❌ GELU failed")
        if not relu_result.get('passed', False):
            print("  ❌ ReLU failed")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
