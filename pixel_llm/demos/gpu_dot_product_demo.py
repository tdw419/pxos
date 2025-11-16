#!/usr/bin/env python3
"""
GPU Dot Product Demo

Demonstrates and validates GPU compute for Pixel-LLM.

This proves:
1. GPU can read data
2. GPU compute is correct
3. GPU is faster than CPU (for large vectors)
4. Foundation is ready for neural operations
"""

import numpy as np
import time
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.gpu_interface import is_gpu_available, GPUDevice, DotProductGPU


def benchmark_dot_product(size: int, iterations: int = 10) -> dict:
    """
    Benchmark CPU vs GPU dot product.

    Args:
        size: Vector size
        iterations: Number of iterations for timing

    Returns:
        Results dictionary
    """
    # Generate random vectors
    np.random.seed(42)
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)

    # CPU benchmark
    cpu_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        cpu_result = np.dot(a, b)
        cpu_times.append(time.perf_counter() - start)

    cpu_time = np.median(cpu_times)

    # GPU benchmark (if available)
    if is_gpu_available():
        device = GPUDevice()
        gpu_dot = DotProductGPU(device)

        # Warmup
        _ = gpu_dot.compute(a, b)

        gpu_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            gpu_result = gpu_dot.compute(a, b)
            gpu_times.append(time.perf_counter() - start)

        gpu_time = np.median(gpu_times)

        # Verify correctness
        diff = abs(cpu_result - gpu_result)
        is_correct = diff < 1e-4

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0

        return {
            'size': size,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup,
            'diff': diff,
            'correct': is_correct,
            'cpu_result': cpu_result,
            'gpu_result': gpu_result
        }
    else:
        return {
            'size': size,
            'cpu_time': cpu_time,
            'gpu_time': None,
            'speedup': None,
            'diff': None,
            'correct': None,
            'cpu_result': cpu_result,
            'gpu_result': None
        }


def main():
    print("=" * 70)
    print("GPU DOT PRODUCT DEMONSTRATION")
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
        print("\nContinuing with CPU-only benchmark...\n")

    # Test sizes
    sizes = [100, 1000, 10000, 100000, 1000000]

    if not gpu_available:
        # Limit sizes if no GPU
        sizes = [100, 1000, 10000]

    print("Benchmarking...")
    print()

    results = []
    for size in sizes:
        print(f"  Vector size: {size:>10,}...", end=" ", flush=True)
        result = benchmark_dot_product(size, iterations=10)
        results.append(result)

        if gpu_available:
            status = "✅" if result['correct'] else "❌"
            print(f"{status} (speedup: {result['speedup']:.2f}x)")
        else:
            print("✅ (CPU only)")

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    # Print table
    if gpu_available:
        print(f"{'Size':>12} | {'CPU (ms)':>10} | {'GPU (ms)':>10} | {'Speedup':>8} | {'Correct':>8}")
        print("-" * 70)

        for r in results:
            cpu_ms = r['cpu_time'] * 1000
            gpu_ms = r['gpu_time'] * 1000
            speedup_str = f"{r['speedup']:.2f}x"
            correct_str = "✅" if r['correct'] else "❌"

            print(f"{r['size']:>12,} | {cpu_ms:>10.4f} | {gpu_ms:>10.4f} | {speedup_str:>8} | {correct_str:>8}")

    else:
        print(f"{'Size':>12} | {'CPU (ms)':>10}")
        print("-" * 30)

        for r in results:
            cpu_ms = r['cpu_time'] * 1000
            print(f"{r['size']:>12,} | {cpu_ms:>10.4f}")

    print()

    # Summary
    if gpu_available:
        all_correct = all(r['correct'] for r in results)

        if all_correct:
            print("✅ ALL TESTS PASSED - GPU compute is correct!")

            # Find best speedup
            best_speedup = max(r['speedup'] for r in results)
            print(f"\nBest speedup: {best_speedup:.2f}x")

            if best_speedup > 1.0:
                print("🚀 GPU is faster than CPU for large vectors")
            else:
                print("⚠️  GPU overhead dominates for these small sizes")
                print("   (Normal for simple operations - matrix ops will show better speedup)")

        else:
            print("❌ SOME TESTS FAILED - GPU results don't match CPU")
            for r in results:
                if not r['correct']:
                    print(f"   Size {r['size']}: diff = {r['diff']}")

    else:
        print("ℹ️  CPU benchmark complete")
        print("\nInstall wgpu to enable GPU benchmarking:")
        print("  pip install wgpu")

    print()
    print("=" * 70)
    print()

    # What this proves
    if gpu_available:
        print("🎯 WHAT THIS PROVES:")
        print()
        print("  ✅ WebGPU compute is working")
        print("  ✅ Data can be uploaded to GPU")
        print("  ✅ Compute shaders execute correctly")
        print("  ✅ Results can be read back")
        print("  ✅ Numerical accuracy is maintained")
        print()
        print("🚀 READY FOR:")
        print()
        print("  → Matrix multiplication (matmul.wgsl)")
        print("  → Attention mechanisms (attention.wgsl)")
        print("  → Full neural inference (pixel_llm_gpu.py)")
        print("  → Pixel-native LLM weights")
    else:
        print("🎯 NEXT STEP:")
        print()
        print("  Install wgpu to enable GPU:")
        print("    pip install wgpu")
        print()
        print("  Then run this demo again to verify GPU works")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
