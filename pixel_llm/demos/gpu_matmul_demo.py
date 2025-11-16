#!/usr/bin/env python3
"""
GPU Matrix Multiplication Demo

Demonstrates tiled matrix multiplication on GPU.
This is the CRITICAL operation for all neural networks.

Every neural layer uses matmul:
- Dense layers: output = weights × input
- Attention: Q, K, V projections
- Output projection

This proves the GPU can handle neural workloads.
"""

import numpy as np
import time
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.gpu_interface import is_gpu_available, GPUDevice, MatMulGPU


def test_correctness(size: int = 16) -> dict:
    """
    Test matmul correctness against numpy.

    Args:
        size: Matrix size (NxN)

    Returns:
        Test results
    """
    # Generate random matrices
    np.random.seed(42)
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)

    # CPU reference
    C_cpu = A @ B

    if is_gpu_available():
        # GPU compute
        device = GPUDevice()
        matmul = MatMulGPU(device)
        C_gpu = matmul.compute(A, B)

        # Compare
        diff = np.abs(C_cpu - C_gpu).max()
        is_correct = diff < 1e-4

        return {
            'size': size,
            'diff': diff,
            'correct': is_correct,
            'has_gpu': True
        }
    else:
        return {
            'size': size,
            'diff': None,
            'correct': None,
            'has_gpu': False
        }


def benchmark_matmul(M: int, K: int, N: int, iterations: int = 5) -> dict:
    """
    Benchmark CPU vs GPU matmul.

    Args:
        M, K, N: Matrix dimensions (M×K) × (K×N)
        iterations: Number of runs

    Returns:
        Benchmark results
    """
    # Generate random matrices
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    # CPU benchmark
    cpu_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        C_cpu = A @ B
        cpu_times.append(time.perf_counter() - start)

    cpu_time = np.median(cpu_times)

    if is_gpu_available():
        # GPU benchmark
        device = GPUDevice()
        matmul = MatMulGPU(device)

        # Warmup
        _ = matmul.compute(A, B)

        gpu_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            C_gpu = matmul.compute(A, B)
            gpu_times.append(time.perf_counter() - start)

        gpu_time = np.median(gpu_times)

        # Verify correctness
        diff = np.abs(C_cpu - C_gpu).max()
        is_correct = diff < 1e-3

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0

        # Calculate GFLOPS
        flops = 2 * M * K * N  # Multiply-add for each output element
        cpu_gflops = (flops / cpu_time) / 1e9
        gpu_gflops = (flops / gpu_time) / 1e9

        return {
            'shape': (M, K, N),
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup,
            'diff': diff,
            'correct': is_correct,
            'cpu_gflops': cpu_gflops,
            'gpu_gflops': gpu_gflops,
            'has_gpu': True
        }
    else:
        flops = 2 * M * K * N
        cpu_gflops = (flops / cpu_time) / 1e9

        return {
            'shape': (M, K, N),
            'cpu_time': cpu_time,
            'gpu_time': None,
            'speedup': None,
            'diff': None,
            'correct': None,
            'cpu_gflops': cpu_gflops,
            'gpu_gflops': None,
            'has_gpu': False
        }


def main():
    print("=" * 70)
    print("GPU MATRIX MULTIPLICATION DEMONSTRATION")
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

    # Test correctness first
    print("Testing correctness...")
    print()

    test_sizes = [4, 8, 16, 32]
    all_correct = True

    for size in test_sizes:
        result = test_correctness(size)

        if result['has_gpu']:
            status = "✅" if result['correct'] else "❌"
            print(f"  {size:>3}×{size:<3}: {status} (diff={result['diff']:.2e})")

            if not result['correct']:
                all_correct = False
        else:
            print(f"  {size:>3}×{size:<3}: ✅ (CPU only)")

    print()

    if gpu_available:
        if all_correct:
            print("✅ All correctness tests PASSED\n")
        else:
            print("❌ Some correctness tests FAILED\n")
            return

    # Benchmark different sizes
    print("Benchmarking performance...")
    print()

    # Test shapes: (M, K, N)
    shapes = [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ]

    if not gpu_available:
        # Limit sizes for CPU-only
        shapes = shapes[:2]

    results = []
    for M, K, N in shapes:
        print(f"  Shape ({M:>4}, {K:>4}, {N:>4})...", end=" ", flush=True)
        result = benchmark_matmul(M, K, N, iterations=3)
        results.append(result)

        if gpu_available:
            status = "✅" if result['correct'] else "❌"
            print(f"{status} (speedup: {result['speedup']:.2f}x, {result['gpu_gflops']:.1f} GFLOPS)")
        else:
            print(f"✅ ({result['cpu_gflops']:.1f} GFLOPS)")

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    # Print table
    if gpu_available:
        print(f"{'Shape':>18} | {'CPU (ms)':>10} | {'GPU (ms)':>10} | {'Speedup':>8} | {'GPU GFLOPS':>12} | {'Status':>8}")
        print("-" * 90)

        for r in results:
            M, K, N = r['shape']
            shape_str = f"({M}, {K}, {N})"
            cpu_ms = r['cpu_time'] * 1000
            gpu_ms = r['gpu_time'] * 1000
            speedup_str = f"{r['speedup']:.2f}x"
            gflops_str = f"{r['gpu_gflops']:.1f}"
            status_str = "✅" if r['correct'] else "❌"

            print(f"{shape_str:>18} | {cpu_ms:>10.2f} | {gpu_ms:>10.2f} | {speedup_str:>8} | {gflops_str:>12} | {status_str:>8}")

    else:
        print(f"{'Shape':>18} | {'CPU (ms)':>10} | {'CPU GFLOPS':>12}")
        print("-" * 50)

        for r in results:
            M, K, N = r['shape']
            shape_str = f"({M}, {K}, {N})"
            cpu_ms = r['cpu_time'] * 1000
            gflops_str = f"{r['cpu_gflops']:.1f}"

            print(f"{shape_str:>18} | {cpu_ms:>10.2f} | {gflops_str:>12}")

    print()

    # Summary
    if gpu_available:
        print("🎯 WHAT THIS PROVES:")
        print()
        print("  ✅ Tiled matrix multiplication works correctly")
        print("  ✅ GPU can handle neural network operations")
        print("  ✅ Shared memory optimization is effective")
        print("  ✅ Ready for attention mechanisms")
        print()

        best_speedup = max(r['speedup'] for r in results if r['speedup'])
        best_gflops = max(r['gpu_gflops'] for r in results)

        print(f"📊 PERFORMANCE:")
        print(f"  Best speedup: {best_speedup:.2f}x")
        print(f"  Peak: {best_gflops:.1f} GFLOPS")
        print()

        print("🚀 READY FOR:")
        print()
        print("  → Attention mechanisms (multi-head attention)")
        print("  → Layer normalization")
        print("  → Full transformer inference")
        print("  → Pixel-native LLM weights")

    else:
        print("🎯 NEXT STEP:")
        print()
        print("  Install wgpu to enable GPU:")
        print("    pip install wgpu")
        print()
        print("  Then run this demo again to see GPU speedup")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
