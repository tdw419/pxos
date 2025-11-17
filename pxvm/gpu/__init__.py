"""
pxvm.gpu - GPU execution backend for pxVM

Provides GPU-accelerated execution of pixel programs using WebGPU (wgpu-py).

Key Features:
- MatMul acceleration via WGSL compute shaders
- Executor-agnostic design: same .pxi runs on CPU or GPU
- Transparent fallback to CPU if GPU unavailable

Usage:
    from pxvm.gpu import GPUExecutor

    executor = GPUExecutor()
    result = executor.matmul(A, B)
"""

__all__ = ["executor"]
