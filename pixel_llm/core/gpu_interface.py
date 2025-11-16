#!/usr/bin/env python3
"""
GPU Interface for Pixel-LLM

Provides WebGPU access for running compute shaders on pixel data.
Uses wgpu-py for cross-platform GPU compute.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings

# Try to import wgpu - graceful degradation if not available
try:
    import wgpu
    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    warnings.warn(
        "wgpu-py not installed. GPU operations will not be available.\n"
        "Install with: pip install wgpu"
    )


class GPUDevice:
    """
    WebGPU device wrapper for compute operations.

    This is the bridge between pixel data and GPU shaders.
    """

    def __init__(self):
        """Initialize GPU device"""
        if not WGPU_AVAILABLE:
            raise RuntimeError("wgpu-py not installed. Cannot use GPU operations.")

        # Request GPU adapter and device
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")

        if self.adapter is None:
            raise RuntimeError("No GPU adapter found. WebGPU may not be supported on this system.")

        self.device = self.adapter.request_device_sync()

        print(f"GPU Device: {self.adapter.request_adapter_info()['description']}")

    def create_buffer(self, data: np.ndarray, usage: str = "storage") -> "wgpu.GPUBuffer":
        """
        Create GPU buffer from numpy array.

        Args:
            data: Numpy array to upload
            usage: Buffer usage ("storage", "uniform", "storage_read_write")

        Returns:
            GPU buffer
        """
        # Ensure contiguous array
        data = np.ascontiguousarray(data)

        # Determine buffer usage flags
        usage_flags = wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC

        if usage == "storage":
            usage_flags |= wgpu.BufferUsage.STORAGE
        elif usage == "storage_read_write":
            usage_flags |= wgpu.BufferUsage.STORAGE
        elif usage == "uniform":
            usage_flags |= wgpu.BufferUsage.UNIFORM

        # Create buffer
        buffer = self.device.create_buffer(
            size=data.nbytes,
            usage=usage_flags
        )

        # Upload data
        self.device.queue.write_buffer(buffer, 0, data)

        return buffer

    def create_shader_module(self, shader_code: str) -> "wgpu.GPUShaderModule":
        """
        Create shader module from WGSL code.

        Args:
            shader_code: WGSL shader source code

        Returns:
            Shader module
        """
        return self.device.create_shader_module(code=shader_code)

    def read_buffer(self, buffer: "wgpu.GPUBuffer", size: int, dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Read data back from GPU buffer.

        Args:
            buffer: GPU buffer to read
            size: Number of elements
            dtype: Data type

        Returns:
            Numpy array with buffer contents
        """
        # Create staging buffer for readback
        staging_buffer = self.device.create_buffer(
            size=buffer.size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
        )

        # Copy GPU buffer to staging
        command_encoder = self.device.create_command_encoder()
        command_encoder.copy_buffer_to_buffer(buffer, 0, staging_buffer, 0, buffer.size)
        self.device.queue.submit([command_encoder.finish()])

        # Map and read
        staging_buffer.map_sync(mode=wgpu.MapMode.READ)
        data = np.frombuffer(staging_buffer.read_mapped(), dtype=dtype).copy()
        staging_buffer.unmap()

        return data[:size]

    def __del__(self):
        """Cleanup GPU resources"""
        if hasattr(self, 'device'):
            self.device.destroy()


class DotProductGPU:
    """
    GPU-accelerated dot product using WebGPU compute shaders.

    This is the simplest possible GPU operation and proves that:
    1. We can upload data to GPU
    2. We can run compute shaders
    3. We can read results back

    Foundation for all neural operations.
    """

    def __init__(self, device: Optional[GPUDevice] = None):
        """
        Initialize dot product GPU operator.

        Args:
            device: GPU device (creates new one if None)
        """
        self.device = device or GPUDevice()

        # Load shader
        shader_path = Path(__file__).parent.parent / "gpu_kernels" / "dot_product.wgsl"
        with open(shader_path, 'r') as f:
            shader_code = f.read()

        self.shader_module = self.device.create_shader_module(shader_code)

        # Create compute pipeline for first pass
        self.pipeline = self.device.device.create_compute_pipeline(
            layout=None,
            compute={
                "module": self.shader_module,
                "entry_point": "dot_product_kernel"
            }
        )

        # Create pipeline for reduction pass
        self.reduce_pipeline = self.device.device.create_compute_pipeline(
            layout=None,
            compute={
                "module": self.shader_module,
                "entry_point": "reduce_partial_sums"
            }
        )

    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute dot product on GPU.

        Args:
            a: First vector (1D numpy array)
            b: Second vector (1D numpy array)

        Returns:
            Dot product (scalar)
        """
        assert a.shape == b.shape, "Vectors must have same shape"
        assert a.ndim == 1, "Only 1D vectors supported"

        n = len(a)

        # Convert to float32
        a = a.astype(np.float32)
        b = b.astype(np.float32)

        # Calculate workgroup dimensions
        workgroup_size = 256
        num_workgroups = (n + workgroup_size - 1) // workgroup_size

        # Create GPU buffers
        buffer_a = self.device.create_buffer(a, usage="storage")
        buffer_b = self.device.create_buffer(b, usage="storage")

        # Buffer for partial sums (one per workgroup)
        partial_sums = np.zeros(num_workgroups, dtype=np.float32)
        buffer_partial = self.device.create_buffer(partial_sums, usage="storage_read_write")

        # Metadata buffer (vector length, num workgroups)
        metadata = np.array([n, num_workgroups], dtype=np.uint32)
        buffer_metadata = self.device.create_buffer(metadata, usage="storage")

        # Create bind group
        bind_group = self.device.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": buffer_a, "offset": 0, "size": buffer_a.size}},
                {"binding": 1, "resource": {"buffer": buffer_b, "offset": 0, "size": buffer_b.size}},
                {"binding": 2, "resource": {"buffer": buffer_partial, "offset": 0, "size": buffer_partial.size}},
                {"binding": 3, "resource": {"buffer": buffer_metadata, "offset": 0, "size": buffer_metadata.size}},
            ]
        )

        # Execute first pass (compute partial sums)
        command_encoder = self.device.device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(num_workgroups, 1, 1)
        compute_pass.end()
        self.device.device.queue.submit([command_encoder.finish()])

        # If we have multiple workgroups, need a second reduction pass
        if num_workgroups > 1:
            bind_group_reduce = self.device.device.create_bind_group(
                layout=self.reduce_pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": buffer_a, "offset": 0, "size": buffer_a.size}},
                    {"binding": 1, "resource": {"buffer": buffer_b, "offset": 0, "size": buffer_b.size}},
                    {"binding": 2, "resource": {"buffer": buffer_partial, "offset": 0, "size": buffer_partial.size}},
                    {"binding": 3, "resource": {"buffer": buffer_metadata, "offset": 0, "size": buffer_metadata.size}},
                ]
            )

            command_encoder = self.device.device.create_command_encoder()
            compute_pass = command_encoder.begin_compute_pass()
            compute_pass.set_pipeline(self.reduce_pipeline)
            compute_pass.set_bind_group(0, bind_group_reduce)
            compute_pass.dispatch_workgroups(1, 1, 1)
            compute_pass.end()
            self.device.device.queue.submit([command_encoder.finish()])

        # Read result
        result = self.device.read_buffer(buffer_partial, 1, dtype=np.float32)

        return float(result[0])


def is_gpu_available() -> bool:
    """Check if GPU compute is available"""
    if not WGPU_AVAILABLE:
        return False

    try:
        adapter = wgpu.gpu.request_adapter_sync()
        return adapter is not None
    except Exception:
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("GPU Interface Test")
    print("=" * 70)
    print()

    if not is_gpu_available():
        print("❌ GPU not available")
        print("\nInstall wgpu:")
        print("  pip install wgpu")
        exit(1)

    print("✅ GPU available")
    print()

    # Create device
    device = GPUDevice()
    print()

    # Test dot product
    print("Testing dot product...")
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

    # CPU reference
    cpu_result = np.dot(a, b)
    print(f"  CPU result: {cpu_result}")

    # GPU compute
    gpu_dot = DotProductGPU(device)
    gpu_result = gpu_dot.compute(a, b)
    print(f"  GPU result: {gpu_result}")

    # Compare
    diff = abs(cpu_result - gpu_result)
    print(f"  Difference: {diff}")

    if diff < 1e-5:
        print("\n✅ GPU dot product PASSED")
    else:
        print(f"\n❌ GPU dot product FAILED (diff={diff})")

    print()
    print("=" * 70)
