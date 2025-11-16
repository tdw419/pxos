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


class MatMulGPU:
    """
    GPU-accelerated matrix multiplication using tiled algorithm.

    Implements C = A × B using shared memory optimization.
    This is the foundation for all neural network operations.
    """

    def __init__(self, device: Optional[GPUDevice] = None):
        """
        Initialize matrix multiplication GPU operator.

        Args:
            device: GPU device (creates new one if None)
        """
        self.device = device or GPUDevice()

        # Load shader
        shader_path = Path(__file__).parent.parent / "gpu_kernels" / "matmul.wgsl"
        with open(shader_path, 'r') as f:
            shader_code = f.read()

        self.shader_module = self.device.create_shader_module(shader_code)

        # Create pipelines for different entry points
        self.pipeline_tiled = self.device.device.create_compute_pipeline(
            layout=None,
            compute={
                "module": self.shader_module,
                "entry_point": "matmul_tiled"
            }
        )

        self.pipeline_square = self.device.device.create_compute_pipeline(
            layout=None,
            compute={
                "module": self.shader_module,
                "entry_point": "matmul_square"
            }
        )

        self.pipeline_matvec = self.device.device.create_compute_pipeline(
            layout=None,
            compute={
                "module": self.shader_module,
                "entry_point": "matvec"
            }
        )

    def compute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute matrix multiplication: C = A × B

        Args:
            a: Matrix A (M × K)
            b: Matrix B (K × N)

        Returns:
            Result matrix C (M × N)
        """
        assert a.ndim == 2 and b.ndim == 2, "Both inputs must be 2D matrices"
        assert a.shape[1] == b.shape[0], f"Incompatible shapes: {a.shape} × {b.shape}"

        M, K = a.shape
        K2, N = b.shape

        # Convert to float32
        a = a.astype(np.float32)
        b = b.astype(np.float32)

        # Create GPU buffers
        buffer_a = self.device.create_buffer(a.flatten(), usage="storage")
        buffer_b = self.device.create_buffer(b.flatten(), usage="storage")

        # Output buffer
        c = np.zeros((M, N), dtype=np.float32)
        buffer_c = self.device.create_buffer(c.flatten(), usage="storage_read_write")

        # Metadata buffer
        metadata = np.array([M, K, N], dtype=np.uint32)
        buffer_metadata = self.device.create_buffer(metadata, usage="storage")

        # Choose pipeline based on matrix shape
        if M == N and M == K and M % 16 == 0:
            # Use optimized square matrix kernel
            pipeline = self.pipeline_square
        else:
            # Use general tiled kernel
            pipeline = self.pipeline_tiled

        # Create bind group
        bind_group = self.device.device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": buffer_a, "offset": 0, "size": buffer_a.size}},
                {"binding": 1, "resource": {"buffer": buffer_b, "offset": 0, "size": buffer_b.size}},
                {"binding": 2, "resource": {"buffer": buffer_c, "offset": 0, "size": buffer_c.size}},
                {"binding": 3, "resource": {"buffer": buffer_metadata, "offset": 0, "size": buffer_metadata.size}},
            ]
        )

        # Calculate workgroup dispatch size
        # Each workgroup computes a 16×16 tile
        workgroups_x = (N + 15) // 16
        workgroups_y = (M + 15) // 16

        # Execute matmul
        command_encoder = self.device.device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1)
        compute_pass.end()
        self.device.device.queue.submit([command_encoder.finish()])

        # Read result
        result = self.device.read_buffer(buffer_c, M * N, dtype=np.float32)

        return result.reshape((M, N))

    def matvec(self, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Compute matrix-vector multiplication: y = A × x

        Args:
            a: Matrix A (M × N)
            x: Vector x (N,)

        Returns:
            Result vector y (M,)
        """
        assert a.ndim == 2, "A must be a 2D matrix"
        assert x.ndim == 1, "x must be a 1D vector"
        assert a.shape[1] == x.shape[0], f"Incompatible shapes: {a.shape} × {x.shape}"

        M, N = a.shape

        # Convert to float32
        a = a.astype(np.float32)
        x = x.astype(np.float32)

        # Create GPU buffers
        buffer_a = self.device.create_buffer(a.flatten(), usage="storage")
        buffer_x = self.device.create_buffer(x, usage="storage")

        # Output buffer
        y = np.zeros(M, dtype=np.float32)
        buffer_y = self.device.create_buffer(y, usage="storage_read_write")

        # Metadata buffer
        metadata = np.array([M, N], dtype=np.uint32)
        buffer_metadata = self.device.create_buffer(metadata, usage="storage")

        # Create bind group
        bind_group = self.device.device.create_bind_group(
            layout=self.pipeline_matvec.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": buffer_a, "offset": 0, "size": buffer_a.size}},
                {"binding": 1, "resource": {"buffer": buffer_x, "offset": 0, "size": buffer_x.size}},
                {"binding": 2, "resource": {"buffer": buffer_y, "offset": 0, "size": buffer_y.size}},
                {"binding": 3, "resource": {"buffer": buffer_metadata, "offset": 0, "size": buffer_metadata.size}},
            ]
        )

        # Calculate workgroups
        workgroups = (M + 255) // 256

        # Execute matvec
        command_encoder = self.device.device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline_matvec)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(workgroups, 1, 1)
        compute_pass.end()
        self.device.device.queue.submit([command_encoder.finish()])

        # Read result
        result = self.device.read_buffer(buffer_y, M, dtype=np.float32)

        return result


class SoftmaxGPU:
    """
    GPU-accelerated softmax activation.

    Converts logits to probabilities: softmax(x)_i = exp(x_i) / sum(exp(x))
    Uses numerically stable algorithm with max subtraction.

    Critical for attention mechanisms and output layers.
    """

    def __init__(self, device: Optional[GPUDevice] = None):
        """
        Initialize softmax GPU operator.

        Args:
            device: GPU device (creates new one if None)
        """
        self.device = device or GPUDevice()

        # Load shader
        shader_path = Path(__file__).parent.parent / "gpu_kernels" / "activations.wgsl"
        with open(shader_path, 'r') as f:
            shader_code = f.read()

        self.shader_module = self.device.create_shader_module(shader_code)

        # Create pipeline for combined softmax (works for vectors up to 256 elements)
        self.pipeline = self.device.device.create_compute_pipeline(
            layout=None,
            compute={
                "module": self.shader_module,
                "entry_point": "softmax_combined"
            }
        )

    def compute(self, x: np.ndarray) -> np.ndarray:
        """
        Compute softmax activation.

        Args:
            x: Input vector (1D numpy array)

        Returns:
            Softmax probabilities (same shape as input)
        """
        assert x.ndim == 1, "Only 1D vectors supported"
        assert len(x) <= 256, "Current implementation limited to 256 elements"

        n = len(x)

        # Convert to float32
        x = x.astype(np.float32)

        # Create GPU buffers
        buffer_input = self.device.create_buffer(x, usage="storage")

        # Output buffer
        output = np.zeros(n, dtype=np.float32)
        buffer_output = self.device.create_buffer(output, usage="storage_read_write")

        # Metadata buffer (vector length)
        metadata = np.array([n], dtype=np.uint32)
        buffer_metadata = self.device.create_buffer(metadata, usage="storage")

        # Create bind group
        bind_group = self.device.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": buffer_input, "offset": 0, "size": buffer_input.size}},
                {"binding": 1, "resource": {"buffer": buffer_output, "offset": 0, "size": buffer_output.size}},
                {"binding": 2, "resource": {"buffer": buffer_metadata, "offset": 0, "size": buffer_metadata.size}},
            ]
        )

        # Execute softmax
        command_encoder = self.device.device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(1, 1, 1)  # Single workgroup for combined version
        compute_pass.end()
        self.device.device.queue.submit([command_encoder.finish()])

        # Read result
        result = self.device.read_buffer(buffer_output, n, dtype=np.float32)

        return result


class GELUGPU:
    """
    GPU-accelerated GELU (Gaussian Error Linear Unit) activation.

    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    Used in GPT-2, GPT-3, and BERT feed-forward layers.
    """

    def __init__(self, device: Optional[GPUDevice] = None):
        """
        Initialize GELU GPU operator.

        Args:
            device: GPU device (creates new one if None)
        """
        self.device = device or GPUDevice()

        # Load shader
        shader_path = Path(__file__).parent.parent / "gpu_kernels" / "activations.wgsl"
        with open(shader_path, 'r') as f:
            shader_code = f.read()

        self.shader_module = self.device.create_shader_module(shader_code)

        # Create pipeline
        self.pipeline = self.device.device.create_compute_pipeline(
            layout=None,
            compute={
                "module": self.shader_module,
                "entry_point": "gelu"
            }
        )

    def compute(self, x: np.ndarray) -> np.ndarray:
        """
        Compute GELU activation.

        Args:
            x: Input array (any shape, flattened internally)

        Returns:
            GELU output (same shape as input)
        """
        original_shape = x.shape
        x_flat = x.flatten().astype(np.float32)
        n = len(x_flat)

        # Create GPU buffers
        buffer_input = self.device.create_buffer(x_flat, usage="storage")

        # Output buffer
        output = np.zeros(n, dtype=np.float32)
        buffer_output = self.device.create_buffer(output, usage="storage_read_write")

        # Metadata buffer (vector length)
        metadata = np.array([n], dtype=np.uint32)
        buffer_metadata = self.device.create_buffer(metadata, usage="storage")

        # Create bind group
        bind_group = self.device.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": buffer_input, "offset": 0, "size": buffer_input.size}},
                {"binding": 1, "resource": {"buffer": buffer_output, "offset": 0, "size": buffer_output.size}},
                {"binding": 2, "resource": {"buffer": buffer_metadata, "offset": 0, "size": buffer_metadata.size}},
            ]
        )

        # Calculate workgroups
        workgroups = (n + 255) // 256

        # Execute GELU
        command_encoder = self.device.device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(workgroups, 1, 1)
        compute_pass.end()
        self.device.device.queue.submit([command_encoder.finish()])

        # Read result
        result = self.device.read_buffer(buffer_output, n, dtype=np.float32)

        return result.reshape(original_shape)


class ReLUGPU:
    """
    GPU-accelerated ReLU (Rectified Linear Unit) activation.

    ReLU(x) = max(0, x)

    Simple but effective baseline activation.
    """

    def __init__(self, device: Optional[GPUDevice] = None):
        """
        Initialize ReLU GPU operator.

        Args:
            device: GPU device (creates new one if None)
        """
        self.device = device or GPUDevice()

        # Load shader
        shader_path = Path(__file__).parent.parent / "gpu_kernels" / "activations.wgsl"
        with open(shader_path, 'r') as f:
            shader_code = f.read()

        self.shader_module = self.device.create_shader_module(shader_code)

        # Create pipeline
        self.pipeline = self.device.device.create_compute_pipeline(
            layout=None,
            compute={
                "module": self.shader_module,
                "entry_point": "relu"
            }
        )

    def compute(self, x: np.ndarray) -> np.ndarray:
        """
        Compute ReLU activation.

        Args:
            x: Input array (any shape, flattened internally)

        Returns:
            ReLU output (same shape as input)
        """
        original_shape = x.shape
        x_flat = x.flatten().astype(np.float32)
        n = len(x_flat)

        # Create GPU buffers
        buffer_input = self.device.create_buffer(x_flat, usage="storage")

        # Output buffer
        output = np.zeros(n, dtype=np.float32)
        buffer_output = self.device.create_buffer(output, usage="storage_read_write")

        # Metadata buffer (vector length)
        metadata = np.array([n], dtype=np.uint32)
        buffer_metadata = self.device.create_buffer(metadata, usage="storage")

        # Create bind group
        bind_group = self.device.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": buffer_input, "offset": 0, "size": buffer_input.size}},
                {"binding": 1, "resource": {"buffer": buffer_output, "offset": 0, "size": buffer_output.size}},
                {"binding": 2, "resource": {"buffer": buffer_metadata, "offset": 0, "size": buffer_metadata.size}},
            ]
        )

        # Calculate workgroups
        workgroups = (n + 255) // 256

        # Execute ReLU
        command_encoder = self.device.device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(workgroups, 1, 1)
        compute_pass.end()
        self.device.device.queue.submit([command_encoder.finish()])

        # Read result
        result = self.device.read_buffer(buffer_output, n, dtype=np.float32)

        return result.reshape(original_shape)


class AttentionGPU:
    """
    GPU-accelerated scaled dot-product attention.

    Implements: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    This is the core mechanism of transformers.
    Supports both bidirectional and causal (autoregressive) attention.
    """

    def __init__(self, device: Optional[GPUDevice] = None):
        """
        Initialize attention GPU operator.

        Args:
            device: GPU device (creates new one if None)
        """
        self.device = device or GPUDevice()

        # Load shader
        shader_path = Path(__file__).parent.parent / "gpu_kernels" / "attention.wgsl"
        with open(shader_path, 'r') as f:
            shader_code = f.read()

        self.shader_module = self.device.create_shader_module(shader_code)

        # Create pipeline for fused attention (efficient for small sequences)
        self.pipeline_fused = self.device.device.create_compute_pipeline(
            layout=None,
            compute={
                "module": self.shader_module,
                "entry_point": "single_head_attention_fused"
            }
        )

    def compute(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        causal: bool = False
    ) -> np.ndarray:
        """
        Compute scaled dot-product attention.

        Args:
            query: Query matrix (seq_len, d_k)
            key: Key matrix (seq_len, d_k)
            value: Value matrix (seq_len, d_v)
            causal: If True, apply causal mask (for autoregressive generation)

        Returns:
            Attention output (seq_len, d_v)
        """
        assert query.ndim == 2, "Query must be 2D (seq_len, d_k)"
        assert key.ndim == 2, "Key must be 2D (seq_len, d_k)"
        assert value.ndim == 2, "Value must be 2D (seq_len, d_v)"

        seq_len_q, d_k = query.shape
        seq_len_k, d_k_key = key.shape
        seq_len_v, d_v = value.shape

        assert seq_len_q == seq_len_k == seq_len_v, "Sequence lengths must match"
        assert d_k == d_k_key, "Query and key dimensions must match"
        assert seq_len_q <= 256, "Current implementation limited to seq_len <= 256"

        seq_len = seq_len_q

        # Convert to float32
        query = query.astype(np.float32)
        key = key.astype(np.float32)
        value = value.astype(np.float32)

        # Create GPU buffers
        buffer_query = self.device.create_buffer(query.flatten(), usage="storage")
        buffer_key = self.device.create_buffer(key.flatten(), usage="storage")
        buffer_value = self.device.create_buffer(value.flatten(), usage="storage")

        # Output buffer
        output = np.zeros((seq_len, d_v), dtype=np.float32)
        buffer_output = self.device.create_buffer(output.flatten(), usage="storage_read_write")

        # Metadata buffer
        use_causal_mask = 1 if causal else 0
        metadata = np.array([seq_len, d_k, d_v, use_causal_mask], dtype=np.uint32)
        buffer_metadata = self.device.create_buffer(metadata, usage="storage")

        # Create bind group
        bind_group = self.device.device.create_bind_group(
            layout=self.pipeline_fused.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": buffer_query, "offset": 0, "size": buffer_query.size}},
                {"binding": 1, "resource": {"buffer": buffer_key, "offset": 0, "size": buffer_key.size}},
                {"binding": 2, "resource": {"buffer": buffer_value, "offset": 0, "size": buffer_value.size}},
                {"binding": 3, "resource": {"buffer": buffer_output, "offset": 0, "size": buffer_output.size}},
                {"binding": 4, "resource": {"buffer": buffer_metadata, "offset": 0, "size": buffer_metadata.size}},
            ]
        )

        # Execute attention (one workgroup per query position)
        command_encoder = self.device.device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline_fused)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(seq_len, 1, 1)  # One workgroup per query
        compute_pass.end()
        self.device.device.queue.submit([command_encoder.finish()])

        # Read result
        result = self.device.read_buffer(buffer_output, seq_len * d_v, dtype=np.float32)

        return result.reshape((seq_len, d_v))


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
