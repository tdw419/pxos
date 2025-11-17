"""
pxvm.gpu - GPU-accelerated pixel VM execution

Executes .pxi programs on GPU using WGSL compute shaders.
Produces identical results to CPU interpreter (pxvm.core.interpreter).

Requires: wgpu-py
Install: pip install wgpu

Architecture:
    1. Upload numpy image array to GPU storage buffer
    2. Dispatch interpreter.wgsl compute shader (1 workgroup)
    3. Read back mutated buffer
    4. Return as numpy array

The Pixel Protocol guarantees that CPU and GPU produce identical results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# Check for wgpu availability
try:
    import wgpu
    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    wgpu = None


class GPUNotAvailableError(Exception):
    """Raised when GPU execution is requested but wgpu is not installed."""

    pass


def _check_wgpu_available() -> None:
    """Check if wgpu is available, raise helpful error if not."""
    if not WGPU_AVAILABLE:
        raise GPUNotAvailableError(
            "GPU execution requires wgpu-py.\n"
            "Install with: pip install wgpu\n"
            "\n"
            "Note: wgpu requires a compatible GPU and drivers.\n"
            "See: https://github.com/pygfx/wgpu-py"
        )


def run_program_gpu(
    img: np.ndarray,
    device: Optional[object] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Execute a pxVM program on GPU using WGSL interpreter.

    Args:
        img: RGBA uint8 image array, shape (H, W, 4)
        device: Optional wgpu device (creates one if None)
        verbose: Print execution details

    Returns:
        Mutated image array (same shape, dtype)

    Raises:
        GPUNotAvailableError: If wgpu is not installed
        AssertionError: If image format is invalid
    """
    _check_wgpu_available()

    assert img.ndim == 3 and img.shape[2] == 4, "Expected RGBA image"
    assert img.dtype == np.uint8, "Expected uint8 dtype"

    height, width, channels = img.shape

    if verbose:
        print(f"[GPU] Executing pxVM program on {width}×{height} image")

    # Create device if not provided
    if device is None:
        adapter = wgpu.gpu.request_adapter_sync(
            power_preference="high-performance"
        )
        device = adapter.request_device_sync()

        if verbose:
            print(f"[GPU] Created device: {adapter.summary}")

    # Load WGSL shader
    shader_path = Path(__file__).parent / "interpreter.wgsl"
    shader_source = shader_path.read_text()

    # Create shader module
    shader_module = device.create_shader_module(code=shader_source)

    if verbose:
        print(f"[GPU] Loaded shader: {shader_path.name}")

    # Convert RGBA uint8 to vec4<u32> for shader
    # Shape (H, W, 4) uint8 → Shape (H*W, 4) uint32
    pixels_u32 = img.astype(np.uint32).reshape(-1, 4)

    # Create storage buffer for pixels (read_write)
    pixel_buffer_size = pixels_u32.nbytes
    pixel_buffer = device.create_buffer_with_data(
        data=pixels_u32,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )

    if verbose:
        print(f"[GPU] Created pixel buffer: {pixel_buffer_size} bytes")

    # Create uniform buffer for layout
    layout_data = np.array([width, height, 0], dtype=np.uint32)  # instruction_row=0
    layout_buffer = device.create_buffer_with_data(
        data=layout_data,
        usage=wgpu.BufferUsage.UNIFORM,
    )

    # Create bind group layout
    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                },
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.uniform,
                },
            },
        ],
    )

    # Create bind group
    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": pixel_buffer,
                    "offset": 0,
                    "size": pixel_buffer_size,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": layout_buffer,
                    "offset": 0,
                    "size": layout_data.nbytes,
                },
            },
        ],
    )

    # Create pipeline layout
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    # Create compute pipeline
    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader_module, "entry_point": "run_program"},
    )

    if verbose:
        print(f"[GPU] Created compute pipeline")

    # Create command encoder
    command_encoder = device.create_command_encoder()

    # Dispatch compute shader (1 workgroup = 1 interpreter instance)
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(1, 1, 1)  # Single workgroup
    compute_pass.end()

    # Submit commands
    device.queue.submit([command_encoder.finish()])

    if verbose:
        print(f"[GPU] Dispatched compute shader (1 workgroup)")

    # Read back results
    # Create staging buffer for readback
    staging_buffer = device.create_buffer(
        size=pixel_buffer_size,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )

    # Copy GPU buffer to staging
    copy_encoder = device.create_command_encoder()
    copy_encoder.copy_buffer_to_buffer(
        pixel_buffer, 0, staging_buffer, 0, pixel_buffer_size
    )
    device.queue.submit([copy_encoder.finish()])

    # Map and read
    staging_buffer.map(wgpu.MapMode.READ)
    result_u32 = np.frombuffer(staging_buffer.read_mapped(), dtype=np.uint32).copy()
    staging_buffer.unmap()

    if verbose:
        print(f"[GPU] Read back {len(result_u32)} uint32 values")

    # Reshape and convert back to uint8 RGBA
    result_u32 = result_u32.reshape(height, width, 4)
    result_u8 = result_u32.astype(np.uint8)

    if verbose:
        print(f"[GPU] Execution complete")

    return result_u8


__all__ = [
    "run_program_gpu",
    "GPUNotAvailableError",
    "WGPU_AVAILABLE",
]
