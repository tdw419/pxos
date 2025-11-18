#!/usr/bin/env python3
"""
Test harness for pixel-encoded serial driver

Uses WebGPU to execute serial_driver.wgsl shader on serial_driver.pxi
"""

import sys
import time
import numpy as np
from PIL import Image

try:
    import wgpu
    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    print("WARNING: wgpu-py not installed. Install with: pip install wgpu")


def load_pxi_driver(filename):
    """
    Load pixel-encoded driver from PXI file

    Args:
        filename: Path to PXI file

    Returns:
        numpy array (height, width, 4) with RGBA values
    """
    img = Image.open(filename)
    return np.array(img, dtype=np.uint8)


def create_test_driver_inline(message="Hello from GPU!"):
    """
    Create test driver inline (for testing without file I/O)

    Args:
        message: Message to encode

    Returns:
        numpy array (1, len(message), 4)
    """
    pixels = []
    for char in message:
        pixels.append([
            ord(char),  # R: character
            0x01,       # G: baud divisor (115200)
            0x03,       # B: line control (8N1)
            0x01        # A: write operation
        ])

    # Pad to 256 pixels
    while len(pixels) < 256:
        pixels.append([0, 0, 0, 0])  # NOP

    return np.array([pixels], dtype=np.uint8)  # Shape: (1, 256, 4)


def execute_serial_driver_cpu(driver_data):
    """
    CPU reference implementation (for comparison)

    Args:
        driver_data: numpy array (height, width, 4)

    Returns:
        (output_string, execution_time_ns)
    """
    start = time.perf_counter_ns()

    output = []
    height, width, _ = driver_data.shape

    for y in range(height):
        for x in range(width):
            r, g, b, a = driver_data[y, x]

            # Skip NOP (A=0)
            if a == 0:
                continue

            # Execute write (A=1)
            if a == 1:
                output.append(chr(r))

    end = time.perf_counter_ns()
    execution_time = end - start

    return ''.join(output), execution_time


def execute_serial_driver_gpu(driver_data, shader_code):
    """
    GPU implementation using WebGPU

    Args:
        driver_data: numpy array (height, width, 4)
        shader_code: WGSL shader source code

    Returns:
        (output_string, execution_time_ns, stats)
    """
    if not WGPU_AVAILABLE:
        raise RuntimeError("WebGPU not available. Install wgpu-py.")

    # Get adapter and device
    adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
    device = adapter.request_device()

    height, width, channels = driver_data.shape

    # Create texture for pixel-encoded driver
    # Convert RGBA u8 to f32 normalized (0.0-1.0)
    driver_normalized = driver_data.astype(np.float32) / 255.0

    texture = device.create_texture(
        size=(width, height, 1),
        format=wgpu.TextureFormat.rgba32float,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )

    # Upload driver data to texture
    device.queue.write_texture(
        {
            "texture": texture,
            "origin": (0, 0, 0),
        },
        driver_normalized.tobytes(),
        {
            "bytes_per_row": width * 4 * 4,  # 4 channels * 4 bytes (f32)
            "rows_per_image": height,
        },
        (width, height, 1),
    )

    # Create output buffer (UART simulation)
    output_buffer = device.create_buffer(
        size=width * height * 4,  # u32 per pixel
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    # Create statistics buffer
    stats_buffer = device.create_buffer(
        size=16,  # 4 atomic u32 counters
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    # Create shader module
    shader_module = device.create_shader_module(code=shader_code)

    # Create bind group layout
    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.float,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
        ]
    )

    # Create pipeline layout
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    # Create compute pipeline
    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader_module, "entry_point": "serial_driver_main"},
    )

    # Create bind group
    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {"binding": 0, "resource": texture.create_view()},
            {"binding": 1, "resource": {"buffer": output_buffer}},
            {"binding": 2, "resource": {"buffer": stats_buffer}},
        ],
    )

    # Create command encoder
    command_encoder = device.create_command_encoder()

    # Start timer
    start = time.perf_counter_ns()

    # Dispatch compute shader
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(1, height)  # width=256 handled by workgroup_size
    compute_pass.end()

    # Submit commands
    device.queue.submit([command_encoder.finish()])

    # Wait for completion
    device.queue.on_submitted_work_done(lambda: None)

    end = time.perf_counter_ns()
    execution_time = end - start

    # Read back results
    output_data = device.queue.read_buffer(output_buffer).cast('I')  # u32 array
    stats_data = device.queue.read_buffer(stats_buffer).cast('I')  # u32 array

    # Convert output to string
    output = []
    for i in range(len(output_data)):
        if output_data[i] != 0:
            output.append(chr(output_data[i]))

    stats = {
        "ops_executed": stats_data[0],
        "chars_written": stats_data[1],
        "errors": stats_data[2],
    }

    return ''.join(output), execution_time, stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test pixel-encoded serial driver')
    parser.add_argument(
        '-i', '--input',
        default=None,
        help='Input PXI file (default: create inline test driver)'
    )
    parser.add_argument(
        '-m', '--message',
        default='Hello from GPU!',
        help='Message for inline driver (default: "Hello from GPU!")'
    )
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Run CPU reference only (no GPU)'
    )

    args = parser.parse_args()

    # Load driver data
    if args.input:
        print(f"Loading driver from {args.input}...")
        driver_data = load_pxi_driver(args.input)
    else:
        print(f"Creating inline driver for: {repr(args.message)}")
        driver_data = create_test_driver_inline(args.message)

    print(f"Driver size: {driver_data.shape[1]}x{driver_data.shape[0]} pixels")
    print()

    # Run CPU reference
    print("=" * 60)
    print("CPU Reference Implementation")
    print("=" * 60)

    cpu_output, cpu_time = execute_serial_driver_cpu(driver_data)

    print(f"Output: {repr(cpu_output)}")
    print(f"Execution time: {cpu_time:,} ns ({cpu_time / 1000:.2f} µs)")
    print()

    if args.cpu_only:
        return

    # Run GPU implementation
    if not WGPU_AVAILABLE:
        print("WebGPU not available. Skipping GPU test.")
        print("Install with: pip install wgpu")
        return

    print("=" * 60)
    print("GPU WebGPU Implementation")
    print("=" * 60)

    # Load shader
    shader_path = "serial_driver.wgsl"
    try:
        with open(shader_path, 'r') as f:
            shader_code = f.read()
    except FileNotFoundError:
        print(f"ERROR: Shader not found at {shader_path}")
        print("Run this script from the serial_driver_poc/ directory")
        sys.exit(1)

    try:
        gpu_output, gpu_time, stats = execute_serial_driver_gpu(driver_data, shader_code)

        print(f"Output: {repr(gpu_output)}")
        print(f"Execution time: {gpu_time:,} ns ({gpu_time / 1000:.2f} µs)")
        print(f"Statistics:")
        print(f"  Operations executed: {stats['ops_executed']}")
        print(f"  Characters written: {stats['chars_written']}")
        print(f"  Errors: {stats['errors']}")
        print()

        # Compare
        print("=" * 60)
        print("Performance Comparison")
        print("=" * 60)

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0

        print(f"CPU time:  {cpu_time:,} ns ({cpu_time / 1000:.2f} µs)")
        print(f"GPU time:  {gpu_time:,} ns ({gpu_time / 1000:.2f} µs)")
        print(f"Speedup:   {speedup:.2f}x")
        print()

        if cpu_output == gpu_output:
            print("✅ Output matches! GPU implementation is correct.")
        else:
            print("❌ Output mismatch!")
            print(f"  CPU: {repr(cpu_output)}")
            print(f"  GPU: {repr(gpu_output)}")

    except Exception as e:
        print(f"ERROR: GPU execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
