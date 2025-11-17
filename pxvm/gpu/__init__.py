import numpy as np
import wgpu
import os
from pathlib import Path

# Fix for wgpu-py logger
os.environ["WGPU_LOG_LEVEL"] = "WARNING"

def run_program_gpu(img: np.ndarray) -> np.ndarray:
    """
    Maps numpy image to GPU buffer and executes WGSL.
    """
    height, width, channels = img.shape

    # Pack RGBA into u32
    img_u32 = np.ascontiguousarray(img).view(np.uint32)

    # 1. Create WebGPU device
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()

    # 2. Upload img as storage buffer
    pixel_buffer = device.create_buffer_with_data(data=img_u32, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    layout_data = np.array([width, height, 0], dtype=np.uint32)
    image_layout_buffer = device.create_buffer_with_data(data=layout_data, usage=wgpu.BufferUsage.UNIFORM)

    # Shader
    shader_path = Path(__file__).parent / "interpreter.wgsl"
    shader_code = shader_path.read_text()
    shader = device.create_shader_module(code=shader_code)

    # Bindings and pipeline
    bind_group_layout = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "uniform"}},
    ])
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[
        {"binding": 0, "resource": {"buffer": pixel_buffer, "offset": 0, "size": pixel_buffer.size}},
        {"binding": 1, "resource": {"buffer": image_layout_buffer, "offset": 0, "size": image_layout_buffer.size}},
    ])
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    compute_pipeline = device.create_compute_pipeline(layout=pipeline_layout, compute={"module": shader, "entry_point": "run_program"})

    # 3. Dispatch compute shader
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999)
    compute_pass.dispatch_workgroups(1, 1, 1)
    compute_pass.end()
    device.queue.submit([command_encoder.finish()])

    # 4. Read back buffer
    out_buffer = device.create_buffer(size=pixel_buffer.size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_buffer(pixel_buffer, 0, out_buffer, 0, pixel_buffer.size)
    device.queue.submit([command_encoder.finish()])

    result_raw = out_buffer.map(wgpu.MapMode.READ)
    out_buffer.unmap()

    # 5. Return mutated image
    result_u32 = np.frombuffer(result_raw, dtype=np.uint32).reshape((height, width))
    result_rgba = np.ascontiguousarray(result_u32).view(np.uint8).reshape((height, width, 4))

    return result_rgba
