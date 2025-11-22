
import asyncio
import wgpu
import numpy as np
from PIL import Image

def main():
    """
    Main function to set up and run the Pixel VM on the GPU.
    """
    # 1. Initialize WGPU and select a device
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("Could not get a WGPU adapter.")
    device = adapter.request_device_sync()
    if device is None:
        raise RuntimeError("Could not get a WGPU device.")

    # 2. Load the WGSL shader code
    try:
        with open("src/pixel_vm.wgsl", "r") as f:
            shader_code = f.read()
        shader_module = device.create_shader_module(code=shader_code)
    except FileNotFoundError:
        print("Error: src/pixel_vm.wgsl not found. Make sure you are in the pxos-v2.0 directory.")
        return

    # 3. Load the program from the PNG file
    try:
        program_image = Image.open("programs/simple.png").convert("RGBA")
        program_array = np.array(program_image, dtype=np.uint8)
        program_texture_size = (program_image.width, program_image.height, 1)
    except FileNotFoundError:
        print("Error: programs/simple.png not found.")
        return

    # 4. Prepare data and state buffers
    # Data buffer: [value1, value2, result_placeholder, ...]
    data_buffer_array = np.array([5, 7, 0, 0, 0, 0, 0, 0], dtype=np.uint32)

    # State buffer: [ip, zero_flag, halted, regs...]
    # We have 16 registers (16 * u32).
    state_buffer_array = np.zeros(3 + 16, dtype=np.uint32)

    # 5. Create GPU buffers and texture
    # Create the texture for the program code
    code_texture = device.create_texture(
        size=program_texture_size,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba8uint, # Corresponds to a u32 per pixel
    )
    device.queue.write_texture(
        {"texture": code_texture, "mip_level": 0, "origin": (0, 0, 0)},
        program_array,
        {"bytes_per_row": program_array.shape[1] * 4, "rows_per_image": program_array.shape[0]},
        program_texture_size,
    )

    # Create the buffer for the data
    data_buffer = device.create_buffer_with_data(
        data=data_buffer_array,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    # Create the buffer for the state
    state_buffer = device.create_buffer_with_data(
        data=state_buffer_array,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    # Create a buffer to read the results back from the GPU
    output_data_buffer = device.create_buffer(
        size=data_buffer.size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
    )
    output_state_buffer = device.create_buffer(
        size=state_buffer.size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
    )

    # 6. Set up the pipeline and bindings
    binding_layouts = [
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "texture": {"sample_type": wgpu.TextureSampleType.uint, "view_dimension": wgpu.TextureViewDimension.d2},
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
    bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])

    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader_module, "entry_point": "step_pixel_vm"},
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {"binding": 0, "resource": code_texture.create_view()},
            {"binding": 1, "resource": {"buffer": data_buffer, "offset": 0, "size": data_buffer.size}},
            {"binding": 2, "resource": {"buffer": state_buffer, "offset": 0, "size": state_buffer.size}},
        ],
    )

    # 7. Execute the compute shader in a loop
    # The sample program has 5 instructions, so running 6 steps is enough to ensure completion.
    num_steps = 6
    print(f"Running Pixel VM for {num_steps} steps...")
    for i in range(num_steps):
        command_encoder = device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(compute_pipeline)
        compute_pass.set_bind_group(0, bind_group, [], 0, 999999)
        compute_pass.dispatch_workgroups(1, 1, 1)  # dispatch a single workgroup
        compute_pass.end()
        device.queue.submit([command_encoder.finish()])

    # 8. Read back the results
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_buffer(data_buffer, 0, output_data_buffer, 0, data_buffer.size)
    command_encoder.copy_buffer_to_buffer(state_buffer, 0, output_state_buffer, 0, state_buffer.size)
    device.queue.submit([command_encoder.finish()])

    # Map the output buffers to read the data
    final_data = output_data_buffer.map_sync(wgpu.MapMode.READ)
    final_state = output_state_buffer.map_sync(wgpu.MapMode.READ)

    # 9. Print the final state and data
    result_data = np.frombuffer(final_data, dtype=np.uint32)
    result_state = np.frombuffer(final_state, dtype=np.uint32)

    print("\n--- Execution Complete ---")
    print(f"Final IP: {result_state[0]}")
    print(f"Halted: {'Yes' if result_state[2] == 1 else 'No'}")

    print("\nFinal Data Buffer:")
    print(result_data)

    print("\nFinal Registers:")
    print(result_state[3:])

    expected_result = 5 + 7
    actual_result = result_data[2]
    print(f"\nVerification: data[2] should be {expected_result}. Got: {actual_result}")
    if actual_result == expected_result:
        print("✅ Success!")
    else:
        print("❌ Failure!")

if __name__ == "__main__":
    # Some environments need this to run wgpu correctly
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        print("If you see 'No backend available', you might be missing system dependencies for wgpu.")
        print("On Debian/Ubuntu, try: sudo apt-get install libglfw3-dev libstb-dev")
