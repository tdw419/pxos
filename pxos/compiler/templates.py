WGSL_TEMPLATE = """
// PxOS State (Uniform Buffer)
struct PxOSState {{
    screen_dims: vec2<f32>,
    time: f32,
}};
@group(0) @binding(0) var<uniform> pxos: PxOSState;

// Pixel Output (Storage Buffer)
@group(0) @binding(1) var<storage, read_write> pixel_buffer: array<vec4<f32>>;

// --- User's PxSL code is injected here ---
{helper_functions}

fn pixel_main_internal(coord: vec2<f32>, screen_dims: vec2<f32>, time: f32) -> vec4<f32> {{
    {main_function_body}
}}
// --- End of user code ---

// GPU Entry Point
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let screen_width = u32(pxos.screen_dims.x);
    let pixel_index = global_id.y * screen_width + global_id.x;

    // Stop if we're outside the screen bounds
    if (global_id.x >= screen_width || global_id.y >= u32(pxos.screen_dims.y)) {{
        return;
    }}

    let coord = vec2<f32>(f32(global_id.x), f32(global_id.y));

    // Call the user's main function
    let color = pixel_main_internal(coord, pxos.screen_dims, pxos.time);

    // Write the final color to the buffer
    pixel_buffer[pixel_index] = color;
}}
"""
