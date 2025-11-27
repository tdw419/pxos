import os
import json
import traceback
import wgpu
import wgpu.backends.wgpu_native
from flask import Flask, request, jsonify
import numpy as np
import openai

app = Flask(__name__)

# System Prompt for the Shader Lab
SHADER_LAB_SYSTEM_PROMPT = """
You are a WGSL (WebGPU Shader Language) code generator.
Your ONLY task is to write a valid WGSL compute shader that renders a 32x32 image based on the user's prompt.
Rules:
1. Output ONLY the WGSL code in a fenced block. No prose.
2. Assume the output is a storage buffer at binding 0 with 1024 u32 elements (32x32 pixels).
   The buffer type should be `array<u32>`.
3. Use @compute and @workgroup_size(8, 8) for parallelism.
4. If the prompt is impossible, return "// ERROR: [reason]".
5. Write final color as u32: 0xAABBGGRR (Alpha=High, Red=Low).
   Example: 0xFF0000FF is Red (if RGBA) or Blue?
   Let's stick to standard ABGR packing usually:
   Actually, let's use 0xFF00FF00 for Green.
   Format: 0xAABBGGRR (Little Endian u32) -> R is lowest byte.
   Wait, HTML5 Canvas ImageData expects RGBA.
   Little Endian u32 0xAABBGGRR -> Memory: RR GG BB AA.
   So R is low byte.
   So to write Red: 0xFF0000FF.
   To write Green: 0xFF00FF00.

Example:
Prompt: "draw a red circle"
Output:
```wgsl
@group(0) @binding(0) var<storage, read_write> outputGrid: array<u32>;

const WIDTH: u32 = 32u;
const HEIGHT: u32 = 32u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    if (x >= WIDTH || y >= HEIGHT) { return; }

    let idx = y * WIDTH + x;
    let center = vec2f(16.0, 16.0);
    let pos = vec2f(f32(x), f32(y));
    let dist = distance(pos, center);

    if (dist < 10.0) {
        outputGrid[idx] = 0xFF0000FFu; // Red
    } else {
        outputGrid[idx] = 0xFF000000u; // Black
    }
}
```
"""

# Mock LLM response for testing if OPENAI_API_KEY is missing
MOCK_WGSL = """
@group(0) @binding(0) var<storage, read_write> outputGrid: array<u32>;
const WIDTH: u32 = 32u;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.y * WIDTH + id.x;
    if (idx >= 1024u) { return; }
    // Gradient: Red increases with X, Green with Y
    let r = id.x * 8u;
    let g = id.y * 8u;
    outputGrid[idx] = 0xFF000000u | (g << 8) | r;
}
"""

def generate_wgsl_from_llm(prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Using mock WGSL.")
        return MOCK_WGSL

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SHADER_LAB_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        content = response.choices[0].message.content
        # Extract code block
        if "```wgsl" in content:
            return content.split("```wgsl")[1].split("```")[0].strip()
        elif "```" in content:
            return content.split("```")[1].split("```")[0].strip()
        return content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return MOCK_WGSL

def execute_wgsl_on_backend(wgsl_code):
    try:
        device = wgpu.utils.get_default_device()

        shader = device.create_shader_module(code=wgsl_code)

        # Buffer size: 32x32 pixels * 4 bytes/pixel = 4096 bytes
        buffer_size = 32 * 32 * 4

        # Create storage buffer
        storage_buffer = device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )

        # Bind group layout (inferred or explicit)
        bind_group_layout = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}}
        ])

        bind_group = device.create_bind_group(
            layout=bind_group_layout,
            entries=[{"binding": 0, "resource": {"buffer": storage_buffer}}]
        )

        pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])

        compute_pipeline = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader, "entry_point": "main"}
        )

        command_encoder = device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(compute_pipeline)
        compute_pass.set_bind_group(0, bind_group, [], 0, 99999)
        compute_pass.dispatch_workgroups(4, 4, 1) # 32x32 / 8x8 = 4x4 workgroups
        compute_pass.end()

        device.queue.submit([command_encoder.finish()])

        # Read buffer
        output_data = device.queue.read_buffer(storage_buffer)
        return np.frombuffer(output_data, dtype=np.uint32).tolist()

    except Exception as e:
        print(f"WGPU Execution Error: {e}")
        traceback.print_exc()
        return None

@app.route('/api/generate-shader', methods=['POST'])
def generate_shader():
    data = request.json
    prompt = data.get("prompt", "")

    print(f"Received prompt: {prompt}")

    # 1. Generate WGSL
    wgsl_code = generate_wgsl_from_llm(prompt)
    print("Generated WGSL.")

    # 2. Execute on Backend (Validation)
    framebuffer = execute_wgsl_on_backend(wgsl_code)

    if framebuffer:
        print("Execution successful.")
        return jsonify({
            "status": "success",
            "wgsl": wgsl_code,
            "framebuffer": framebuffer
        })
    else:
        print("Execution failed.")
        return jsonify({
            "status": "error",
            "error": "Failed to compile or execute WGSL on backend",
            "wgsl": wgsl_code
        }), 400

if __name__ == '__main__':
    app.run(port=5000)
