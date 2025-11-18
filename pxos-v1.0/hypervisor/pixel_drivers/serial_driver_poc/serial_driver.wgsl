// Pixel-Native Serial Driver (WebGPU Compute Shader)
//
// Executes pixel-encoded UART operations in parallel on GPU
// Each pixel = one serial operation (RGBA encoding)

// Input: Pixel-encoded serial operations (texture)
@group(0) @binding(0) var serial_ops_texture: texture_2d<f32>;

// Output: Serial output buffer (simulated UART MMIO)
@group(0) @binding(1) var<storage, read_write> uart_output: array<u32>;

// Statistics/debugging
@group(0) @binding(2) var<storage, read_write> stats: array<atomic<u32>>;

// UART registers (simulated MMIO)
const UART_DATA: u32 = 0u;      // Data register
const UART_STATUS: u32 = 1u;    // Status register
const UART_CONTROL: u32 = 2u;   // Control register

// Statistics indices
const STAT_OPS_EXECUTED: u32 = 0u;
const STAT_CHARS_WRITTEN: u32 = 1u;
const STAT_ERRORS: u32 = 2u;

// Decode pixel into serial operation
struct SerialOp {
    data: u32,          // Character/data byte (R channel)
    baud_divisor: u32,  // Baud rate divisor (G channel)
    line_control: u32,  // Line control (B channel)
    op_type: u32,       // Operation type (A channel)
}

fn decode_pixel(pixel: vec4<f32>) -> SerialOp {
    var op: SerialOp;
    op.data = u32(pixel.r * 255.0);
    op.baud_divisor = u32(pixel.g * 255.0);
    op.line_control = u32(pixel.b * 255.0);
    op.op_type = u32(pixel.a * 255.0);
    return op;
}

// Execute serial write operation
fn execute_write(op: SerialOp, output_idx: u32) {
    // Write character to simulated UART data register
    // In real implementation, this would be MMIO to 0x3F8 (COM1)
    uart_output[output_idx] = op.data;

    // Update statistics
    atomicAdd(&stats[STAT_CHARS_WRITTEN], 1u);
}

// Execute serial read operation
fn execute_read(op: SerialOp, output_idx: u32) {
    // Read from simulated UART data register
    // In real implementation, this would be MMIO from 0x3F8 (COM1)
    let data = uart_output[output_idx];

    // For now, just echo back (loopback mode)
    uart_output[output_idx] = data;
}

// Main compute shader - processes serial operations in parallel
@compute @workgroup_size(256)
fn serial_driver_main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Load pixel from texture
    let pixel = textureLoad(serial_ops_texture, vec2<i32>(i32(id.x), i32(id.y)), 0);

    // Decode operation
    let op = decode_pixel(pixel);

    // Skip NOP operations (A channel = 0)
    if (op.op_type == 0u) {
        return;
    }

    // Calculate output index
    let output_idx = id.y * 256u + id.x;

    // Dispatch operation
    switch (op.op_type) {
        case 1u: {  // Write operation
            execute_write(op, output_idx);
        }
        case 2u: {  // Read operation
            execute_read(op, output_idx);
        }
        default: {  // Unknown operation
            atomicAdd(&stats[STAT_ERRORS], 1u);
            return;
        }
    }

    // Update operations executed counter
    atomicAdd(&stats[STAT_OPS_EXECUTED], 1u);
}

// Alternative entry point for batched operations
// Processes multiple rows in parallel for higher throughput
@compute @workgroup_size(256, 1, 1)
fn serial_driver_batch(@builtin(global_invocation_id) id: vec3<u32>) {
    // Get texture dimensions
    let dims = textureDimensions(serial_ops_texture);

    // Process entire column (all rows for this X coordinate)
    for (var y = 0u; y < u32(dims.y); y = y + 1u) {
        let pixel = textureLoad(serial_ops_texture, vec2<i32>(i32(id.x), i32(y)), 0);
        let op = decode_pixel(pixel);

        if (op.op_type == 0u) {
            continue;
        }

        let output_idx = y * 256u + id.x;

        switch (op.op_type) {
            case 1u: {
                execute_write(op, output_idx);
            }
            case 2u: {
                execute_read(op, output_idx);
            }
            default: {
                atomicAdd(&stats[STAT_ERRORS], 1u);
            }
        }

        atomicAdd(&stats[STAT_OPS_EXECUTED], 1u);
    }
}
