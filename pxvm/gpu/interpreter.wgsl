// pxvm/gpu/interpreter.wgsl
//
// WGSL Pixel VM Interpreter
//
// Executes the exact same .pxi format as the CPU interpreter.
// Proves that the Pixel Protocol is executor-agnostic.
//
// Layout:
// - pixels: RGBA8 image as array of vec4<u32> (unpacked from uint8)
// - Row 0: instruction pixels
// - Rows 1+: data/output pixels
//
// Instruction encoding (per pixel):
//   R = OPCODE
//   G = ARG0
//   B = ARG1
//   A = ARG2

struct ImageLayout {
    width: u32,
    height: u32,
    instruction_row: u32,  // Always 0 for v0.0.1
}

@group(0) @binding(0)
var<storage, read_write> pixels: array<vec4<u32>>;

@group(0) @binding(1)
var<uniform> layout: ImageLayout;

// Opcodes (must match pxvm/core/opcodes.py)
const OP_HALT: u32 = 0u;
const OP_DOT_RGB: u32 = 1u;

// Access pixel at (x, y)
fn pixel_at(x: u32, y: u32) -> vec4<u32> {
    let idx = y * layout.width + x;
    return pixels[idx];
}

// Write pixel at (x, y)
fn set_pixel(x: u32, y: u32, val: vec4<u32>) {
    let idx = y * layout.width + x;
    pixels[idx] = val;
}

// Infer vector length by scanning rows until both pixels are zero
fn vector_length(row_a: u32, row_b: u32) -> u32 {
    var len: u32 = 0u;

    for (var x: u32 = 0u; x < layout.width; x = x + 1u) {
        let a = pixel_at(x, row_a);
        let b = pixel_at(x, row_b);

        // Both pixels fully zero = end of vectors
        if (all(a == vec4<u32>(0u, 0u, 0u, 0u)) &&
            all(b == vec4<u32>(0u, 0u, 0u, 0u))) {
            break;
        }

        len = len + 1u;
    }

    return len;
}

// Execute OP_DOT_RGB instruction
// instr.r = opcode (1)
// instr.g = row index for vector A
// instr.b = row index for vector B
// instr.a = row index for output
fn exec_dot_rgb(instr: vec4<u32>) {
    let row_a = instr.g;      // ARG0
    let row_b = instr.b;      // ARG1
    let row_out = instr.a;    // ARG2

    // Bounds check
    if (row_a >= layout.height || row_b >= layout.height || row_out >= layout.height) {
        return;  // Out of bounds, skip
    }

    // Infer vector length
    let len = vector_length(row_a, row_b);

    // Compute dot product (R channel only)
    var dot: u32 = 0u;
    for (var i: u32 = 0u; i < len; i = i + 1u) {
        let a_r = pixel_at(i, row_a).r;
        let b_r = pixel_at(i, row_b).r;
        dot = dot + (a_r * b_r);
    }

    // Encode as 16-bit little-endian in (R, G)
    let low = dot & 0xFFu;
    let high = (dot >> 8u) & 0xFFu;

    // Read current output pixel
    var result = pixel_at(0u, row_out);

    // Write low/high bytes
    result.r = low;
    result.g = high;
    // B and A remain unchanged

    set_pixel(0u, row_out, result);
}

// Main interpreter entry point
// Executes instruction pixels from row 0, left to right
@compute @workgroup_size(1, 1, 1)
fn run_program() {
    var pc_x: u32 = 0u;  // Program counter (column in instruction row)
    let max_steps: u32 = 1024u;  // Safety limit
    var steps: u32 = 0u;

    // Fetch-decode-execute loop
    loop {
        if (pc_x >= layout.width) {
            break;  // End of instruction row
        }

        if (steps >= max_steps) {
            break;  // Safety limit
        }

        // Fetch instruction
        let instr = pixel_at(pc_x, layout.instruction_row);
        let opcode = instr.r;

        // Decode and execute
        if (opcode == OP_HALT) {
            break;  // Halt execution
        }

        if (opcode == OP_DOT_RGB) {
            exec_dot_rgb(instr);
        }

        // Advance PC
        pc_x = pc_x + 1u;
        steps = steps + 1u;
    }
}
