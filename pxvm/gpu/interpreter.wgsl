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
const OP_ADD: u32 = 2u;
const OP_RELU: u32 = 3u;
const OP_MATMUL: u32 = 4u;  // Reserved for future

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

// Execute OP_ADD instruction (element-wise addition)
fn exec_add(instr: vec4<u32>) {
    let row_a = instr.g;      // ARG0
    let row_b = instr.b;      // ARG1
    let row_out = instr.a;    // ARG2

    // Bounds check
    if (row_a >= layout.height || row_b >= layout.height || row_out >= layout.height) {
        return;  // Out of bounds, skip
    }

    // Infer vector length
    let len = vector_length(row_a, row_b);

    // Element-wise addition (R channel only)
    for (var i: u32 = 0u; i < len; i = i + 1u) {
        let a_r = pixel_at(i, row_a).r;
        let b_r = pixel_at(i, row_b).r;
        let sum_val = a_r + b_r;

        // Clamp to uint8 range (0-255)
        let clamped = min(255u, max(0u, sum_val));

        // Write to output row
        var result = pixel_at(i, row_out);
        result.r = clamped;
        // G, B, A remain unchanged
        set_pixel(i, row_out, result);
    }
}

// Execute OP_RELU instruction (in-place ReLU activation)
fn exec_relu(instr: vec4<u32>) {
    let row_data = instr.g;   // ARG0

    // Bounds check
    if (row_data >= layout.height) {
        return;  // Out of bounds, skip
    }

    // Apply ReLU to all pixels in row (R channel)
    // For uint8: ReLU(x) = max(x, 0) = x (since x >= 0)
    // This structure becomes meaningful with signed/float support
    for (var i: u32 = 0u; i < layout.width; i = i + 1u) {
        var px = pixel_at(i, row_data);
        px.r = max(0u, px.r);
        set_pixel(i, row_data, px);
    }
}

// Read matrix shape from header pixel
fn read_shape(row_start: u32) -> vec2<u32> {
    let hdr = pixel_at(0u, row_start);
    let cols = hdr.r | (hdr.g << 8u);
    let rows = hdr.b | (hdr.a << 8u);
    return vec2<u32>(cols, rows);
}

// Get value from flattened matrix data
fn get_matrix_val(row_start: u32, idx: u32) -> u32 {
    let stride = layout.width - 1u;  // Column 0 is header
    let x = 1u + (idx % stride);
    let y = row_start + (idx / stride);
    return pixel_at(x, y).r;
}

// Set value in flattened matrix data
fn set_matrix_val(row_start: u32, idx: u32, v: u32) {
    let stride = layout.width - 1u;  // Column 0 is header
    let x = 1u + (idx % stride);
    let y = row_start + (idx / stride);
    var p = pixel_at(x, y);
    p.r = v;
    set_pixel(x, y, p);
}

// Execute OP_MATMUL instruction (matrix multiply)
fn exec_matmul(instr: vec4<u32>) {
    let row_a = instr.g;  // ARG0 - A matrix row
    let row_b = instr.b;  // ARG1 - B matrix row
    let row_c = instr.a;  // ARG2 - C matrix row (output)

    // Bounds check
    if (row_a >= layout.height || row_b >= layout.height || row_c >= layout.height) {
        return;  // Out of bounds, skip
    }

    // Read matrix shapes
    let a_shape = read_shape(row_a);  // (cols, rows) = (K, M)
    let b_shape = read_shape(row_b);  // (cols, rows) = (N, K)

    let K = a_shape.x;  // Inner dimension
    let M = a_shape.y;  // Output rows
    let N = b_shape.x;  // Output cols

    // Validate inner dimension
    if (b_shape.y != K) {
        return;  // Invalid shapes for multiplication
    }

    // Write output shape header at row_c
    var hdr = pixel_at(0u, row_c);
    hdr.r = N & 0xFFu;         // cols_low
    hdr.g = (N >> 8u) & 0xFFu; // cols_high
    hdr.b = M & 0xFFu;         // rows_low
    hdr.a = (M >> 8u) & 0xFFu; // rows_high
    set_pixel(0u, row_c, hdr);

    // Compute C = A @ B
    // C[m,n] = sum_k A[m,k] * B[k,n]
    for (var m: u32 = 0u; m < M; m = m + 1u) {
        for (var n: u32 = 0u; n < N; n = n + 1u) {
            var acc: u32 = 0u;

            for (var k: u32 = 0u; k < K; k = k + 1u) {
                let a_idx = m * K + k;
                let b_idx = k * N + n;

                let a_val = get_matrix_val(row_a, a_idx);
                let b_val = get_matrix_val(row_b, b_idx);

                acc = acc + (a_val * b_val);
            }

            // Clamp to uint8 range
            let clamped = min(acc, 255u);

            let c_idx = m * N + n;
            set_matrix_val(row_c, c_idx, clamped);
        }
    }
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
        } else if (opcode == OP_ADD) {
            exec_add(instr);
        } else if (opcode == OP_RELU) {
            exec_relu(instr);
        } else if (opcode == OP_MATMUL) {
            exec_matmul(instr);
        }

        // Advance PC
        pc_x = pc_x + 1u;
        steps = steps + 1u;
    }
}
