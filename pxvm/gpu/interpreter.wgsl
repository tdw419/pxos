// WGSL pixel VM interpreter
// Executes the exact same .pxi format as CPU version

struct ImageLayout {
    width: u32,
    height: u32,
    instruction_row: u32,  // = 0
}

@group(0) @binding(0) var<storage, read_write> pixels: array<vec4<u32>>;  // RGBA8 packed as u32
@group(0) @binding(1) var<uniform> image_layout: ImageLayout;

const OP_HALT: u32 = 0u;
const OP_DOT_R: u32 = 1u;  // Renamed from OP_DOT_RGB for clarity

fn pixel_at(x: u32, y: u32) -> vec4<u32> {
    let idx = y * image_layout.width + x;
    return pixels[idx];
}

fn set_pixel(x: u32, y: u32, val: vec4<u32>) {
    let idx = y * image_layout.width + x;
    pixels[idx] = val;
}

// Same zero-delimited vector length inference
fn vector_length(row_a: u32, row_b: u32) -> u32 {
    var len: u32 = 0u;
    loop {
        if len >= image_layout.width { break; }
        let a = pixel_at(len, row_a);
        let b = pixel_at(len, row_b);
        if all(a == vec4<u32>(0u)) && all(b == vec4<u32>(0u)) {
            break;
        }
        len = len + 1u;
    }
    return len;
}

// Execute one OP_DOT_R instruction (in-place)
fn exec_dot_r(instr: vec4<u32>) {
    let row_a = instr.y;      // ARG0 in G channel
    let row_b = instr.z;      // ARG1 in B channel
    let row_out = instr.w;    // ARG2 in A channel

    let len = vector_length(row_a, row_b);

    var dot: u32 = 0u;
    for (var i: u32 = 0u; i < len; i = i + 1u) {
        let a_r = pixel_at(i, row_a).x;
        let b_r = pixel_at(i, row_b).x;
        dot = dot + a_r * b_r;
    }

    let low = dot & 0xFFu;
    let high = (dot >> 8u) & 0xFFu;

    var result = pixel_at(0u, row_out);
    result.x = low;
    result.y = high;
    set_pixel(0u, row_out, result);
}

// Main interpreter loop (invoked per-workgroup)
@compute @workgroup_size(1, 1, 1)
fn run_program() {
    var pc_x: u32 = 0u;  // Program counter (x coordinate)

    loop {
        if pc_x >= image_layout.width { break; }

        let instr = pixel_at(pc_x, image_layout.instruction_row);
        let opcode = instr.x;

        if opcode == OP_HALT {
            break;
        }

        if opcode == OP_DOT_R {
            exec_dot_r(instr);
        }

        pc_x = pc_x + 1u;
    }
}
