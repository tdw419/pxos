// Shader Virtual Machine Runtime
// Executes bytecode on GPU compute shaders

struct Uniforms {
    time: f32,
    resolution: vec2<f32>,
    mouse: vec2<f32>,
    _padding: f32,
}

struct VMState {
    stack: array<f32, 256>,
    memory: array<f32, 1024>,
    stack_ptr: u32,
    program_ptr: u32,
    halted: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> bytecode: array<u32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba8unorm, write>;

// Opcode constants (must match Python Opcode enum)
const OP_PUSH: u32 = 0u;
const OP_POP: u32 = 1u;
const OP_DUP: u32 = 2u;
const OP_SWAP: u32 = 3u;

const OP_ADD: u32 = 10u;
const OP_SUB: u32 = 11u;
const OP_MUL: u32 = 12u;
const OP_DIV: u32 = 13u;
const OP_MOD: u32 = 14u;
const OP_NEG: u32 = 15u;

const OP_SIN: u32 = 20u;
const OP_COS: u32 = 21u;
const OP_TAN: u32 = 22u;
const OP_ASIN: u32 = 23u;
const OP_ACOS: u32 = 24u;
const OP_ATAN: u32 = 25u;
const OP_ATAN2: u32 = 26u;
const OP_EXP: u32 = 27u;
const OP_LOG: u32 = 28u;
const OP_POW: u32 = 29u;
const OP_SQRT: u32 = 30u;
const OP_ABS: u32 = 31u;
const OP_FLOOR: u32 = 32u;
const OP_CEIL: u32 = 33u;
const OP_FRACT: u32 = 34u;
const OP_MIN: u32 = 35u;
const OP_MAX: u32 = 36u;
const OP_CLAMP: u32 = 37u;

const OP_LT: u32 = 40u;
const OP_GT: u32 = 41u;
const OP_LE: u32 = 42u;
const OP_GE: u32 = 43u;
const OP_EQ: u32 = 44u;
const OP_NE: u32 = 45u;

const OP_AND: u32 = 50u;
const OP_OR: u32 = 51u;
const OP_NOT: u32 = 52u;

const OP_JMP: u32 = 60u;
const OP_JMP_IF: u32 = 61u;
const OP_JMP_IF_NOT: u32 = 62u;

const OP_UV: u32 = 70u;
const OP_TIME: u32 = 71u;
const OP_RESOLUTION: u32 = 72u;
const OP_MOUSE: u32 = 73u;

const OP_RGB: u32 = 80u;
const OP_RGBA: u32 = 81u;
const OP_HSV: u32 = 82u;
const OP_COLOR: u32 = 83u;

const OP_CIRCLE: u32 = 100u;
const OP_RECT: u32 = 101u;

const OP_HALT: u32 = 255u;

// HSV to RGB conversion
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let h6 = h * 6.0;
    let x = c * (1.0 - abs(h6 % 2.0 - 1.0));
    let m = v - c;

    var rgb: vec3<f32>;
    if (h6 < 1.0) {
        rgb = vec3<f32>(c, x, 0.0);
    } else if (h6 < 2.0) {
        rgb = vec3<f32>(x, c, 0.0);
    } else if (h6 < 3.0) {
        rgb = vec3<f32>(0.0, c, x);
    } else if (h6 < 4.0) {
        rgb = vec3<f32>(0.0, x, c);
    } else if (h6 < 5.0) {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }

    return rgb + vec3<f32>(m, m, m);
}

// Execute bytecode for a single pixel
fn execute_bytecode(pixel_coord: vec2<u32>) -> vec4<f32> {
    var stack: array<f32, 256>;
    var sp: u32 = 0u;
    var pc: u32 = 0u;

    let uv = vec2<f32>(pixel_coord) / uniforms.resolution;
    let max_instructions: u32 = 10000u;
    var instruction_count: u32 = 0u;

    loop {
        if (pc >= arrayLength(&bytecode)) {
            break;
        }

        if (instruction_count >= max_instructions) {
            // Infinite loop protection - return error color
            return vec4<f32>(1.0, 0.0, 1.0, 1.0);
        }
        instruction_count += 1u;

        let opcode = bytecode[pc];
        pc += 1u;

        switch opcode {
            // Stack operations
            case OP_PUSH: {
                let val = bitcast<f32>(bytecode[pc]);
                stack[sp] = val;
                sp += 1u;
                pc += 1u;
            }
            case OP_POP: {
                if (sp > 0u) {
                    sp -= 1u;
                }
            }
            case OP_DUP: {
                if (sp > 0u) {
                    stack[sp] = stack[sp - 1u];
                    sp += 1u;
                }
            }
            case OP_SWAP: {
                if (sp >= 2u) {
                    let temp = stack[sp - 1u];
                    stack[sp - 1u] = stack[sp - 2u];
                    stack[sp - 2u] = temp;
                }
            }

            // Arithmetic
            case OP_ADD: {
                if (sp >= 2u) {
                    let b = stack[sp - 1u];
                    let a = stack[sp - 2u];
                    stack[sp - 2u] = a + b;
                    sp -= 1u;
                }
            }
            case OP_SUB: {
                if (sp >= 2u) {
                    let b = stack[sp - 1u];
                    let a = stack[sp - 2u];
                    stack[sp - 2u] = a - b;
                    sp -= 1u;
                }
            }
            case OP_MUL: {
                if (sp >= 2u) {
                    let b = stack[sp - 1u];
                    let a = stack[sp - 2u];
                    stack[sp - 2u] = a * b;
                    sp -= 1u;
                }
            }
            case OP_DIV: {
                if (sp >= 2u) {
                    let b = stack[sp - 1u];
                    let a = stack[sp - 2u];
                    if (abs(b) > 0.0001) {
                        stack[sp - 2u] = a / b;
                    } else {
                        stack[sp - 2u] = 0.0; // Avoid divide by zero
                    }
                    sp -= 1u;
                }
            }
            case OP_NEG: {
                if (sp > 0u) {
                    stack[sp - 1u] = -stack[sp - 1u];
                }
            }

            // Math functions
            case OP_SIN: {
                if (sp > 0u) {
                    stack[sp - 1u] = sin(stack[sp - 1u]);
                }
            }
            case OP_COS: {
                if (sp > 0u) {
                    stack[sp - 1u] = cos(stack[sp - 1u]);
                }
            }
            case OP_TAN: {
                if (sp > 0u) {
                    stack[sp - 1u] = tan(stack[sp - 1u]);
                }
            }
            case OP_SQRT: {
                if (sp > 0u) {
                    stack[sp - 1u] = sqrt(max(0.0, stack[sp - 1u]));
                }
            }
            case OP_ABS: {
                if (sp > 0u) {
                    stack[sp - 1u] = abs(stack[sp - 1u]);
                }
            }
            case OP_FLOOR: {
                if (sp > 0u) {
                    stack[sp - 1u] = floor(stack[sp - 1u]);
                }
            }
            case OP_CEIL: {
                if (sp > 0u) {
                    stack[sp - 1u] = ceil(stack[sp - 1u]);
                }
            }
            case OP_FRACT: {
                if (sp > 0u) {
                    stack[sp - 1u] = fract(stack[sp - 1u]);
                }
            }
            case OP_MIN: {
                if (sp >= 2u) {
                    let b = stack[sp - 1u];
                    let a = stack[sp - 2u];
                    stack[sp - 2u] = min(a, b);
                    sp -= 1u;
                }
            }
            case OP_MAX: {
                if (sp >= 2u) {
                    let b = stack[sp - 1u];
                    let a = stack[sp - 2u];
                    stack[sp - 2u] = max(a, b);
                    sp -= 1u;
                }
            }
            case OP_POW: {
                if (sp >= 2u) {
                    let exp = stack[sp - 1u];
                    let base = stack[sp - 2u];
                    stack[sp - 2u] = pow(base, exp);
                    sp -= 1u;
                }
            }
            case OP_ATAN2: {
                if (sp >= 2u) {
                    let x = stack[sp - 1u];
                    let y = stack[sp - 2u];
                    stack[sp - 2u] = atan2(y, x);
                    sp -= 1u;
                }
            }

            // Comparisons (return 1.0 for true, 0.0 for false)
            case OP_LT: {
                if (sp >= 2u) {
                    let b = stack[sp - 1u];
                    let a = stack[sp - 2u];
                    stack[sp - 2u] = select(0.0, 1.0, a < b);
                    sp -= 1u;
                }
            }
            case OP_GT: {
                if (sp >= 2u) {
                    let b = stack[sp - 1u];
                    let a = stack[sp - 2u];
                    stack[sp - 2u] = select(0.0, 1.0, a > b);
                    sp -= 1u;
                }
            }
            case OP_EQ: {
                if (sp >= 2u) {
                    let b = stack[sp - 1u];
                    let a = stack[sp - 2u];
                    stack[sp - 2u] = select(0.0, 1.0, abs(a - b) < 0.0001);
                    sp -= 1u;
                }
            }

            // Graphics primitives
            case OP_UV: {
                stack[sp] = uv.x;
                stack[sp + 1u] = uv.y;
                sp += 2u;
            }
            case OP_TIME: {
                stack[sp] = uniforms.time;
                sp += 1u;
            }
            case OP_RESOLUTION: {
                stack[sp] = uniforms.resolution.x;
                stack[sp + 1u] = uniforms.resolution.y;
                sp += 2u;
            }
            case OP_MOUSE: {
                stack[sp] = uniforms.mouse.x;
                stack[sp + 1u] = uniforms.mouse.y;
                sp += 2u;
            }

            // Color operations
            case OP_HSV: {
                if (sp >= 3u) {
                    let v = stack[sp - 1u];
                    let s = stack[sp - 2u];
                    let h = stack[sp - 3u];
                    let rgb = hsv_to_rgb(h, s, v);
                    stack[sp - 3u] = rgb.r;
                    stack[sp - 2u] = rgb.g;
                    stack[sp - 1u] = rgb.b;
                }
            }
            case OP_COLOR: {
                // Output final color and terminate
                if (sp >= 4u) {
                    let a = clamp(stack[sp - 1u], 0.0, 1.0);
                    let b = clamp(stack[sp - 2u], 0.0, 1.0);
                    let g = clamp(stack[sp - 3u], 0.0, 1.0);
                    let r = clamp(stack[sp - 4u], 0.0, 1.0);
                    return vec4<f32>(r, g, b, a);
                }
                return vec4<f32>(1.0, 0.0, 0.0, 1.0); // Error: not enough values
            }

            // Control flow
            case OP_JMP: {
                let target = bitcast<u32>(bitcast<f32>(bytecode[pc]));
                pc = target;
            }
            case OP_JMP_IF: {
                if (sp > 0u) {
                    let condition = stack[sp - 1u];
                    sp -= 1u;
                    let target = bitcast<u32>(bitcast<f32>(bytecode[pc]));
                    pc += 1u;
                    if (condition != 0.0) {
                        pc = target;
                    }
                }
            }

            case OP_HALT: {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }

            default: {
                // Unknown opcode - return magenta error color
                return vec4<f32>(1.0, 0.0, 1.0, 1.0);
            }
        }
    }

    // Reached end without COLOR instruction - return black
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_coord = global_id.xy;

    // Bounds check
    if (pixel_coord.x >= u32(uniforms.resolution.x) || pixel_coord.y >= u32(uniforms.resolution.y)) {
        return;
    }

    let color = execute_bytecode(pixel_coord);
    textureStore(output_texture, pixel_coord, color);
}
