// frontend/shaders/library.ts

export const SHADER_LIBRARY: Record<string, string> = {
    "life": `
@group(0) @binding(0) var<storage, read> inputGrid: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputGrid: array<u32>;
const WIDTH: u32 = 32u;
const HEIGHT: u32 = 32u;
fn get_idx(x: u32, y: u32) -> u32 { return y * WIDTH + x; }

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    if (x >= WIDTH || y >= HEIGHT) { return; }

    let idx = get_idx(x, y);
    let alive = inputGrid[idx] > 0u;
    var neighbors = 0u;
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            let nx = u32((i32(x) + dx + i32(WIDTH)) % i32(WIDTH));
            let ny = u32((i32(y) + dy + i32(HEIGHT)) % i32(HEIGHT));
            if (inputGrid[get_idx(nx, ny)] > 0u) { neighbors++; }
        }
    }

    var next_state = 0u;
    if (alive && (neighbors == 2u || neighbors == 3u)) { next_state = 0xFF00FF00u; }
    else if (!alive && neighbors == 3u) { next_state = 0xFF00FF00u; }
    outputGrid[idx] = next_state;
}
`,
    "plasma": `
@group(0) @binding(0) var<storage, read> inputGrid: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputGrid: array<u32>;

const WIDTH: u32 = 32u;
const HEIGHT: u32 = 32u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = f32(global_id.x);
    let y = f32(global_id.y);
    let time = f32(inputGrid[0] & 0xFFu) * 0.1; // Hack: use pixel 0 as time seed for now

    let v = sin(x * 0.1 + time) + sin(y * 0.1 + time) + sin((x + y) * 0.1 + time);
    let r = u32((sin(v) + 1.0) * 127.0);
    let g = u32((cos(v) + 1.0) * 127.0);
    let b = u32((sin(v + 3.14) + 1.0) * 127.0);

    let color = (0xFFu << 24) | (r << 16) | (g << 8) | b;
    outputGrid[global_id.y * WIDTH + global_id.x] = color;
}
`,
    "red_circle": `
@group(0) @binding(0) var<storage, read> inputGrid: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputGrid: array<u32>;

const WIDTH: u32 = 32u;
const HEIGHT: u32 = 32u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cx = 16.0;
    let cy = 16.0;
    let dx = f32(global_id.x) - cx;
    let dy = f32(global_id.y) - cy;
    let dist = sqrt(dx*dx + dy*dy);

    if (dist < 10.0) {
        outputGrid[global_id.y * WIDTH + global_id.x] = 0xFFFF0000u; // Red
    } else {
        outputGrid[global_id.y * WIDTH + global_id.x] = 0xFF000000u; // Black
    }
}
`,
    "gradient": `
@group(0) @binding(0) var<storage, read> inputGrid: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputGrid: array<u32>;

const WIDTH: u32 = 32u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let r = u32(f32(global_id.x) / 32.0 * 255.0);
    let b = u32(f32(global_id.y) / 32.0 * 255.0);
    let color = (0xFFu << 24) | (r << 16) | (0x00u << 8) | b;
    outputGrid[global_id.y * WIDTH + global_id.x] = color;
}
`
};

export const getShaderFromPrompt = (prompt: string): string => {
    const p = prompt.toLowerCase();
    if (p.includes("plasma")) return SHADER_LIBRARY["plasma"];
    if (p.includes("circle") || p.includes("red")) return SHADER_LIBRARY["red_circle"];
    if (p.includes("gradient")) return SHADER_LIBRARY["gradient"];
    if (p.includes("life") || p.includes("game")) return SHADER_LIBRARY["life"];

    // Default fallback template
    return `
@group(0) @binding(0) var<storage, read> inputGrid: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputGrid: array<u32>;
const WIDTH: u32 = 32u;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Generated for prompt: "${prompt}"
    outputGrid[global_id.y * WIDTH + global_id.x] = 0xFF0000FFu; // Blue fallback
}
`;
};
