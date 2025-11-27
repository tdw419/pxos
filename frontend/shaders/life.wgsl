@group(0) @binding(0) var<storage, read> inputGrid: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputGrid: array<u32>;

const WIDTH: u32 = 32u;
const HEIGHT: u32 = 32u;

fn get_idx(x: u32, y: u32) -> u32 {
    return y * WIDTH + x;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= WIDTH || y >= HEIGHT) {
        return;
    }

    let idx = get_idx(x, y);
    let alive = inputGrid[idx] > 0u;
    var neighbors = 0u;

    // Iterate 3x3 grid
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }

            // Toroidal wrap
            let nx = u32((i32(x) + dx + i32(WIDTH)) % i32(WIDTH));
            let ny = u32((i32(y) + dy + i32(HEIGHT)) % i32(HEIGHT));

            if (inputGrid[get_idx(nx, ny)] > 0u) {
                neighbors++;
            }
        }
    }

    var next_state = 0u; // Dead (Black)
    // ARGB: Alpha=FF (255), R=00, G=FF (255), B=00 -> 0xFF00FF00
    let ALIVE_COLOR = 0xFF00FF00u;

    if (alive) {
        if (neighbors == 2u || neighbors == 3u) {
            next_state = ALIVE_COLOR;
        }
    } else {
        if (neighbors == 3u) {
            next_state = ALIVE_COLOR;
        }
    }

    outputGrid[idx] = next_state;
}
