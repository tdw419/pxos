// Matrix Multiplication Compute Shader
//
// Implements tiled matrix multiplication: C = A × B
//
// Where:
//   A: (M × K) matrix
//   B: (K × N) matrix
//   C: (M × N) result matrix
//
// Why this matters for Pixel-LLM:
// - Core operation for every neural layer
// - Foundation for attention mechanisms
// - Enables pixel-native weight matrices
//
// Algorithm: Tiled matrix multiplication using shared memory
// - Each workgroup computes a TILE_SIZE × TILE_SIZE block of C
// - Uses shared memory to cache tiles of A and B
// - Reduces global memory bandwidth by ~TILE_SIZE factor

// Workgroup tile size (16x16 = 256 threads, good GPU occupancy)
const TILE_SIZE: u32 = 16u;

// Input/Output buffers
@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;
@group(0) @binding(3) var<storage, read> metadata: array<u32>;

// Shared memory tiles (cached in GPU local memory)
var<workgroup> tile_a: array<f32, 256>;  // TILE_SIZE × TILE_SIZE
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn matmul_tiled(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    // Matrix dimensions from metadata
    // metadata[0] = M (rows of A)
    // metadata[1] = K (cols of A, rows of B)
    // metadata[2] = N (cols of B)
    let M = metadata[0];
    let K = metadata[1];
    let N = metadata[2];

    // Global position in output matrix C
    let row = global_id.y;  // Which row of C
    let col = global_id.x;  // Which column of C

    // Local position within tile
    let local_row = local_id.y;
    let local_col = local_id.x;

    // Accumulator for this thread's output element
    var sum: f32 = 0.0;

    // Number of tiles needed along K dimension
    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    // Process all tiles along K dimension
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        // Load tile of A into shared memory
        // Each thread loads one element
        let a_row = row;
        let a_col = t * TILE_SIZE + local_col;

        if (a_row < M && a_col < K) {
            tile_a[local_row * TILE_SIZE + local_col] = matrix_a[a_row * K + a_col];
        } else {
            tile_a[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Load tile of B into shared memory
        let b_row = t * TILE_SIZE + local_row;
        let b_col = col;

        if (b_row < K && b_col < N) {
            tile_b[local_row * TILE_SIZE + local_col] = matrix_b[b_row * N + b_col];
        } else {
            tile_b[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Synchronize to ensure all threads have loaded their data
        workgroupBarrier();

        // Compute partial dot product for this tile
        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tile_a[local_row * TILE_SIZE + k] * tile_b[k * TILE_SIZE + local_col];
        }

        // Synchronize before loading next tile
        workgroupBarrier();
    }

    // Write result to global memory
    if (row < M && col < N) {
        matrix_c[row * N + col] = sum;
    }
}

// Optimized version for square matrices (power of 2 sizes)
// Slightly faster due to simpler indexing
@compute @workgroup_size(16, 16, 1)
fn matmul_square(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    // Assume square matrices: N × N
    let N = metadata[0];

    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    var sum: f32 = 0.0;

    let num_tiles = N / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t = t + 1u) {
        // Load tiles (simplified indexing for square case)
        let a_col = t * TILE_SIZE + local_col;
        let b_row = t * TILE_SIZE + local_row;

        tile_a[local_row * TILE_SIZE + local_col] = matrix_a[row * N + a_col];
        tile_b[local_row * TILE_SIZE + local_col] = matrix_b[b_row * N + col];

        workgroupBarrier();

        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tile_a[local_row * TILE_SIZE + k] * tile_b[k * TILE_SIZE + local_col];
        }

        workgroupBarrier();
    }

    matrix_c[row * N + col] = sum;
}

// Matrix-vector multiplication (optimized special case)
// Computes: y = A × x
// Where A is (M × N) and x is (N × 1)
@compute @workgroup_size(256, 1, 1)
fn matvec(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    // metadata[0] = M (rows of A)
    // metadata[1] = N (cols of A, length of x)
    let M = metadata[0];
    let N = metadata[1];

    let row = global_id.x;

    if (row >= M) {
        return;
    }

    // Compute dot product of row with vector
    var sum: f32 = 0.0;

    for (var i = 0u; i < N; i = i + 1u) {
        // matrix_a contains A (stored row-major)
        // matrix_b contains x (vector)
        sum = sum + matrix_a[row * N + i] * matrix_b[i];
    }

    matrix_c[row] = sum;
}

// Transposed matrix multiplication: C = A^T × B
// Useful for backpropagation and weight updates
@compute @workgroup_size(16, 16, 1)
fn matmul_transposed_a(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    // A is (K × M) but we access it as transposed (M × K)
    // B is (K × N)
    // C is (M × N)
    let M = metadata[0];
    let K = metadata[1];
    let N = metadata[2];

    let row = global_id.y;
    let col = global_id.x;

    if (row >= M || col >= N) {
        return;
    }

    var sum: f32 = 0.0;

    // Dot product with transposed access pattern
    for (var k = 0u; k < K; k = k + 1u) {
        // A^T[row, k] = A[k, row]
        let a_val = matrix_a[k * M + row];  // Transposed access
        let b_val = matrix_b[k * N + col];
        sum = sum + a_val * b_val;
    }

    matrix_c[row * N + col] = sum;
}
