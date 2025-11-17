// pxvm/gpu/matmul.wgsl
//
// Matrix multiplication compute shader for pxVM GPU execution.
//
// Computes C = A @ B where:
//   A: [M, K] matrix
//   B: [K, N] matrix
//   C: [M, N] output matrix
//
// Workgroup size: 8×8 threads
// Each thread computes one element of C

// Input matrix A (read-only)
@group(0) @binding(0)
var<storage, read> A: array<f32>;

// Input matrix B (read-only)
@group(0) @binding(1)
var<storage, read> B: array<f32>;

// Output matrix C (write)
@group(0) @binding(2)
var<storage, read_write> C: array<f32>;

// Dimensions: [M, N, K]
@group(0) @binding(3)
var<uniform> dims: vec3<u32>;

@compute @workgroup_size(8, 8, 1)
fn matmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;  // M dimension
    let col = gid.y;  // N dimension

    let M = dims.x;
    let N = dims.y;
    let K = dims.z;

    // Bounds check
    if (row >= M || col >= N) {
        return;
    }

    // Compute dot product of row from A and column from B
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        let a_val = A[row * K + k];      // A[row, k]
        let b_val = B[k * N + col];      // B[k, col]
        sum = sum + (a_val * b_val);
    }

    // Write result
    C[row * N + col] = sum;
}
