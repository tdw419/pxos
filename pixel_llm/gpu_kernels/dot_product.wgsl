// Dot Product Compute Shader
//
// The simplest possible GPU operation: dot product of two vectors.
// This is the foundation for ALL neural network operations.
//
// Compute: result = sum(a[i] * b[i]) for all i
//
// Why this matters for Pixel-LLM:
// - Proves GPU can read pixel data
// - Foundation for matrix multiplication
// - First step toward attention mechanisms

// Input/Output buffers
@group(0) @binding(0) var<storage, read> vector_a: array<f32>;
@group(0) @binding(1) var<storage, read> vector_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial_sums: array<f32>;
@group(0) @binding(3) var<storage, read> metadata: array<u32>;

// Workgroup shared memory for reduction
var<workgroup> shared_sums: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn dot_product_kernel(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    // metadata[0] = vector length
    let n = metadata[0];
    let global_idx = global_id.x;
    let local_idx = local_id.x;

    // Each thread computes one element's contribution
    var thread_sum: f32 = 0.0;

    if (global_idx < n) {
        thread_sum = vector_a[global_idx] * vector_b[global_idx];
    }

    // Store in shared memory
    shared_sums[local_idx] = thread_sum;

    // Synchronize workgroup
    workgroupBarrier();

    // Parallel reduction within workgroup (256 threads -> 1 sum)
    // This is faster than atomic operations
    var stride = 128u;
    while (stride > 0u) {
        if (local_idx < stride && (local_idx + stride) < 256u) {
            shared_sums[local_idx] = shared_sums[local_idx] + shared_sums[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // First thread in workgroup writes partial sum
    if (local_idx == 0u) {
        partial_sums[workgroup_id.x] = shared_sums[0];
    }
}

// Second pass: reduce partial sums (if needed for large vectors)
@compute @workgroup_size(256, 1, 1)
fn reduce_partial_sums(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let local_idx = local_id.x;
    let global_idx = global_id.x;

    // metadata[1] = number of partial sums
    let num_partials = metadata[1];

    // Load partial sum (or 0 if beyond range)
    var value: f32 = 0.0;
    if (global_idx < num_partials) {
        value = partial_sums[global_idx];
    }

    shared_sums[local_idx] = value;
    workgroupBarrier();

    // Reduce
    var stride = 128u;
    while (stride > 0u) {
        if (local_idx < stride && (local_idx + stride) < 256u) {
            shared_sums[local_idx] = shared_sums[local_idx] + shared_sums[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Write final result
    if (local_idx == 0u) {
        partial_sums[0] = shared_sums[0];
    }
}
