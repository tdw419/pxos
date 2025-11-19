// Activation Functions for Neural Networks
//
// Critical operations for LLM inference:
// - Softmax: Converts logits to probabilities (attention, output layer)
// - GELU: Smooth activation for feed-forward layers (GPT-2/3)
// - ReLU: Simple baseline activation
//
// Why these matter for Pixel-LLM:
// - Softmax enables attention mechanism (core of transformers)
// - GELU provides non-linearity in feed-forward layers
// - All operations must preserve numerical stability

// Input/Output buffers
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> metadata: array<u32>;

// Shared memory for reductions
var<workgroup> shared_data: array<f32, 256>;

// =============================================================================
// SOFTMAX
// =============================================================================
//
// Converts logits to probabilities: output[i] = exp(x[i]) / sum(exp(x))
//
// Numerically stable version:
// 1. Find max value (prevents overflow in exp)
// 2. Compute exp(x - max)
// 3. Normalize by sum
//
// Two-pass algorithm:
// - Pass 1: Find max value (this kernel does both passes)
// - Pass 2: Compute exp and sum
// - Pass 3: Normalize

// Pass 1: Find maximum value in array (for numerical stability)
@compute @workgroup_size(256, 1, 1)
fn softmax_find_max(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let n = metadata[0];
    let idx = global_id.x;

    // Load value (or -inf if out of bounds)
    var local_max: f32 = -3.402823466e+38;  // -FLT_MAX
    if (idx < n) {
        local_max = input[idx];
    }

    shared_data[local_id.x] = local_max;
    workgroupBarrier();

    // Parallel reduction to find max
    var stride = 128u;
    while (stride > 0u) {
        if (local_id.x < stride && (local_id.x + stride) < 256u) {
            shared_data[local_id.x] = max(shared_data[local_id.x], shared_data[local_id.x + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // First thread writes result
    if (local_id.x == 0u) {
        output[0] = shared_data[0];
    }
}

// Pass 2: Compute exp(x - max) and sum
@compute @workgroup_size(256, 1, 1)
fn softmax_exp_sum(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let n = metadata[0];
    let max_val = metadata[1];  // Pass max as float bits
    let max_f = bitcast<f32>(max_val);
    let idx = global_id.x;

    // Compute exp(x - max)
    var local_exp: f32 = 0.0;
    if (idx < n) {
        local_exp = exp(input[idx] - max_f);
        output[idx] = local_exp;  // Store for pass 3
    }

    shared_data[local_id.x] = local_exp;
    workgroupBarrier();

    // Parallel reduction to find sum
    var stride = 128u;
    while (stride > 0u) {
        if (local_id.x < stride && (local_id.x + stride) < 256u) {
            shared_data[local_id.x] = shared_data[local_id.x] + shared_data[local_id.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // First thread writes sum to metadata output buffer
    if (local_id.x == 0u) {
        // Write sum to second element of output
        output[n] = shared_data[0];
    }
}

// Pass 3: Normalize by sum
@compute @workgroup_size(256, 1, 1)
fn softmax_normalize(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let n = metadata[0];
    let sum_bits = metadata[1];
    let sum = bitcast<f32>(sum_bits);
    let idx = global_id.x;

    if (idx < n) {
        output[idx] = output[idx] / sum;
    }
}

// Combined softmax (for small vectors that fit in one workgroup)
@compute @workgroup_size(256, 1, 1)
fn softmax_combined(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let n = metadata[0];
    let idx = global_id.x;

    // Pass 1: Find max
    var local_val: f32 = -3.402823466e+38;
    if (idx < n) {
        local_val = input[idx];
    }
    shared_data[local_id.x] = local_val;
    workgroupBarrier();

    var stride = 128u;
    while (stride > 0u) {
        if (local_id.x < stride) {
            shared_data[local_id.x] = max(shared_data[local_id.x], shared_data[local_id.x + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let max_val = shared_data[0];
    workgroupBarrier();

    // Pass 2: Compute exp(x - max) and sum
    var local_exp: f32 = 0.0;
    if (idx < n) {
        local_exp = exp(input[idx] - max_val);
    }
    shared_data[local_id.x] = local_exp;
    workgroupBarrier();

    stride = 128u;
    while (stride > 0u) {
        if (local_id.x < stride) {
            shared_data[local_id.x] = shared_data[local_id.x] + shared_data[local_id.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let sum = shared_data[0];

    // Pass 3: Normalize
    if (idx < n) {
        output[idx] = local_exp / sum;
    }
}

// =============================================================================
// GELU (Gaussian Error Linear Unit)
// =============================================================================
//
// GELU(x) = x * Φ(x)
// where Φ(x) is the cumulative distribution function of standard normal
//
// Approximation used in GPT-2/3:
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
//
// Even simpler approximation:
// GELU(x) ≈ x * sigmoid(1.702 * x)

@compute @workgroup_size(256, 1, 1)
fn gelu(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let n = metadata[0];
    let idx = global_id.x;

    if (idx >= n) {
        return;
    }

    let x = input[idx];

    // Use the tanh approximation (same as GPT-2)
    let sqrt_2_over_pi = 0.7978845608;  // sqrt(2/π)
    let coeff = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
    let result = 0.5 * x * (1.0 + tanh(coeff));

    output[idx] = result;
}

// Fast GELU using sigmoid approximation
@compute @workgroup_size(256, 1, 1)
fn gelu_fast(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let n = metadata[0];
    let idx = global_id.x;

    if (idx >= n) {
        return;
    }

    let x = input[idx];

    // Fast approximation: x * sigmoid(1.702 * x)
    let sigmoid_input = 1.702 * x;
    let sigmoid = 1.0 / (1.0 + exp(-sigmoid_input));

    output[idx] = x * sigmoid;
}

// =============================================================================
// ReLU (Rectified Linear Unit)
// =============================================================================
//
// ReLU(x) = max(0, x)
// Simple but effective baseline activation

@compute @workgroup_size(256, 1, 1)
fn relu(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let n = metadata[0];
    let idx = global_id.x;

    if (idx >= n) {
        return;
    }

    output[idx] = max(0.0, input[idx]);
}

// =============================================================================
// SiLU / Swish (Sigmoid Linear Unit)
// =============================================================================
//
// SiLU(x) = x * sigmoid(x)
// Also known as Swish, used in some modern architectures

@compute @workgroup_size(256, 1, 1)
fn silu(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let n = metadata[0];
    let idx = global_id.x;

    if (idx >= n) {
        return;
    }

    let x = input[idx];
    let sigmoid = 1.0 / (1.0 + exp(-x));

    output[idx] = x * sigmoid;
}

// =============================================================================
// Leaky ReLU
// =============================================================================
//
// LeakyReLU(x) = x if x > 0, else alpha * x
// Prevents "dying ReLU" problem

@compute @workgroup_size(256, 1, 1)
fn leaky_relu(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let n = metadata[0];
    let alpha_bits = metadata[1];
    let alpha = bitcast<f32>(alpha_bits);  // Default: 0.01
    let idx = global_id.x;

    if (idx >= n) {
        return;
    }

    let x = input[idx];
    output[idx] = select(alpha * x, x, x > 0.0);
}
