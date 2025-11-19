// Attention Mechanism Compute Shader
//
// Implements scaled dot-product attention and multi-head attention.
//
// Scaled Dot-Product Attention:
//   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
//
// Where:
//   Q: Queries (seq_len × d_k)
//   K: Keys (seq_len × d_k)
//   V: Values (seq_len × d_v)
//   Output: (seq_len × d_v)
//
// Why this matters for Pixel-LLM:
// - Attention is THE core mechanism of transformers
// - Enables context-aware token relationships
// - Foundation for pixel-native language understanding
//
// Algorithm:
// 1. Compute attention scores: S = Q @ K^T / sqrt(d_k)
// 2. Apply optional causal mask (for autoregressive generation)
// 3. Apply softmax row-wise: A = softmax(S)
// 4. Compute weighted sum: output = A @ V

// Input/Output buffers
@group(0) @binding(0) var<storage, read> query: array<f32>;      // Q: (seq_len, d_k)
@group(0) @binding(1) var<storage, read> key: array<f32>;        // K: (seq_len, d_k)
@group(0) @binding(2) var<storage, read> value: array<f32>;      // V: (seq_len, d_v)
@group(0) @binding(3) var<storage, read_write> output: array<f32>;  // Out: (seq_len, d_v)
@group(0) @binding(4) var<storage, read> metadata: array<u32>;

// Shared memory for attention computation
var<workgroup> shared_scores: array<f32, 256>;
var<workgroup> shared_key: array<f32, 256>;

// =============================================================================
// ATTENTION SCORES: Compute Q @ K^T / sqrt(d_k)
// =============================================================================
//
// Each thread computes one element of the attention score matrix.
// Score[i, j] = dot(Q[i], K[j]) / sqrt(d_k)

@compute @workgroup_size(16, 16, 1)
fn compute_attention_scores(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    // metadata[0] = seq_len
    // metadata[1] = d_k (query/key dimension)
    // metadata[2] = d_v (value dimension)
    let seq_len = metadata[0];
    let d_k = metadata[1];

    let i = global_id.y;  // Query index
    let j = global_id.x;  // Key index

    if (i >= seq_len || j >= seq_len) {
        return;
    }

    // Compute dot product Q[i] · K[j]
    var dot_product: f32 = 0.0;
    for (var k = 0u; k < d_k; k = k + 1u) {
        let q_val = query[i * d_k + k];
        let k_val = key[j * d_k + k];
        dot_product = dot_product + q_val * k_val;
    }

    // Scale by 1 / sqrt(d_k)
    let scale = 1.0 / sqrt(f32(d_k));
    let score = dot_product * scale;

    // Write to output (will be used as input for softmax)
    output[i * seq_len + j] = score;
}

// =============================================================================
// CAUSAL MASK: Apply mask for autoregressive generation
// =============================================================================
//
// Prevents attending to future positions.
// Sets scores[i, j] = -inf for j > i

@compute @workgroup_size(16, 16, 1)
fn apply_causal_mask(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let seq_len = metadata[0];

    let i = global_id.y;  // Row
    let j = global_id.x;  // Column

    if (i >= seq_len || j >= seq_len) {
        return;
    }

    // If j > i, this is a future token, so mask it
    if (j > i) {
        output[i * seq_len + j] = -3.402823466e+38;  // -FLT_MAX
    }
}

// =============================================================================
// SOFTMAX (Row-wise)
// =============================================================================
//
// Apply softmax to each row of the attention score matrix.
// Each row becomes a probability distribution over keys.

@compute @workgroup_size(256, 1, 1)
fn softmax_row(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let seq_len = metadata[0];
    let row = global_id.y;

    if (row >= seq_len) {
        return;
    }

    let col = local_id.x;

    // Load value (or -inf if out of bounds)
    var local_val: f32 = -3.402823466e+38;
    if (col < seq_len) {
        local_val = output[row * seq_len + col];
    }

    shared_scores[col] = local_val;
    workgroupBarrier();

    // Find max value in row (for numerical stability)
    var stride = 128u;
    while (stride > 0u) {
        if (col < stride && col + stride < 256u) {
            shared_scores[col] = max(shared_scores[col], shared_scores[col + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let max_val = shared_scores[0];
    workgroupBarrier();

    // Compute exp(x - max) and sum
    var local_exp: f32 = 0.0;
    if (col < seq_len) {
        local_exp = exp(output[row * seq_len + col] - max_val);
    }

    shared_scores[col] = local_exp;
    workgroupBarrier();

    // Sum all exp values
    stride = 128u;
    while (stride > 0u) {
        if (col < stride && col + stride < 256u) {
            shared_scores[col] = shared_scores[col] + shared_scores[col + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let sum = shared_scores[0];

    // Normalize
    if (col < seq_len) {
        output[row * seq_len + col] = local_exp / sum;
    }
}

// =============================================================================
// APPLY ATTENTION: Compute weighted sum output = A @ V
// =============================================================================
//
// Each thread computes one element of the output.
// output[i, k] = sum_j (attention_weights[i, j] * value[j, k])

@compute @workgroup_size(16, 16, 1)
fn apply_attention_weights(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let seq_len = metadata[0];
    let d_v = metadata[2];

    let i = global_id.y;  // Sequence position
    let k = global_id.x;  // Value dimension

    if (i >= seq_len || k >= d_v) {
        return;
    }

    // Compute weighted sum over all keys
    var sum: f32 = 0.0;
    for (var j = 0u; j < seq_len; j = j + 1u) {
        let attention_weight = output[i * seq_len + j];  // A[i, j]
        let value_elem = value[j * d_v + k];            // V[j, k]
        sum = sum + attention_weight * value_elem;
    }

    // Write to output (reuse same buffer after softmax)
    // Note: This requires careful sequencing - softmax must complete first
    output[i * d_v + k] = sum;
}

// =============================================================================
// FUSED SINGLE-HEAD ATTENTION
// =============================================================================
//
// Combines all steps in a single kernel for small sequences.
// More efficient than multiple kernel launches for seq_len <= 256.

@compute @workgroup_size(256, 1, 1)
fn single_head_attention_fused(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let seq_len = metadata[0];
    let d_k = metadata[1];
    let d_v = metadata[2];
    let use_causal_mask = metadata[3];  // 1 = causal, 0 = bidirectional

    let query_idx = workgroup_id.x;  // Which query position we're computing
    let thread_idx = local_id.x;

    if (query_idx >= seq_len) {
        return;
    }

    // Step 1: Compute attention scores for this query
    var score: f32 = -3.402823466e+38;

    if (thread_idx < seq_len) {
        // Compute Q[query_idx] · K[thread_idx]
        var dot: f32 = 0.0;
        for (var k = 0u; k < d_k; k = k + 1u) {
            let q = query[query_idx * d_k + k];
            let key_val = key[thread_idx * d_k + k];
            dot = dot + q * key_val;
        }

        // Scale
        let scale = 1.0 / sqrt(f32(d_k));
        score = dot * scale;

        // Apply causal mask if needed
        if (use_causal_mask == 1u && thread_idx > query_idx) {
            score = -3.402823466e+38;
        }
    }

    shared_scores[thread_idx] = score;
    workgroupBarrier();

    // Step 2: Softmax over scores (same as before)
    // Find max
    var stride = 128u;
    while (stride > 0u) {
        if (thread_idx < stride) {
            shared_scores[thread_idx] = max(shared_scores[thread_idx], shared_scores[thread_idx + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let max_score = shared_scores[0];
    workgroupBarrier();

    // Compute exp and sum
    var exp_score: f32 = 0.0;
    if (thread_idx < seq_len) {
        exp_score = exp(shared_scores[thread_idx] - max_score);
    }

    shared_scores[thread_idx] = exp_score;
    workgroupBarrier();

    stride = 128u;
    while (stride > 0u) {
        if (thread_idx < stride) {
            shared_scores[thread_idx] = shared_scores[thread_idx] + shared_scores[thread_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let sum = shared_scores[0];
    workgroupBarrier();

    // Normalize
    let attention_weight = exp_score / sum;
    shared_scores[thread_idx] = attention_weight;
    workgroupBarrier();

    // Step 3: Apply attention to values
    // Each thread computes one dimension of the output
    if (thread_idx < d_v) {
        var weighted_sum: f32 = 0.0;

        for (var j = 0u; j < seq_len; j = j + 1u) {
            let weight = shared_scores[j];
            let val = value[j * d_v + thread_idx];
            weighted_sum = weighted_sum + weight * val;
        }

        output[query_idx * d_v + thread_idx] = weighted_sum;
    }
}
