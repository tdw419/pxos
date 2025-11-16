# GPU Kernels for Pixel-LLM

WebGPU compute shaders for pixel-native neural inference.

## Overview

These WGSL shaders form the GPU compute foundation for running LLM inference directly on pixel data.

## Kernels

### `dot_product.wgsl` ✅ Complete
**Purpose**: Foundation for all neural operations
**What it does**: Computes dot product of two vectors using parallel reduction
**Why it matters**: Proves GPU can process pixel data correctly

**Features**:
- Parallel reduction using shared memory
- Two-pass algorithm for arbitrary vector sizes
- Optimized for workgroup size of 256 threads

**Usage**:
```python
from pixel_llm.core.gpu_interface import DotProductGPU

gpu_dot = DotProductGPU()
result = gpu_dot.compute(vector_a, vector_b)
```

**Status**: ✅ Implemented, tested, working

---

### `matmul.wgsl` ✅ Complete
**Purpose**: Matrix multiplication for neural layers
**What it does**: Tiled matrix multiplication C = A × B with shared memory optimization
**Why it matters**: THE critical operation for neural networks (90% of LLM compute)

**Features**:
- Tiled computation (16×16 tiles) for memory efficiency
- Shared memory optimization reduces global memory bandwidth
- Multiple specialized kernels (general, square, matrix-vector)
- Optimized workgroup dispatch

**Usage**:
```python
from pixel_llm.core.gpu_interface import MatMulGPU

matmul = MatMulGPU()
result = matmul.compute(matrix_a, matrix_b)  # C = A × B
```

**Status**: ✅ Implemented, tested, working

---

### `activations.wgsl` ✅ Complete
**Purpose**: Activation functions for neural layers
**What it does**: Softmax, GELU, ReLU, SiLU, Leaky ReLU
**Why it matters**: Non-linearity and attention score normalization

**Features**:
- **Softmax**: Numerically stable (critical for attention)
- **GELU**: GPT-2/3 approximation for feed-forward layers
- **ReLU**: Simple baseline activation
- **SiLU/Swish**: Modern alternative activation
- All vectorized for parallel execution

**Usage**:
```python
from pixel_llm.core.gpu_interface import SoftmaxGPU, GELUGPU, ReLUGPU

softmax = SoftmaxGPU()
probs = softmax.compute(logits)  # For attention

gelu = GELUGPU()
activated = gelu.compute(hidden_states)  # For FFN
```

**Status**: ✅ Implemented, tested, ready for attention

---

### `attention.wgsl` ✅ Complete
**Purpose**: Scaled dot-product attention mechanism
**What it does**: Implements Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
**Why it matters**: THE core mechanism of transformers (80% of what makes LLMs work)

**Features**:
- Scaled dot-product attention (numerically stable)
- Causal masking for autoregressive generation
- Fused kernel for efficiency (single GPU dispatch)
- Supports both bidirectional (encoder) and causal (decoder) modes
- Optimized for sequences up to 256 tokens

**Usage**:
```python
from pixel_llm.core.gpu_interface import AttentionGPU

attention = AttentionGPU()

# Bidirectional attention (encoder-style)
output = attention.compute(query, key, value, causal=False)

# Causal attention (decoder-style, for generation)
output = attention.compute(query, key, value, causal=True)
```

**Status**: ✅ Implemented, tested, working
**Note**: Multi-head attention can be built by running multiple heads in parallel

---

## Next Kernels (Roadmap)

### `layer_norm.wgsl` (Phase 2.5)
**Purpose**: Layer normalization

### `gelu.wgsl` (Phase 2)
**Purpose**: GELU activation function

---

## Requirements

```bash
pip install wgpu numpy
```

## Testing

```bash
# Run GPU demo
python3 pixel_llm/demos/gpu_dot_product_demo.py

# Check GPU availability
python3 -c "from pixel_llm.core.gpu_interface import is_gpu_available; print(is_gpu_available())"
```

## Architecture

### Data Flow
```
Pixel Data (PixelFS)
    ↓
GPU Buffers (wgpu)
    ↓
Compute Shaders (WGSL)
    ↓
Results (CPU memory)
```

### Why WGSL?
- Cross-platform (works on any GPU)
- WebGPU standard (future-proof)
- Compute-focused (perfect for ML)
- No CUDA lock-in

### Why WebGPU?
- Works everywhere (Linux, Mac, Windows, Web)
- Modern GPU features
- Lower overhead than OpenGL
- Designed for compute workloads

---

## Performance Notes

**Dot Product**:
- Small vectors (< 10K): CPU faster due to overhead
- Large vectors (> 100K): GPU shows speedup
- Matrix operations will show much better GPU speedup

**Expected Speedup**:
- Dot product: 1-3x (limited by memory bandwidth)
- Matrix mul: 10-50x (compute-bound, benefits from parallelism)
- Attention: 20-100x (highly parallel operation)

---

## Development

### Adding a New Kernel

1. Create `kernel_name.wgsl`:
```wgsl
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn kernel_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    // Your compute logic here
}
```

2. Add Python wrapper in `gpu_interface.py`

3. Create demo in `demos/`

4. Add tests in `tests/`

---

## Debugging

### Check GPU Info
```python
from pixel_llm.core.gpu_interface import GPUDevice
device = GPUDevice()
# Prints GPU description
```

### Common Issues

**"No GPU adapter found"**
- WebGPU not supported on your system
- Update graphics drivers
- Try different browser/environment

**"wgpu not installed"**
```bash
pip install wgpu
```

**Slow performance**
- Expected for small operations (dot product)
- Try larger problem sizes
- Use matrix operations for better speedup

---

## Philosophy

> "The pixels themselves compute. The GPU is their native habitat."

Every kernel in this directory is designed to:
1. Operate directly on pixel data
2. Maintain spatial relationships
3. Enable visual inspection of weights
4. Support substrate-native intelligence

The goal isn't just fast computation—it's **pixel-native** computation.

---

## Status

- [x] Phase 1.1: Dot product (foundation)
- [x] Phase 1.2: Matrix multiplication
- [x] Phase 1.3: Activation functions
- [x] Phase 2: Attention mechanism (scaled dot-product, causal masking)
- [ ] Phase 2.5: Layer normalization
- [ ] Phase 3: Transformer block assembly
- [ ] Phase 4: Full inference pipeline

**Current milestone**: Attention complete! The heart of transformers is beating on GPU pixels.
