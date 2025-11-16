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

## Next Kernels (Roadmap)

### `matmul.wgsl` (Phase 1.2)
**Purpose**: Matrix multiplication for neural layers
**Features**:
- Tiled computation for memory efficiency
- Shared memory optimization
- FP16 support for memory bandwidth

### `attention.wgsl` (Phase 2)
**Purpose**: Multi-head attention mechanism
**Features**:
- Spatial neighborhood operations
- Parallel head computation
- Efficient softmax

### `layer_norm.wgsl` (Phase 2)
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
- [ ] Phase 1.2: Matrix multiplication
- [ ] Phase 1.3: Activation functions
- [ ] Phase 2: Attention mechanisms
- [ ] Phase 3: Full inference pipeline

**Current milestone**: Foundation complete, ready for matrix multiplication.
