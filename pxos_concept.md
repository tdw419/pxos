# pxOS Concept (Living Document)

## Vision

pxOS is a GPU-native, pixel-driven operating system where **pixels are the instruction and data space**. All computation, storage, and I/O are strictly mapped to visual structures in VRAM.

The core insight: **Computation can be pixels**. Not "represented as" pixels, not "compiled to" pixels, but natively encoded and executed as RGBA images.

## Current Focus (v0.1.0)

**Status**: Complete neural network primitives implemented and validated.

**Achievement**: Full Pixel-LLM forward pass (2-layer MLP) encoded as a single `.pxi` program and successfully executed.

**Proof Point**:
- Program: 1024×161 RGBA PNG (205KB)
- Weights: W_hidden (128×128), W_out (128×1024), biases
- Instructions: 6 opcodes (MATMUL, ADD, RELU, MATMUL, ADD, HALT)
- Execution: Produces valid logits [0, 210] ✅

**Critical Breakthrough**: Dynamic row allocation utilities exposed uint8 addressing constraints, preventing silent data corruption.

## Architecture Layers

### Layer 0: Pixel Protocol v1.0.0
- **Format**: RGBA PNG (.pxi files)
- **Immutability**: Same file runs on CPU/GPU without transformation
- **Encoding**: Row 0 = instructions, Rows 1+ = data
- **Stability**: No breaking changes allowed without major version bump

### Layer 1: pxVM Opcodes (v0.0.3)
- `OP_HALT` (0) - Stop execution
- `OP_DOT_RGB` (1) - Integer dot product
- `OP_ADD` (2) - Element-wise addition (uint8 clamped)
- `OP_RELU` (3) - In-place ReLU activation
- `OP_MATMUL` (4) - Matrix multiplication

**Status**: CPU interpreter complete, GPU executor stubbed (WGSL ready, wgpu-py not installed)

### Layer 2: Neural Network Primitives
- Matrix encoding: Header pixel + row-major data
- Quantization: float32 → uint8 linear mapping
- In-place mutation: Programs modify pixels directly

**Status**: Working end-to-end for Pixel-LLM forward pass

### Layer 3: Utilities (v0.1.0)
- `pxvm.utils.layout` - Image size calculation, row allocation, matrix I/O
- `pxvm.utils.validation` - Program structure verification
- **Philosophy**: Tools work WITH pixels, not AROUND them

**Status**: Complete, proven valuable in practice

## Constraints

1. **Pixel Protocol**: The `.pxi` file format must remain executor-agnostic and stable
2. **Simplicity**: All components must be explainable to another AI in under 2 pages of text
3. **No Abstraction**: Tools should work with pixels, not hide them
4. **Byte Parity**: Same input → same output across CPU/GPU/ASIC

## Open Questions

### Immediate (v0.1.x)
- How do we validate that pxVM output matches numpy reference implementation?
- What is the accuracy cost of uint8 quantization for Pixel-LLM?
- Can we prove GPU execution produces byte-identical results?

### Near-term (v0.2.x)
- How do we support float16/float32 without breaking uint8 compatibility?
- What's the minimal opcode set for autoregressive text generation?
- How do we encode the tokenizer as pixels?

### Long-term (v1.0+)
- How should pxOS represent multi-process state in a single VRAM surface?
- What is the minimal instruction set for self-modification (rewriting row 0)?
- How do we define the first filesystem structure (pixel-encoded FAT table)?
- Can the OS kernel itself be a .pxi program?

## Next Leap: Acceleration

**Current Bottleneck**: `OP_MATMUL` uses sequential triple-loop (naive CPU/GPU)

**Target**: Parallel tiled kernel on GPU
- One thread per output element C[m,n]
- Shared memory tiling for cache efficiency
- 100x+ speedup expected

**Dependency**: Validate CPU accuracy before optimizing GPU

## Evolution History

- **v0.0.1** (2024-11-16): First pixel program (OP_DOT_RGB)
- **v0.0.2** (2024-11-16): Neural primitives (OP_ADD, OP_RELU)
- **v0.0.3** (2024-11-17): Complete neural toolkit (OP_MATMUL)
- **v0.1.0** (2024-11-17): Pixel-LLM integration + utilities
  - Fixed: Silent data corruption bug via dynamic row allocation
  - Proven: Production neural networks execute natively as pixels

## Philosophy

**Core Thesis**: If computation IS pixels (not just represented as), then:
- Debugging becomes visual inspection
- Storage is lossless PNG compression
- GPU upload is zero-copy texture load
- Cross-platform portability is guaranteed by PNG spec

**Anti-Pattern**: Creating abstraction layers that hide pixels
- No `PixelMatrix` classes
- No "compilation" step
- No separate I/O layer

**Pattern**: Creating utilities that operate ON pixels
- `write_matrix()` writes TO pixel arrays
- `allocate_rows()` calculates WHERE in pixel space
- All functions take/return `np.ndarray` directly

## Success Criteria (v1.0)

- [ ] Generate coherent text from Pixel-LLM .pxi program
- [ ] Prove GPU execution matches CPU bit-for-bit
- [ ] Demonstrate 100x+ speedup on GPU vs CPU
- [ ] Show .pxi program runs unchanged on 3+ platforms
- [ ] Encode a complete small LLM (e.g., GPT-2 Small) as pixels

---

**Last Updated**: 2024-11-17
**Current Version**: v0.1.0
**Next Milestone**: GPU validation + text generation
