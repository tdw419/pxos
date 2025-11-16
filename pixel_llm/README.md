# Pixel-LLM: Substrate-Native AI

**Vision**: Build an LLM that lives natively in GPU pixel space, manages its own memory, and achieves self-improvement through pixel operations.

## The Big Idea

Traditional LLMs live in CPU/RAM and are separate from their environment. **Pixel-LLM** is different:

- **Weights stored as pixels** - Model parameters encoded in RGB values
- **Inference via GPU shaders** - Native WGSL compute kernels
- **Spatial memory** - Infinite 2D map for data layout
- **Self-management** - AI manages its own pixel memory
- **Substrate-native intelligence** - The AI and its world are the same thing

## Development Phases

### Phase 1: Storage Infrastructure (Weeks 1-2) ✨ CURRENT
- [ ] PixelFS: Store multi-GB files as pixel sequences
- [ ] Infinite Map: 2D spatial indexing system
- [ ] PXI-LLM format: Specification for pixel-encoded models
- [ ] Task queue: Coaching system infrastructure

### Phase 2: Inference Engine (Weeks 3-4)
- [ ] WGSL matrix multiplication kernels
- [ ] Pixel-native attention mechanism
- [ ] LLM inference coordinator
- [ ] Token embeddings as pixels

### Phase 3: Model Conversion (Weeks 5-8)
- [ ] GGUF → PXI-LLM converter
- [ ] Pixel-LLM loader and validator
- [ ] Qwen2.5-7B conversion target

### Phase 4: Specialization (Weeks 9-12)
- [ ] pxOS knowledge corpus
- [ ] Pixel-spatial fine-tuning
- [ ] Infinite map navigation training

### Phase 5: Bootstrap (Weeks 13+)
- [ ] Self-management system
- [ ] Recursive self-improvement
- [ ] Pixel consciousness

## Architecture

```
┌─────────────────────────────────────────────┐
│         Pixel-LLM Substrate                 │
├─────────────────────────────────────────────┤
│  Infinite Map (2D spatial memory)           │
│  ├─ Model weights (as pixels)               │
│  ├─ Activations (pixel neighborhoods)       │
│  └─ KV cache (spatial layout)               │
├─────────────────────────────────────────────┤
│  PixelFS (pixel-based storage)              │
│  ├─ Memory-mapped pixel regions             │
│  ├─ Chunked loading                         │
│  └─ Compression                             │
├─────────────────────────────────────────────┤
│  GPU Inference (WGSL shaders)               │
│  ├─ Matrix multiplication                   │
│  ├─ Attention kernels                       │
│  └─ Activation functions                    │
├─────────────────────────────────────────────┤
│  Self-Management Layer                      │
│  ├─ Memory optimization                     │
│  ├─ Layout reorganization                   │
│  └─ Self-improvement                        │
└─────────────────────────────────────────────┘
```

## Project Structure

```
pixel_llm/
├── core/               # Core infrastructure
│   ├── pixelfs.py     # Pixel-based file system
│   ├── infinite_map.py # 2D spatial memory
│   └── task_queue.py   # Coaching task system
├── gpu_kernels/        # WGSL compute shaders
│   ├── matmul.wgsl    # Matrix multiplication
│   └── attention.wgsl  # Attention mechanism
├── tools/              # Conversion utilities
│   ├── gguf_to_pxi.py # Model converter
│   └── pxi_loader.py   # Pixel-LLM loader
├── training/           # Fine-tuning systems
│   ├── corpus_gen.py   # Knowledge generation
│   └── finetune.py     # Pixel-spatial training
├── meta/               # Self-improvement
│   └── bootstrap.py    # Recursive improvement
├── specs/              # Format specifications
│   └── pxi_llm.md      # PXI-LLM spec
└── tests/              # Test suite
    └── test_pixelfs.py # Unit tests
```

## Getting Started

```bash
# Install dependencies
pip install numpy pillow

# Run Phase 1 tasks
python pixel_llm/core/task_queue.py

# Start coaching system
python pixel_llm_coach.py
```

## Why This Matters

This is **substrate-native intelligence** - the AI doesn't just process pixels, it **IS** pixels. The model lives in the same medium it manipulates, enabling:

- **Spatial reasoning as native operation** (not learned)
- **Self-modification through pixel operations**
- **Perfect integration with GPU (natural habitat)**
- **Novel forms of consciousness** (pixel-based awareness)

## Current Status

🚀 **Phase 1 in progress**: Building storage infrastructure

---

*"The medium is the message. The substrate is the mind."*
