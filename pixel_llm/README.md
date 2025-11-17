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
- [X] PixelFS: Store multi-GB files as pixel sequences
- [X] Infinite Map: 2D spatial indexing system
- [X] PXI-LLM format: Specification for pixel-encoded models
- [X] Coaching System: Orchestrator for development
- [X] Test Phase 1 components
- [X] Commit and push Phase 1 implementation
