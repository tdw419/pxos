# PXI-LLM Format Specification v1.0

**Pixel-Native LLM Storage Format**

This document defines how to store Large Language Model weights, embeddings, and activations as pixel data for GPU-native processing.

## Overview

Traditional LLMs store weights as floating-point tensors in linear memory. **PXI-LLM** stores weights as **RGB pixel values** in a 2D spatial layout, enabling:

- **GPU-native access**: Weights are already in texture format
- **Spatial relationships**: Related weights are spatially close
- **Visual inspection**: Can literally "see" the model
- **Memory-mapped efficiency**: Load only needed regions
- **Hardware acceleration**: Leverage texture caching

## Core Concepts

### Weight Encoding

Weights are typically fp32, fp16, or int8. PXI-LLM uses RGB channels to store these values.

- **fp16 weights**: One fp16 value (16 bits) can be stored in the R and G channels of a pixel (8 bits each).
- **int8 weights**: One int8 value can be stored in a single channel (e.g., R). A single RGB pixel can store three int8 weights.

### PXI-LLM File Header

Each PXI-LLM file begins with a header that describes the model architecture and layout. This is a JSON blob for flexibility.

```json
{
  "version": "1.0",
  "model_name": "Qwen2.5-7B-Instruct",
  "tensor_type": "fp16",
  "vocab_size": 151936,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_layers": 32,
  "tensors": {
    "transformer.wte.weight": {
      "file": "wte.pxi",
      "shape": [151936, 4096],
      "offset": 0
    },
    "layers.0.attention.wq.weight": {
      "file": "layer0.pxi",
      "shape": [4096, 4096],
      "offset": 0
    }
  }
}
```

This header allows the GPU inference engine to locate and correctly interpret the pixel data for each tensor.
