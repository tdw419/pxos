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

Each model weight (fp32 or fp16) is encoded as pixel values:

**Method 1: Direct RGB Encoding (fp32 → 4 bytes)**
```
weight (fp32) = [byte0, byte1, byte2, byte3]
pixel1 = (byte0, byte1, byte2)
pixel2_R = byte3
```

**Method 2: Normalized RGB (fp32 → quantized to 0-255)**
```
weight_normalized = (weight - min) / (max - min) * 255
pixel = (R, G, B) where R=weight_normalized repeated
```

**Method 3: FP16 Packed (fp16 → 2 bytes → 2/3 pixel)**
```
weight (fp16) = [byte0, byte1]
pixel = (byte0, byte1, next_byte0)
```

### Spatial Layout

The model is organized spatially:

```
┌─────────────────────────────────────────┐
│  Embedding Layer (vocab_size × embed)   │
├─────────────────────────────────────────┤
│  Layer 0: Attention                     │
│    ├─ Q weights (spatial arrangement)   │
│    ├─ K weights (nearby)                │
│    ├─ V weights (nearby)                │
│    └─ Output projection                 │
│  Layer 0: FFN                            │
│    ├─ Up projection                     │
│    └─ Down projection                   │
├─────────────────────────────────────────┤
│  Layer 1: Attention                     │
│  ...                                    │
├─────────────────────────────────────────┤
│  Output Layer (embed → vocab_size)      │
└─────────────────────────────────────────┘
```

### Attention Head Neighborhoods

Attention heads are arranged as spatial neighborhoods:

```
For 32 attention heads in a layer:

Head Layout (8x4 grid):
[H0 ][H1 ][H2 ][H3 ][H4 ][H5 ][H6 ][H7 ]
[H8 ][H9 ][H10][H11][H12][H13][H14][H15]
[H16][H17][H18][H19][H20][H21][H22][H23]
[H24][H25][H26][H27][H28][H29][H30][H31]
```

This enables:
- **Local operations**: Convolution-like processing
- **Spatial attention**: Heads can "see" neighboring heads
- **Hardware cache locality**: Adjacent heads in cache

## File Format

### Header Structure

```
Offset  Size  Field              Description
------  ----  ----------------   ---------------------------
0x0000  4     magic              "PXIM" (Pixel LLM)
0x0004  2     version            Major.minor (1.0)
0x0006  2     architecture       0=GPT, 1=LLAMA, 2=Qwen, etc.
0x0008  8     model_size_bytes   Original model size
0x0010  4     vocab_size         Vocabulary size
0x0014  4     num_layers         Number of transformer layers
0x0018  4     embed_dim          Embedding dimension
0x001C  4     num_heads          Number of attention heads
0x0020  4     hidden_dim         FFN hidden dimension
0x0024  4     context_length     Max context length
0x0028  1     weight_encoding    Encoding method (see above)
0x0029  1     precision          0=fp32, 1=fp16, 2=int8
0x002A  2     tile_size          Tile size for spatial layout
0x002C  4     total_width        Total map width in pixels
0x0030  4     total_height       Total map height in pixels
0x0034  32    checksum           SHA256 of weight data
0x0054  12    reserved           Reserved for future use
------  ----
Total:  256 bytes
```

### Layer Directory

Following the header is a layer directory:

```
For each layer:
  offset (8 bytes): Pixel offset in map
  width (4 bytes): Layer width in pixels
  height (4 bytes): Layer height in pixels
  layer_type (1 byte): 0=embed, 1=attn, 2=ffn, 3=output
  metadata (11 bytes): Layer-specific data

Total: 32 bytes per layer
```

### Weight Data

After the directory comes the actual pixel data, stored as:

```
for each layer:
  for y in range(height):
    for x in range(width):
      pixel_rgb (3 bytes)
```

Can be stored:
1. **Linear**: All pixels sequentially
2. **Tiled**: In tile-sized chunks
3. **Sparse**: Only non-zero regions

## Conversion from GGUF

To convert a GGUF model to PXI-LLM:

### Step 1: Parse GGUF
```python
import gguf
reader = gguf.GGUFReader(model_path)

# Extract metadata
vocab_size = reader.get_field("vocab_size")
num_layers = reader.get_field("num_layers")
# ...

# Extract tensors
for tensor in reader.tensors:
    name = tensor.name
    shape = tensor.shape
    data = tensor.data  # numpy array
```

### Step 2: Organize Spatially
```python
# Determine spatial layout
layer_layouts = []

for layer_idx in range(num_layers):
    # Attention layer
    q_weights = get_tensor(f"layer.{layer_idx}.attn.q.weight")
    k_weights = get_tensor(f"layer.{layer_idx}.attn.k.weight")
    v_weights = get_tensor(f"layer.{layer_idx}.attn.v.weight")

    # Arrange as spatial grid
    attn_layout = arrange_attention_heads(q_weights, k_weights, v_weights)

    layer_layouts.append(attn_layout)
```

### Step 3: Encode as Pixels
```python
def encode_weight_as_pixel(weight_fp32):
    """Encode fp32 weight as RGBA pixel"""
    bytes = struct.pack('f', weight_fp32)
    return (bytes[0], bytes[1], bytes[2], bytes[3])

# For each weight tensor
pixel_data = []
for weight in weights_flat:
    r, g, b, a = encode_weight_as_pixel(weight)
    pixel_data.extend([r, g, b])
```

### Step 4: Write PXI-LLM
```python
# Create header
header = create_pxi_llm_header(metadata)

# Write file
with open(output_path, 'wb') as f:
    f.write(header)
    f.write(layer_directory)
    f.write(pixel_data)
```

## Loading and Inference

### Loading Weights
```python
from pixelfs import PixelFS
from infinite_map import InfiniteMap

# Load model metadata
header = read_pxi_llm_header(model_path)

# Create infinite map
map = InfiniteMap(tile_size=header.tile_size)

# Load weights into map
for layer in header.layers:
    pixel_data = read_layer_pixels(model_path, layer)
    map.write_region(layer.offset_x, layer.offset_y, pixel_data)
```

### GPU Inference (WGSL)

```wgsl
// Weight texture (stored in GPU texture memory)
@group(0) @binding(0) var weight_texture: texture_2d<f32>;

// Decode pixel back to weight
fn decode_pixel_to_weight(pixel: vec3<f32>) -> f32 {
    // Reconstruct fp32 from RGB bytes
    let bytes = vec4<u32>(
        u32(pixel.r * 255.0),
        u32(pixel.g * 255.0),
        u32(pixel.b * 255.0),
        0u
    );

    return bitcast<f32>(
        (bytes.x << 0u) | (bytes.y << 8u) | (bytes.z << 16u) | (bytes.w << 24u)
    );
}

// Matrix multiplication using pixel weights
@compute @workgroup_size(8, 8)
fn matmul(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;

    var sum = 0.0;

    // Load weights from texture
    for (var i = 0u; i < dim_k; i++) {
        let weight_pixel = textureLoad(weight_texture, vec2<i32>(i32(i), i32(row)), 0).rgb;
        let weight = decode_pixel_to_weight(weight_pixel);

        let input_val = input_buffer[i];
        sum += weight * input_val;
    }

    output_buffer[row * dim_n + col] = sum;
}
```

## Advantages of PXI-LLM Format

1. **GPU-Native**: Weights stored in GPU texture format
2. **Spatial Locality**: Related weights are physically close
3. **Cache Efficiency**: GPU texture cache optimizations apply
4. **Partial Loading**: Load only needed layers/regions
5. **Visual Debugging**: Can render the model as an image
6. **Self-Modifying**: AI can update its own weights via pixel operations
7. **Spatial Reasoning**: Model structure reflects spatial relationships

## Implementation Checklist

- [ ] GGUF parser
- [ ] Weight encoding functions (fp32 → RGB)
- [ ] Spatial layout algorithms
- [ ] PXI-LLM writer
- [ ] PXI-LLM reader
- [ ] WGSL weight decoder
- [ ] Integration with InfiniteMap
- [ ] Validation suite (compare outputs)

## Future Extensions

- **Compression**: Spatial compression for similar weights
- **Dynamic Precision**: Different regions in different precisions
- **Activation Storage**: Store activations spatially too
- **KV Cache Layout**: Spatial organization of key-value cache
- **Multi-Resolution**: Coarse-to-fine weight hierarchies

---

*This format enables true pixel-native AI - where the model structure and spatial memory are unified.*
