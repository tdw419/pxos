# Pixel-Only Inference - Pure Pixel AI Execution

**Vision**: Run Pixel-LLM inference where input, weights, computation, and output are all pixels. No Python arrays, no numpy - just pixel textures and pixel programs.

---

## Current State vs. Vision

### Current (Pixel-Stored)
```
Input (text) → Python tokenizer → numpy array →
    Load weights from pixels → numpy matmul →
    numpy array → Python detokenizer → Output (text)
```

**Pixel parts**: Weights storage only
**Non-pixel parts**: Input, computation, output

### Vision (Pixel-Native)
```
Input (pixels) → Pixel program → Pixel computation → Output (pixels)
```

**Pixel parts**: Everything
**Non-pixel parts**: Tiny launcher only

---

## Architecture: Three Pixel Buffers

### 1. Weight Texture (Already Have ✅)
**File**: `pixellm_v0.pxi` (306×305 RGB)

**Layout**:
```
Pixel[0..N]: Flattened weight bytes
- b_hidden: 128 bytes
- b_out: 1024 bytes
- embed: 131,072 bytes (1024×128 matrix)
- hidden: 16,384 bytes (128×128 matrix)
- out: 131,072 bytes (128×1024 matrix)
```

**Access**: Read-only texture in GPU/CPU

### 2. Token Texture (Need to Build)
**File**: `input_tokens.pxi` (1×N RGB or grayscale)

**Layout**:
```
Pixel[0]: Token 0 (R=token_id, G/B=unused or metadata)
Pixel[1]: Token 1
...
Pixel[N-1]: Token N-1
```

**For vocab=1024**, we need 10 bits per token:
- **Simple**: R=token_id % 256, G=token_id // 256, B=unused
- **Or**: Just R channel if vocab ≤ 256

**Access**: Read-only input to inference

### 3. Activation Textures (Need to Build)
**Intermediate buffers** (can be GPU-only, never touch CPU):

```
T_embed:  1×128 image (pooled embeddings)
T_hidden: 1×128 image (hidden layer activations)
T_logits: 1×1024 or 32×32 image (output logits)
```

**Output**:
```
output_token.pxi: 1×1 image
  R = predicted_token_id % 256
  G = predicted_token_id // 256
  B = confidence (0-255)
```

---

## Execution Models

### Model A: Shader-Based (GPU-Native)

**Components**:
1. `pixellm_weights.pxi` - Weight texture (already have)
2. `pixellm_forward.wgsl` - Compute shader
3. `input_tokens.pxi` - Input texture
4. `output_token.pxi` - Output texture

**Execution**:
```wgsl
// Shader Pass 1: Embedding + Pooling
@compute @workgroup_size(128)
fn embed_and_pool(
    @binding(0) weights: texture_2d<f32>,
    @binding(1) tokens: texture_2d<u32>,
    @binding(2) embed_out: texture_storage_2d<rgba32float, write>
) {
    let d = workgroup_id.x; // Embedding dimension

    var sum = 0.0;
    let n_tokens = textureLoad(tokens, vec2(0, 0), 0).r;

    for (var i = 0u; i < n_tokens; i++) {
        let token = textureLoad(tokens, vec2(i + 1, 0), 0).r;
        let weight = load_weight(weights, EMBED_OFFSET + token * 128 + d);
        sum += weight;
    }

    let pooled = sum / f32(n_tokens);
    textureStore(embed_out, vec2(d, 0), vec4(pooled, 0.0, 0.0, 0.0));
}

// Shader Pass 2: Hidden Layer
@compute @workgroup_size(128)
fn hidden_layer(
    @binding(0) weights: texture_2d<f32>,
    @binding(1) embed: texture_2d<f32>,
    @binding(2) hidden_out: texture_storage_2d<rgba32float, write>
) {
    let h = workgroup_id.x;

    var sum = 0.0;
    for (var d = 0u; d < 128u; d++) {
        let emb_val = textureLoad(embed, vec2(d, 0), 0).r;
        let weight = load_weight(weights, HIDDEN_OFFSET + d * 128 + h);
        sum += emb_val * weight;
    }

    let bias = load_weight(weights, BHIDDEN_OFFSET + h);
    let relu = max(0.0, sum + bias);

    textureStore(hidden_out, vec2(h, 0), vec4(relu, 0.0, 0.0, 0.0));
}

// Shader Pass 3: Output Logits
@compute @workgroup_size(256)
fn output_logits(
    @binding(0) weights: texture_2d<f32>,
    @binding(1) hidden: texture_2d<f32>,
    @binding(2) logits_out: texture_storage_2d<rgba32float, write>
) {
    let k = workgroup_id.x * 256 + local_invocation_id.x; // Vocab index

    if (k >= 1024) { return; }

    var sum = 0.0;
    for (var h = 0u; h < 128u; h++) {
        let hid_val = textureLoad(hidden, vec2(h, 0), 0).r;
        let weight = load_weight(weights, OUT_OFFSET + h * 1024 + k);
        sum += hid_val * weight;
    }

    let bias = load_weight(weights, BOUT_OFFSET + k);
    let logit = sum + bias;

    textureStore(logits_out, vec2(k % 32, k / 32), vec4(logit, 0.0, 0.0, 0.0));
}

// Shader Pass 4: Argmax
@compute @workgroup_size(1)
fn find_argmax(
    @binding(0) logits: texture_2d<f32>,
    @binding(1) output: texture_storage_2d<rgba8unorm, write>
) {
    var max_logit = -1e10;
    var max_idx = 0u;

    for (var k = 0u; k < 1024u; k++) {
        let logit = textureLoad(logits, vec2(k % 32, k / 32), 0).r;
        if (logit > max_logit) {
            max_logit = logit;
            max_idx = k;
        }
    }

    // Encode token ID as pixel color
    let r = u8(max_idx % 256);
    let g = u8(max_idx / 256);
    let b = u8(clamp(max_logit * 10.0 + 128.0, 0.0, 255.0));

    textureStore(output, vec2(0, 0), vec4(f32(r)/255.0, f32(g)/255.0, f32(b)/255.0, 1.0));
}
```

**From Python/pxOS perspective**:
```python
# Tiny launcher (this is the ONLY Python needed)
gpu.load_texture("weights", "pixellm_weights.pxi")
gpu.load_texture("tokens", "input_tokens.pxi")
gpu.run_shader("pixellm_forward.wgsl")
gpu.save_texture("output", "output_token.pxi")

# Result is a pixel!
```

### Model B: pxVM-Based (Pixel CPU)

**Components**:
1. `pixellm_program.pxi` - Pixel opcodes
2. `pixellm_weights.pxi` - Weight texture
3. `input_tokens.pxi` - Input texture
4. `output_token.pxi` - Output texture

**Pixel Opcodes** (example encoding):
```
Pixel opcode format: (R, G, B)
  R = opcode type
  G = operand 1
  B = operand 2

Opcodes:
  0x01: LOAD_WEIGHT_SLICE (addr, length)
  0x02: LOAD_TOKEN (index)
  0x03: DOT_PRODUCT (vec_a_addr, vec_b_addr, output_addr)
  0x04: ADD_BIAS (vec_addr, bias_addr)
  0x05: RELU (vec_addr)
  0x06: ARGMAX (vec_addr, output_addr)
  0x07: STORE_PIXEL (addr, pixel_value)
```

**Example program** (stored as pixels):
```
Pixel[0]: (0x01, EMBED_OFFSET_LO, EMBED_OFFSET_HI)  # Load embedding
Pixel[1]: (0x03, EMBED_ADDR, HIDDEN_WEIGHT_ADDR, HIDDEN_ADDR)  # Dot product
Pixel[2]: (0x04, HIDDEN_ADDR, BHIDDEN_ADDR)  # Add bias
Pixel[3]: (0x05, HIDDEN_ADDR, 0)  # ReLU
Pixel[4]: (0x03, HIDDEN_ADDR, OUT_WEIGHT_ADDR, LOGITS_ADDR)  # Output layer
Pixel[5]: (0x06, LOGITS_ADDR, OUTPUT_TOKEN_ADDR)  # Argmax
Pixel[6]: (0x07, OUTPUT_TOKEN_ADDR, OUTPUT_PIXEL)  # Store result
```

**pxVM Runner**:
```python
class PixelCPU:
    def run_program(self, program_pxi, weights_pxi, input_pxi):
        # Load all as pixel images
        program = load_image(program_pxi)
        weights = load_image(weights_pxi)
        tokens = load_image(input_pxi)

        # Execute pixel opcodes
        pc = 0
        while pc < len(program):
            r, g, b = program[pc]

            if r == 0x01:  # LOAD_WEIGHT_SLICE
                self.load_weight_slice(weights, g, b)
            elif r == 0x03:  # DOT_PRODUCT
                self.dot_product(g, b)
            elif r == 0x05:  # RELU
                self.relu(g)
            elif r == 0x06:  # ARGMAX
                result = self.argmax(g)
                return result  # Pixel value!

            pc += 1
```

**From pxOS perspective**:
```python
# Even tinier launcher
cpu = PixelCPU()
output_pixel = cpu.run_program(
    "pixellm_program.pxi",
    "pixellm_weights.pxi",
    "input_tokens.pxi"
)
save_pixel(output_pixel, "output_token.pxi")
```

---

## Practical Implementation Path

### Phase 1: Pixel I/O (Do This First)

**Goal**: Make input/output pixel-native while keeping numpy computation

**Files to create**:
1. `pixel_llm/core/pixel_tokenizer.py` - Encode text → token texture
2. `pixel_llm/core/pixel_detokenizer.py` - Decode token texture → text
3. `pixel_llm/programs/pixellm_infer_pure.py` - Uses pixel I/O

**Example**:
```python
# Input: text string
text = "pxOS evolution"

# Encode as pixels
token_image = pixel_encode_tokens(text)
save_image(token_image, "input_tokens.pxi")

# Run inference (still numpy for now)
loader = PixelModelLoader("pixellm_v0.pxi", "pixellm_v0.meta.json")
token_ids = pixel_decode_tokens("input_tokens.pxi")
logits = pixellm_forward(token_ids, loader)

# Output as pixels
output_image = pixel_encode_prediction(logits)
save_image(output_image, "output_token.pxi")

# Decode output
predicted_text = pixel_decode_prediction("output_token.pxi")
```

**Progress**: Input and output are pixels ✅
**Still need**: Pixel-native computation

### Phase 2: Shader Computation (GPU Path)

**Goal**: Replace numpy with GPU shaders

**Files to create**:
1. `pixel_llm/gpu/pixellm_forward.wgsl` - Compute shaders
2. `pixel_llm/gpu/gpu_runner.py` - WebGPU/wgpu-py wrapper
3. `pixel_llm/programs/pixellm_infer_gpu.py` - Pure GPU inference

**Progress**: Computation on GPU using textures ✅
**Still need**: pxVM integration

### Phase 3: pxVM Execution (Pixel CPU Path)

**Goal**: Replace shaders with pixel opcodes

**Files to create**:
1. `pixel_llm/pxvm/pixel_opcodes.py` - Opcode definitions
2. `pixel_llm/pxvm/pixel_cpu.py` - pxVM executor
3. `pixel_llm/pxvm/compile_pixellm.py` - Generate pixel program
4. `pixel_llm/programs/pixellm_infer_pxvm.py` - Pure pxVM inference

**Progress**: Entire inference is pixels ✅

### Phase 4: Cartridge Integration

**Goal**: Package as `.pxa` cartridge

**Files to create**:
1. `pixel_llm/models/pack_pixellm_cartridge.py`
2. `pixellm_v0.pxa` containing:
   - `weights.pxi` - Model weights
   - `program.pxi` - Inference program (if pxVM)
   - `forward.wgsl` - Shader code (if GPU)
   - `manifest.json` - Cartridge metadata

**From pxOS**:
```bash
$ python pxos_shim.py run --model pixellm_v0.pxa \
    --input "input_tokens.pxi" \
    --output "output_token.pxi"

# Or even simpler:
$ python pxos_shim.py ask pixellm_v0.pxa "pxOS evolution"
# Writes answer as pixels, displays on screen
```

---

## Why This Matters

### 1. True Pixel-Native AI
- Input: Pixels (token texture)
- Weights: Pixels (weight texture)
- Computation: Pixels (GPU textures or pxVM memory)
- Output: Pixels (result texture)

### 2. Cartridge Philosophy
Just like pxOS itself is a cartridge, models become cartridges:
```
pxos_v1_0_0.pxa     - Operating system cartridge
pixellm_v0.pxa      - AI model cartridge
```

Mount different model cartridges like swapping game carts.

### 3. Visual AI
Output is literally visible:
- Logits as heatmap (32×32 image)
- Prediction as colored pixel
- Embeddings as gradients

You can *see* the AI thinking.

### 4. Evolution-Ready
LLMs can propose new model architectures → build them → encode as pixels → test → promote.

The entire evolution loop stays in the pixel world.

---

## Next Concrete Step

**Build Phase 1**: Pixel I/O while keeping numpy computation

This gets us:
```
Pixel input → numpy computation → Pixel output
```

Which is halfway to pure-pixel and immediately usable.

Then Phase 2 (GPU shaders) or Phase 3 (pxVM) can replace the middle.

**Want me to implement Phase 1 now?** I can build:
1. `pixel_encode_tokens()` - text → token.pxi
2. `pixel_decode_tokens()` - token.pxi → token IDs
3. `pixel_encode_prediction()` - logits → output.pxi
4. `pixel_decode_prediction()` - output.pxi → text
5. `pixellm_infer_pure.py` - Uses pixel I/O end-to-end

This would make Pixel-LLM *feel* pixel-native even before GPU/pxVM!
