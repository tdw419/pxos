# Pixel-LLM Pixel Protocol v1

**Status**: Official specification for Pixel-LLM I/O
**Version**: 1.0.0
**Date**: 2025-11-16

---

## Contract: Pixel-Only API

Pixel-LLM is a **pixel coprocessor**. External systems communicate with it exclusively through pixel files (.pxi PNG images).

**Internal implementation is opaque**: Could be numpy, GPU shaders, or pxVM. The contract remains stable.

---

## Input Format: Token Pixels

**File**: `input_tokens.pxi`
**Format**: PNG, RGB, 1 × (N+1) pixels

### Pixel Layout

```
Pixel[0]: Metadata
  R = num_tokens % 256
  G = num_tokens // 256
  B = version (currently 1)

Pixel[1..N]: Token IDs
  R = token_id % 256
  G = token_id // 256
  B = unused (0)
```

### Example

Input text: "pxOS" (4 tokens: [112, 120, 79, 83])

```
Pixel[0] = (4, 0, 1)     # 4 tokens, version 1
Pixel[1] = (112, 0, 0)   # 'p'
Pixel[2] = (120, 0, 0)   # 'x'
Pixel[3] = (79, 0, 0)    # 'O'
Pixel[4] = (83, 0, 0)    # 'S'
```

Image dimensions: 5×1 RGB PNG

### Constraints

- Maximum tokens: 65,535 (16-bit encoding)
- Token IDs: 0-1023 (vocab size)
- Version byte: Must be 1

### Creation

```python
from pixel_llm.core.pixel_io import pixel_encode_tokens
pixel_encode_tokens("text", "input_tokens.pxi")
```

Or manually:
```python
import numpy as np
from PIL import Image

tokens = [112, 120, 79, 83]
n = len(tokens)

img = np.zeros((1, n+1, 3), dtype=np.uint8)
img[0, 0] = (n % 256, n // 256, 1)
for i, tok in enumerate(tokens):
    img[0, i+1] = (tok % 256, tok // 256, 0)

Image.fromarray(img, "RGB").save("input_tokens.pxi", "PNG")
```

---

## Output Format 1: Prediction Pixels

**File**: `output_token.pxi`
**Format**: PNG, RGB, 1 × K pixels (K = top-k predictions, default 5)

### Pixel Layout

```
Pixel[0..K-1]: Predictions (sorted by confidence)
  R = token_id % 256
  G = token_id // 256
  B = confidence (0-255, higher = more confident)
```

### Example

Top-5 predictions: tokens [32, 101, 116, 10, 115] with confidences [0.286, 0.176, 0.176, 0.176, 0.172]

```
Pixel[0] = (32, 0, 73)   # ' ' (space), 28.6% confidence
Pixel[1] = (101, 0, 45)  # 'e', 17.6% confidence
Pixel[2] = (116, 0, 45)  # 't', 17.6% confidence
Pixel[3] = (10, 0, 45)   # '\n', 17.6% confidence
Pixel[4] = (115, 0, 44)  # 's', 17.2% confidence
```

Image dimensions: 5×1 RGB PNG

### Decoding

```python
from pixel_llm.core.pixel_io import pixel_decode_prediction
pred = pixel_decode_prediction("output_token.pxi")
# Returns: {"top_tokens": [...], "top_probs": [...], "top_token": ..., "top_prob": ...}
```

Or manually:
```python
import numpy as np
from PIL import Image

img = Image.open("output_token.pxi").convert("RGB")
arr = np.array(img, dtype=np.uint8)

for i in range(arr.shape[1]):
    token_id = int(arr[0, i, 0]) + int(arr[0, i, 1]) * 256
    prob = float(arr[0, i, 2]) / 255.0
    print(f"Token {token_id}: {prob:.3f}")
```

---

## Output Format 2: Logits Heatmap

**File**: `logits_heatmap.pxi`
**Format**: PNG, RGB, 32 × 32 pixels (or H × W where H*W ≥ vocab_size)

### Pixel Layout

Logits mapped to 2D grid (row-major):
```
Pixel[row, col] represents logit[row * width + col]
  R = normalized_logit (0-255, brighter = higher logit)
  G = normalized_logit (same as R for grayscale)
  B = normalized_logit (same as R for grayscale)
```

For vocab_size=1024, a 32×32 grid covers all tokens.

### Example

Logit values range from -0.22 to +0.69:
- Min logit (-0.22) → pixel (0, 0, 0) - black
- Max logit (+0.69) → pixel (255, 255, 255) - white
- Mid logits → gray

### Visual Inspection

**You can literally see the AI's predictions:**
- Open `logits_heatmap.pxi` in any image viewer
- Bright pixels = high-confidence tokens
- Dark pixels = low-confidence tokens
- Pattern shows what the model learned

### Creation

```python
from pixel_llm.core.pixel_io import pixel_encode_logits_heatmap
pixel_encode_logits_heatmap(logits, "logits_heatmap.pxi", width=32)
```

---

## Protocol: Pixel Coprocessor API

### Request

1. **Create input pixel file**:
   ```bash
   # Method 1: From text
   python3 -c "from pixel_llm.core.pixel_io import pixel_encode_tokens; \
               pixel_encode_tokens('pxOS evolution', 'input_tokens.pxi')"

   # Method 2: Manual creation (any language)
   # Write 1×(N+1) PNG with format specified above
   ```

2. **Invoke Pixel-LLM**:
   ```bash
   python3 pixel_llm/programs/pixellm_infer_pure.py --input input_tokens.pxi
   ```

3. **Read output pixel files**:
   ```bash
   # Output files:
   #   pixel_llm/outputs/output_token.pxi - Prediction
   #   pixel_llm/outputs/logits_heatmap.pxi - Visual logits
   ```

### Response

- `output_token.pxi` - 1×K PNG (default K=5)
- `logits_heatmap.pxi` - 32×32 PNG (vocab visualization)

### Contract Guarantees

**Interface stability**:
- Input format: Fixed (v1)
- Output formats: Fixed (v1)
- Implementation: **Opaque** (can change without breaking protocol)

**Current implementation**: numpy (Phase 1)
**Future implementations**:
- Phase 2: GPU shaders (same pixel protocol)
- Phase 3: pxVM opcodes (same pixel protocol)
- Phase 4: Cartridge execution (same pixel protocol)

**The pixel files are the API. Everything else is firmware.**

---

## Implementation Versions

### Phase 1: Pixel I/O (Current) ✅

**Files**: `pixellm_infer_pure.py`

**Implementation**:
```
input_tokens.pxi → pixel_decode_tokens() → numpy arrays
  ↓
  numpy matmul using weights from pixellm_v0.pxi
  ↓
numpy logits → pixel_encode_prediction() → output_token.pxi
              → pixel_encode_logits_heatmap() → logits_heatmap.pxi
```

**Contract**: ✅ Stable
**Internals**: numpy (visible only to implementation)

### Phase 2: GPU Shaders (Future)

**Implementation**:
```
input_tokens.pxi → GPU texture
pixellm_v0.pxi → GPU texture
  ↓
  WGSL compute shader (all math on GPU)
  ↓
GPU output texture → output_token.pxi
                   → logits_heatmap.pxi
```

**Contract**: ✅ Unchanged
**Internals**: WebGPU shaders (invisible to callers)

### Phase 3: pxVM Execution (Future)

**Implementation**:
```
input_tokens.pxi → pxVM memory
pixellm_v0.pxi → pxVM memory
pixellm_forward.pxi → pxVM program
  ↓
  pxVM executes pixel opcodes
  ↓
pxVM output → output_token.pxi
            → logits_heatmap.pxi
```

**Contract**: ✅ Unchanged
**Internals**: pxVM bytecode (invisible to callers)

---

## Usage Examples

### Example 1: Query from Claude Code Research

```python
# Claude Code Research wants Pixel-LLM's opinion on design choices

# 1. Create input
from pixel_llm.core.pixel_io import pixel_encode_tokens
pixel_encode_tokens("modular architecture", "query.pxi")

# 2. Invoke (black box)
import subprocess
subprocess.run([
    "python3",
    "pixel_llm/programs/pixellm_infer_pure.py",
    "--input", "query.pxi"
])

# 3. Read output
from pixel_llm.core.pixel_io import pixel_decode_prediction
result = pixel_decode_prediction("pixel_llm/outputs/output_token.pxi")
print(f"Pixel-LLM suggests token: {result['top_token']}")
```

**From Claude's perspective**: Never touched numpy, just pixel files.

### Example 2: Batch Processing

```bash
# Create multiple queries
for text in "option_a" "option_b" "option_c"; do
    python3 -c "from pixel_llm.core.pixel_io import pixel_encode_tokens; \
                pixel_encode_tokens('$text', '${text}.pxi')"

    python3 pixel_llm/programs/pixellm_infer_pure.py --input "${text}.pxi"

    mv pixel_llm/outputs/output_token.pxi "${text}_result.pxi"
done

# Compare results visually
open *_result.pxi  # All are pixel images
```

### Example 3: Visual Debugging

```bash
# Run inference
python3 pixel_llm/programs/pixellm_infer_pure.py "pxOS hypervisor"

# Open heatmap
open pixel_llm/outputs/logits_heatmap.pxi

# See what the model thinks:
# - Bright pixels = high confidence
# - Dark pixels = low confidence
# - Visual pattern shows learned preferences
```

---

## Non-Goals

This protocol **does not** specify:

- **How weights are stored internally** (currently .npz → .pxi, could change)
- **What math library is used** (numpy, GPU, pxVM - opaque)
- **Intermediate activations** (internal only)
- **Training format** (separate concern)

These are **implementation details**. Only the pixel I/O contract matters.

---

## Extension Points

### Future: Multi-token Output

Currently outputs top-K single predictions. Future versions could support:
- **Sequence generation**: `output_sequence.pxi` - 1×M pixels (M generated tokens)
- **Beam search**: `output_beams.pxi` - K×M pixels (K beams, M tokens each)

### Future: Attention Visualization

For transformer models:
- **Attention map**: `attention.pxi` - N×N pixels showing attention weights
- **Layer activations**: `activations_L{i}.pxi` - Per-layer activation heatmaps

### Future: Streaming

For real-time generation:
- **Token stream**: Append pixels to `output_sequence.pxi` as they're generated
- **Watch mode**: Tail the file to see predictions appear

---

## Compliance

### Valid Pixel-LLM Implementation

Any implementation is valid if it:
1. Accepts `input_tokens.pxi` in specified format
2. Produces `output_token.pxi` and `logits_heatmap.pxi` in specified formats
3. Maintains semantic consistency (same inputs → similar outputs)

### Testing Compliance

```python
# Reference implementation test
from pixel_llm.core.pixel_io import pixel_encode_tokens, pixel_decode_prediction

# Create test input
pixel_encode_tokens("test", "test_in.pxi")

# Run both implementations
run_implementation_a("test_in.pxi")  # → output_a.pxi
run_implementation_b("test_in.pxi")  # → output_b.pxi

# Compare outputs
pred_a = pixel_decode_prediction("output_a.pxi")
pred_b = pixel_decode_prediction("output_b.pxi")

# Should be similar (not necessarily identical due to precision)
assert abs(pred_a["top_prob"] - pred_b["top_prob"]) < 0.1
```

---

## Summary

**Pixel-LLM Pixel Protocol v1** defines:
- ✅ Input: `input_tokens.pxi` (1×(N+1) PNG)
- ✅ Output: `output_token.pxi` (1×K PNG) + `logits_heatmap.pxi` (32×32 PNG)
- ✅ Contract: Stable across implementation changes
- ✅ Usage: Pixel files in → pixel files out

**The pixels are the API. Everything else is firmware.**

---

**Version History**:
- v1.0.0 (2025-11-16): Initial specification
