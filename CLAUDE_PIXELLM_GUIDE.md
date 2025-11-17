# Guide for Claude Code Research: Using Pixel-LLM as a Pixel Coprocessor

**Audience**: Claude Code Research (AI assistant)
**Purpose**: How to use Pixel-LLM without touching numpy/Python internals
**Contract**: Pixel files only

---

## Core Principle

**Pixel-LLM is a pixel coprocessor**. You communicate with it exclusively through pixel files (.pxi PNG images).

**You are NOT allowed to**:
- ❌ Import `pixel_model_loader` directly
- ❌ Call `pixellm_forward()` directly
- ❌ Access numpy arrays or model weights
- ❌ Inspect internal implementation

**You MUST**:
- ✅ Create `.pxi` pixel files for input
- ✅ Call `pixellm_infer_pure.py` as a black box
- ✅ Read only `.pxi` pixel files for output

---

## The Pixel Coprocessor API

### Input: Token Pixels

**File**: `input_tokens.pxi`
**Format**: 1×(N+1) RGB PNG

**How to create**:
```python
from pixel_llm.core.pixel_io import pixel_encode_tokens

# Encode text as pixels
pixel_encode_tokens("your text here", "input_tokens.pxi")
```

**What this does**: Converts text → token IDs → pixel image
**What's inside**: You don't need to know. It's a pixel file.

### Execution: Black Box

**Command**:
```bash
python3 pixel_llm/programs/pixellm_infer_pure.py --input input_tokens.pxi
```

**What happens**: Pixel-LLM processes the input (implementation is opaque)
**Output location**: `pixel_llm/outputs/`

### Output: Result Pixels

**Files produced**:
1. `output_token.pxi` - Prediction as pixels (1×5 PNG)
2. `logits_heatmap.pxi` - Visual logits (32×32 PNG)

**How to read**:
```python
from pixel_llm.core.pixel_io import pixel_decode_prediction

# Decode prediction from pixels
result = pixel_decode_prediction("pixel_llm/outputs/output_token.pxi")

# Access results
top_token = result["top_token"]        # Most likely token ID
top_prob = result["top_prob"]          # Confidence (0.0-1.0)
top_5_tokens = result["top_tokens"]    # Top 5 token IDs
top_5_probs = result["top_probs"]      # Top 5 probabilities
```

**What's inside**: Pixel-encoded predictions. Implementation details are hidden.

---

## Usage Patterns

### Pattern 1: Simple Query

**Use case**: Get Pixel-LLM's prediction for a prompt

```python
from pixel_llm.core.pixel_io import pixel_encode_tokens, pixel_decode_prediction
import subprocess

# 1. Create input pixels
pixel_encode_tokens("pxOS evolution", "query.pxi")

# 2. Invoke Pixel-LLM (black box)
subprocess.run([
    "python3",
    "pixel_llm/programs/pixellm_infer_pure.py",
    "--input", "query.pxi"
], check=True)

# 3. Read output pixels
result = pixel_decode_prediction("pixel_llm/outputs/output_token.pxi")

print(f"Pixel-LLM predicts token {result['top_token']} "
      f"with {result['top_prob']:.1%} confidence")
```

**Pixel contract**: ✅ Only pixel files touched

### Pattern 2: Score Multiple Options

**Use case**: Compare design alternatives using Pixel-LLM

```python
from pixel_llm.core.pixel_io import pixel_encode_tokens, pixel_decode_prediction
import subprocess

options = [
    "simple modular architecture",
    "complex monolithic design",
    "distributed microservices"
]

scores = []

for option in options:
    # Create input pixels
    pixel_encode_tokens(option, f"option_{len(scores)}.pxi")

    # Invoke Pixel-LLM
    subprocess.run([
        "python3",
        "pixel_llm/programs/pixellm_infer_pure.py",
        "--input", f"option_{len(scores)}.pxi"
    ], check=True, capture_output=True)

    # Read output pixels
    result = pixel_decode_prediction("pixel_llm/outputs/output_token.pxi")
    scores.append((option, result['top_prob']))

# Sort by score
scores.sort(key=lambda x: x[1], reverse=True)

print("Pixel-LLM rankings:")
for i, (option, score) in enumerate(scores, 1):
    print(f"  {i}. {option} (score: {score:.3f})")
```

**Pixel contract**: ✅ Only pixel files touched

### Pattern 3: Visual Inspection

**Use case**: See what Pixel-LLM is "thinking"

```python
from pixel_llm.core.pixel_io import pixel_encode_tokens
import subprocess
from pathlib import Path

# Create query
pixel_encode_tokens("pixel substrate primitives", "design_query.pxi")

# Invoke Pixel-LLM
subprocess.run([
    "python3",
    "pixel_llm/programs/pixellm_infer_pure.py",
    "--input", "design_query.pxi"
], check=True)

# Check if heatmap exists
heatmap = Path("pixel_llm/outputs/logits_heatmap.pxi")
if heatmap.exists():
    print(f"✅ Logits heatmap created: {heatmap}")
    print("   Open in image viewer to see Pixel-LLM's preferences visually")
    print("   Bright pixels = high confidence tokens")
```

**Pixel contract**: ✅ Only pixel files touched
**Bonus**: You can literally **see** the AI's predictions!

---

## Integration Examples

### Example 1: Helper Script

Create `tools/pixellm_helper.py`:

```python
#!/usr/bin/env python3
"""
Helper script for Claude Code Research to use Pixel-LLM.
"""

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixel_llm.core.pixel_io import pixel_encode_tokens, pixel_decode_prediction


def ask_pixellm(text: str) -> dict:
    """
    Ask Pixel-LLM a question using only pixels.

    Args:
        text: Input text

    Returns:
        dict with top_token, top_prob, top_tokens, top_probs
    """
    # Create input pixels
    input_path = Path("temp_query.pxi")
    pixel_encode_tokens(text, str(input_path))

    # Invoke Pixel-LLM (black box)
    result = subprocess.run([
        "python3",
        "pixel_llm/programs/pixellm_infer_pure.py",
        "--input", str(input_path)
    ], capture_output=True, text=True, check=True)

    # Read output pixels
    output_path = Path("pixel_llm/outputs/output_token.pxi")
    prediction = pixel_decode_prediction(str(output_path))

    # Cleanup temp file
    input_path.unlink(missing_ok=True)

    return prediction


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: pixellm_helper.py 'text to query'")
        sys.exit(1)

    text = sys.argv[1]
    result = ask_pixellm(text)

    print(f"Query: '{text}'")
    print(f"Top prediction: token {result['top_token']} ({result['top_prob']:.1%})")
    print(f"Top 5: {result['top_tokens']}")
```

**Usage from Claude Code Research**:
```bash
python3 tools/pixellm_helper.py "evolution system design"
```

### Example 2: Design Choice Script

Create `tools/compare_designs.py`:

```python
#!/usr/bin/env python3
"""
Use Pixel-LLM to compare design options.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.pixellm_helper import ask_pixellm


def compare_designs(options: list[str]) -> list[tuple[str, float]]:
    """Compare designs using Pixel-LLM."""
    scores = []

    print("Consulting Pixel-LLM...")
    for option in options:
        result = ask_pixellm(option)
        score = result['top_prob']
        scores.append((option, score))
        print(f"  {option}: {score:.3f}")

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


if __name__ == "__main__":
    options = [
        "simple architecture with clear separation",
        "complex architecture with many dependencies",
        "modular pixel-based design"
    ]

    print("Design options:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")

    print()
    ranked = compare_designs(options)

    print("\nPixel-LLM ranking:")
    for i, (option, score) in enumerate(ranked, 1):
        marker = "👍" if i == 1 else "  "
        print(f"  {marker} {i}. {option} (score: {score:.3f})")
```

**Usage**:
```bash
python3 tools/compare_designs.py
```

---

## What You Should Know

### Current Implementation (Phase 1)

**Internals**: numpy matmul
**Your concern**: None - it's a black box

**What matters**:
- Input format: `.pxi` pixel files ✅
- Output format: `.pxi` pixel files ✅
- Contract: Stable ✅

### Future Implementations (Phase 2/3)

**Internals**: GPU shaders or pxVM opcodes
**Your concern**: None - still a black box

**What matters**:
- Input format: `.pxi` pixel files ✅ (unchanged)
- Output format: `.pxi` pixel files ✅ (unchanged)
- Contract: Stable ✅

**The implementation can change completely. Your code won't break.**

### Why This Matters

By treating Pixel-LLM as a pixel coprocessor:
- ✅ You never depend on internal implementation
- ✅ Your integration code is future-proof
- ✅ Pixel-LLM can upgrade (numpy → GPU → pxVM) transparently
- ✅ The "pixel-only" vision is real at the interface

---

## Examples of What NOT to Do

### ❌ Don't Do This

```python
# BAD: Accessing internals directly
from pixel_llm.core.pixel_model_loader import get_default_loader
from pixel_llm.programs.pixellm_infer import pixellm_forward

loader = get_default_loader()  # DON'T
logits = pixellm_forward(tokens, loader)  # DON'T
```

**Why bad**: Couples your code to numpy implementation. Breaks when we move to GPU/pxVM.

### ❌ Don't Do This

```python
# BAD: Assuming numpy arrays
import numpy as np

weights = np.load("pixel_llm/models/pixellm_v0.npz")  # DON'T
result = weights["embed"] @ my_vector  # DON'T
```

**Why bad**: Violates pixel-only contract. You're not supposed to see numpy.

### ✅ Do This Instead

```python
# GOOD: Pixel coprocessor API
from pixel_llm.core.pixel_io import pixel_encode_tokens, pixel_decode_prediction
import subprocess

# Create pixel input
pixel_encode_tokens("query", "input.pxi")

# Invoke black box
subprocess.run(["python3", "pixel_llm/programs/pixellm_infer_pure.py",
                "--input", "input.pxi"], check=True)

# Read pixel output
result = pixel_decode_prediction("pixel_llm/outputs/output_token.pxi")
```

**Why good**: Uses only pixel files. Future-proof. Respects abstraction.

---

## Troubleshooting

### Q: How do I know what Pixel-LLM is predicting?

**A**: Look at `output_token.pxi`:

```python
from pixel_llm.core.pixel_io import pixel_decode_prediction

result = pixel_decode_prediction("pixel_llm/outputs/output_token.pxi")

# Top prediction
print(f"Most likely token: {result['top_token']}")
print(f"Confidence: {result['top_prob']:.1%}")

# All top-5
for token, prob in zip(result['top_tokens'], result['top_probs']):
    print(f"  Token {token}: {prob:.1%}")
```

### Q: Can I see what the model is "thinking"?

**A**: Yes! Open `logits_heatmap.pxi` in an image viewer:

```bash
open pixel_llm/outputs/logits_heatmap.pxi
```

- Bright pixels = high confidence
- Dark pixels = low confidence
- Visual pattern shows learned preferences

### Q: How do I create custom input pixels?

**A**: Use `pixel_encode_tokens()`:

```python
from pixel_llm.core.pixel_io import pixel_encode_tokens

pixel_encode_tokens("your custom text", "custom_input.pxi")
```

### Q: Can I process multiple queries in parallel?

**A**: Yes, but manage output files:

```python
import subprocess
from pixel_llm.core.pixel_io import pixel_encode_tokens, pixel_decode_prediction

queries = ["query1", "query2", "query3"]

for i, query in enumerate(queries):
    # Create unique input
    input_file = f"query_{i}.pxi"
    pixel_encode_tokens(query, input_file)

    # Run inference
    subprocess.run([
        "python3",
        "pixel_llm/programs/pixellm_infer_pure.py",
        "--input", input_file
    ], check=True)

    # Save output before next query
    output = Path("pixel_llm/outputs/output_token.pxi")
    output.rename(f"result_{i}.pxi")
```

### Q: What if the pixel files are too large?

**A**: They won't be. Examples:
- `input_tokens.pxi` for "pxOS evolution system" (21 tokens): **115 bytes**
- `output_token.pxi` (top-5 predictions): **81 bytes**
- `logits_heatmap.pxi` (32×32 visual): **2.5 KB**

Pixel files are tiny PNGs.

---

## Summary

**How to use Pixel-LLM from Claude Code Research**:

1. **Create input pixels**:
   ```python
   from pixel_llm.core.pixel_io import pixel_encode_tokens
   pixel_encode_tokens("text", "input.pxi")
   ```

2. **Invoke Pixel-LLM** (black box):
   ```bash
   python3 pixel_llm/programs/pixellm_infer_pure.py --input input.pxi
   ```

3. **Read output pixels**:
   ```python
   from pixel_llm.core.pixel_io import pixel_decode_prediction
   result = pixel_decode_prediction("pixel_llm/outputs/output_token.pxi")
   ```

**Contract**: Pixel files only. No internal access. Future-proof.

**The pixels are the API. Everything else is firmware.**

---

**Questions? See**:
- `PIXEL_PROTOCOL.md` - Official pixel format specification
- `PIXEL_ONLY_INFERENCE.md` - Complete vision and roadmap
- `pixel_llm/core/pixel_io.py` - Pixel I/O implementation
