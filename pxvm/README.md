# pxVM - Pixel Virtual Machine

**Status**: v0.0.1 - First pixel program operational ✅

---

## What is pxVM?

pxVM is a virtual machine where **programs, data, and results are all encoded as pixels**.

**Core principle**: Computation happens *inside* the image.

Unlike traditional VMs where bytecode is stored in binary files, pxVM stores:
- **Instructions** as pixels (one instruction = one RGBA pixel)
- **Data** as pixels (values encoded in color channels)
- **Results** as pixels (computation mutates the image in-place)

---

## First Pixel Program: `OP_DOT_RGB`

The first operational pxVM program computes a **dot product** using only pixels.

### What It Does

Given two vectors encoded as pixels:
- **Vector A** = [1, 2, 3, 4]
- **Vector B** = [10, 20, 30, 40]

Compute: `dot = 1×10 + 2×20 + 3×30 + 4×40 = 300`

Store result in a pixel: `(R=44, G=1)` encoding 300 as a 16-bit integer.

### The Program Image

**File**: `pxvm/examples/dot_test.pxi` (101 bytes!)

**Structure** (16×16 RGBA PNG):

```
Row 0 (Instructions):
  Pixel (0,0): (1, 1, 2, 3) → OP_DOT_RGB(rowA=1, rowB=2, rowOut=3)
  Pixel (1,0): (0, 0, 0, 0) → OP_HALT

Row 1 (Vector A):
  Pixel (0,1): (1, 0, 0, 0) → value=1
  Pixel (1,1): (2, 0, 0, 0) → value=2
  Pixel (2,1): (3, 0, 0, 0) → value=3
  Pixel (3,1): (4, 0, 0, 0) → value=4

Row 2 (Vector B):
  Pixel (0,2): (10, 0, 0, 0) → value=10
  Pixel (1,2): (20, 0, 0, 0) → value=20
  Pixel (2,2): (30, 0, 0, 0) → value=30
  Pixel (3,2): (40, 0, 0, 0) → value=40

Row 3 (Output):
  Pixel (0,3): (0, 0, 0, 0) → Initially zero
                             → After execution: (44, 1, 0, 0) = 300
```

This single image contains:
- ✅ The program (2 instruction pixels)
- ✅ The input data (8 value pixels)
- ✅ The output location (1 result pixel)
- ✅ Workspace (all other pixels)

---

## Instruction Encoding

Each instruction is **one RGBA pixel**:

```
R = OPCODE
G = ARG0
B = ARG1
A = ARG2
```

### Opcodes v0.0.1

| Opcode | Name | Description |
|--------|------|-------------|
| 0 | `OP_HALT` | Stop execution |
| 1 | `OP_DOT_RGB` | Integer dot product over R channel |

### `OP_DOT_RGB` Semantics

```
Pixel format: (1, rowA, rowB, rowOut)

Execution:
1. Read vector A from row rowA (R channel values)
2. Read vector B from row rowB (R channel values)
3. Compute: dot = Σ (A[i].R × B[i].R)
4. Store low/high bytes of dot in pixel (0, rowOut):
   - R channel ← dot & 0xFF (low byte)
   - G channel ← (dot >> 8) & 0xFF (high byte)
```

Vector length is inferred by scanning until both rows have zero pixels.

---

## Running Pixel Programs

### Generate a test program

```bash
python3 -m pxvm.examples.make_dot_test
```

**Output**: `pxvm/examples/dot_test.pxi` (101-byte PNG)

### Execute the program

```bash
python3 -m pxvm.examples.run_dot_test
```

**Output**:
```
Dot product read from pixel (0,3): 300
Encoded as bytes: R=44, G=1
```

### Run unit tests

```bash
python3 -c "from pxvm.tests.test_dot_rgb import test_op_dot_rgb_basic; test_op_dot_rgb_basic(); print('✅ PASSED')"
```

---

## Architecture

### Interpreter Model

```python
from pxvm.core.interpreter import run_program
import numpy as np
from PIL import Image

# Load a .pxi program image
img = np.array(Image.open("program.pxi"))

# Execute (mutates img in-place)
result_img = run_program(img)

# Read results from specific pixels
output_pixel = result_img[3, 0]  # Read pixel at (0, 3)
```

### Execution Model

1. **PC (Program Counter)** starts at pixel (0, 0) in row 0
2. **Fetch** instruction pixel at PC
3. **Decode** opcode from R channel, args from G/B/A
4. **Execute** instruction (may read/write other pixels)
5. **Advance** PC to next pixel
6. **Repeat** until `OP_HALT` or end of row

### Memory Model

- **Row 0** = Instruction memory (program)
- **Rows 1+** = Data memory (vectors, matrices, results)
- No explicit registers - all data lives in pixels
- Instructions reference rows/columns as operands

---

## Why This Matters

### Philosophical Achievement

**Traditional computing**:
```
Program (binary) → CPU → Data (RAM) → Results (RAM)
```

**Pixel computing**:
```
Image → pxVM → Image (mutated)
```

Everything is pixels. The boundary between code and data dissolves.

### Technical Achievement

This proves that:
- ✅ **Instructions can be pixels** (not just binary opcodes)
- ✅ **Data can be pixels** (not just memory addresses)
- ✅ **Computation can happen inside an image** (not just on external data)
- ✅ **Results are pixels** (the image is both input and output)

### Practical Achievement

`dot_test.pxi` is:
- **101 bytes** total
- **Viewable** in any image viewer
- **Executable** by pxVM interpreter
- **Self-contained** (program + data + result)

You can literally **see** the program if you zoom into the pixels.

---

## Next Steps

### Near-term opcodes

- `OP_MUL_ADD` - Fused multiply-add over rows
- `OP_RELU` - ReLU activation over a row
- `OP_LOAD_ROW` / `OP_STORE_ROW` - Memory operations
- `OP_MATMUL` - Matrix multiply over pixel regions

### Integration with Pixel-LLM

Current Pixel-LLM forward pass (numpy):
```python
h_hidden = relu(h_in @ W_hidden + b_hidden)
logits = h_hidden @ W_out + b_out
```

Future Pixel-LLM forward pass (pxVM program):
```python
# Load pixellm_forward.pxi (contains OP_MATMUL, OP_RELU, etc.)
program_img = load_pxi("pixellm_forward.pxi")
result_img = run_program(program_img)
logits = extract_logits_from_pixels(result_img, row=5)
```

The entire forward pass becomes a **pixel program**.

### GPU acceleration

Two paths:

**Path A**: Implement pxVM interpreter in WGSL/compute shaders
```
.pxi program → GPU texture → shader executes opcodes → GPU texture out
```

**Path B**: Compile pxVM opcodes to shader passes
```
OP_DOT_RGB → dot_product_shader.wgsl
OP_MATMUL → matmul_shader.wgsl
```

Either way: **pixels in, pixels out**. Implementation hidden.

---

## Implementation

### Core Files

| File | Purpose |
|------|---------|
| `pxvm/core/opcodes.py` | Opcode definitions (OP_HALT, OP_DOT_RGB) |
| `pxvm/core/interpreter.py` | Execution engine (run_program) |

### Examples

| File | Purpose |
|------|---------|
| `pxvm/examples/make_dot_test.py` | Generate dot_test.pxi |
| `pxvm/examples/run_dot_test.py` | Execute dot_test.pxi |
| `pxvm/examples/dot_test.pxi` | First pixel program (101 bytes) |

### Tests

| File | Purpose |
|------|---------|
| `pxvm/tests/test_dot_rgb.py` | Unit test for OP_DOT_RGB |

---

## Design Principles

### 1. Pixels are the substrate

Everything must be encodable as pixels:
- Instructions (opcodes + args)
- Integers (low/high bytes in channels)
- Floats (future: quantized or IEEE754 in RGBA)
- Pointers (row/column indices)

### 2. Programs are images

A `.pxi` file is:
- Valid PNG (can be viewed/edited with image tools)
- Executable bytecode (can be run by pxVM)
- Self-describing (structure visible in pixel grid)

### 3. Execution is mutation

`run_program(img)` doesn't return separate output - it **mutates the image in-place**.

The program consumes itself, like a chemical reaction.

### 4. No external state

- No stack (use pixel rows)
- No heap (use pixel regions)
- No registers (use designated pixels)
- Everything visible in the image

### 5. Future-proof

pxVM programs should run on:
- ✅ Python interpreter (current)
- 🔲 GPU shaders (future - Phase 2)
- 🔲 pxVM hardware (future - Phase 3)

The pixel encoding never changes. Only the interpreter evolves.

---

## Comparison to Traditional VMs

| Aspect | Traditional VM | pxVM |
|--------|----------------|------|
| **Program format** | Binary bytecode | PNG image |
| **Instruction size** | 1-4 bytes | 4 bytes (1 RGBA pixel) |
| **Data storage** | RAM addresses | Pixel coordinates |
| **Operands** | Registers/stack | Row/column indices |
| **Results** | Write to memory | Mutate pixels in-place |
| **Debuggability** | Requires debugger | Open image in viewer |
| **Human readability** | Disassembler needed | Zoom into pixels |

---

## Validation

### Test Results

```bash
$ python3 -m pxvm.examples.run_dot_test
Dot product read from pixel (0,3): 300
Encoded as bytes: R=44, G=1

$ python3 -c "from pxvm.tests.test_dot_rgb import test_op_dot_rgb_basic; test_op_dot_rgb_basic(); print('✅ PASSED')"
✅ PASSED
```

### Verification

```bash
$ ls -lh pxvm/examples/dot_test.pxi
-rw-r--r-- 1 root root 101 Nov 17 00:39 dot_test.pxi

$ file pxvm/examples/dot_test.pxi
pxvm/examples/dot_test.pxi: PNG image data, 16 x 16, 8-bit/color RGBA, non-interlaced
```

**101 bytes** contains:
- Complete program (2 instructions)
- Complete dataset (2 vectors × 4 elements)
- Complete output (1 result pixel)

---

## Status

**pxVM v0.0.1**: First pixel program operational ✅

**What works**:
- ✅ Instruction encoding (RGBA pixels)
- ✅ Program execution (fetch-decode-execute)
- ✅ Data encoding (integers in R channel)
- ✅ Result storage (low/high bytes in R/G)
- ✅ OP_DOT_RGB implementation
- ✅ Test harness and validation

**What's next**:
- 🔲 Additional opcodes (MUL_ADD, RELU, MATMUL)
- 🔲 Floating-point support (quantized or IEEE754)
- 🔲 Pixel-LLM integration (forward pass as .pxi program)
- 🔲 GPU interpreter (WGSL compute shader)
- 🔲 Performance benchmarks

---

**First proof that computation lives inside the image**: ✅

**pxVM v0.0.1** - Computation is pixels 🎨🧠
