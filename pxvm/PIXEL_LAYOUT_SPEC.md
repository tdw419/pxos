# Pixel Layout Specification v0.0.3

**Status**: Stable
**Last Updated**: 2024-11-17
**Breaking Changes**: None allowed without major version bump

This document defines the canonical pixel encoding format for pxVM programs.

## Overview

pxVM programs are RGBA PNG images where:
- **Instructions** are pixels in row 0
- **Data** (matrices, vectors, buffers) occupy rows 1+
- **Execution** mutates the image in-place

This format is **executor-agnostic**: the same .pxi file runs on CPU (Python) and GPU (WGSL) without transformation.

---

## Image Format

- **Format**: PNG
- **Color Mode**: RGBA (4 channels, 8 bits each)
- **File Extension**: `.pxi` (pixel program)
- **Dimensions**: Variable (typically 16×16 to 4096×4096)
- **Endianness**: Little-endian for multi-byte values

---

## Row 0: Instructions

**Location**: Row 0, columns 0 to N-1
**Format**: One instruction per pixel (RGBA)

### Instruction Encoding

Each pixel encodes one instruction as:

```
(R, G, B, A) = (opcode, arg0, arg1, arg2)
```

- **R channel**: Opcode (0-255)
- **G channel**: Argument 0 (0-255)
- **B channel**: Argument 1 (0-255)
- **A channel**: Argument 2 (0-255)

### Opcodes (v0.0.3)

| Opcode | Name | Arguments | Description |
|--------|------|-----------|-------------|
| 0 | `OP_HALT` | - | Stop execution |
| 1 | `OP_DOT_RGB` | `row_a, row_b, row_out` | Integer dot product |
| 2 | `OP_ADD` | `row_a, row_b, row_out` | Element-wise add (clamped) |
| 3 | `OP_RELU` | `row_data, 0, 0` | In-place ReLU |
| 4 | `OP_MATMUL` | `row_a, row_b, row_c` | Matrix multiply C = A @ B |

### Program Counter

- **Initial PC**: (0, 0) - top-left pixel of row 0
- **Increment**: Move right one column per instruction
- **Termination**: Execute until `OP_HALT` or row boundary

### Example

```
Row 0:
  (0,0): (4, 1, 2, 4)  → MATMUL: row1 @ row2 → row4
  (1,0): (2, 4, 3, 4)  → ADD: row4 + row3 → row4
  (2,0): (3, 4, 0, 0)  → RELU: row4 (in-place)
  (3,0): (0, 0, 0, 0)  → HALT
```

---

## Rows 1+: Data

**Location**: Rows 1 through height-1
**Format**: Matrix encoding (header + data)

### Matrix Encoding

Each matrix occupies **1 header row + N data rows**.

#### Header Pixel (Column 0)

```
(R, G, B, A) = (cols_low, cols_high, rows_low, rows_high)
```

- **cols**: Total columns (width) of matrix
- **rows**: Total rows (height) of matrix
- **Encoding**: 16-bit little-endian
  - `cols = R + (G << 8)`
  - `rows = B + (A << 8)`

**Example**: 128×1024 matrix
```
R = 128 & 0xFF = 128
G = (128 >> 8) & 0xFF = 0
B = 1024 & 0xFF = 0
A = (1024 >> 8) & 0xFF = 4

Header = (128, 0, 0, 4)
```

#### Data Layout (Columns 1+)

Matrix elements are stored **row-major** (C order), flattened into 1D sequence.

- **Start**: Column 1 of header row
- **Channel**: R channel only (G, B, A unused)
- **Wrapping**: Data wraps to subsequent image rows when reaching image width

**Stride Calculation**:
```python
stride = image_width - 1  # Column 0 reserved for header

for i in range(total_elements):
    x = 1 + (i % stride)
    y = row_start + (i // stride)
    value = img[y, x, 0]  # R channel
```

**Example**: 3×2 matrix in 16-pixel-wide image

```
Matrix:
  [[1, 2, 3],
   [4, 5, 6]]

Image layout (row_start=1):
  Row 1, Col 0: (3, 0, 2, 0)  ← Header (cols=3, rows=2)
  Row 1, Col 1: (1, 0, 0, 0)  ← Data[0] = 1
  Row 1, Col 2: (2, 0, 0, 0)  ← Data[1] = 2
  Row 1, Col 3: (3, 0, 0, 0)  ← Data[2] = 3
  Row 1, Col 4: (4, 0, 0, 0)  ← Data[3] = 4
  Row 1, Col 5: (5, 0, 0, 0)  ← Data[4] = 5
  Row 1, Col 6: (6, 0, 0, 0)  ← Data[5] = 6
```

### Vector Encoding

1D vectors are encoded as **1×N matrices** (row vectors).

```python
# Vector [10, 20, 30, 40]
Header: (4, 0, 1, 0)  # cols=4, rows=1
Data: [10, 20, 30, 40]
```

### Data Types

**Current (v0.0.3)**: uint8 only (0-255)

**Quantization** (for float32 → uint8):
```python
min_val = data.min()
max_val = data.max()

if max_val > min_val:
    quantized = ((data - min_val) / (max_val - min_val) * 255).astype(uint8)
else:
    quantized = zeros(shape, dtype=uint8)
```

**Clamping** (on arithmetic overflow):
```python
result = max(0, min(255, acc))
```

---

## Layout Calculations

### Rows Needed for Matrix

Given matrix shape `(cols, rows)` and image width `W`:

```python
total_elements = cols * rows
stride = W - 1  # Column 0 is header
data_rows = (total_elements + stride - 1) // stride  # Ceiling division
total_rows = 1 + data_rows
```

**Example**: 128×128 matrix, image width 256
```
total_elements = 16,384
stride = 255
data_rows = ceil(16384 / 255) = 65
total_rows = 1 + 65 = 66 rows
```

### Image Size Calculation

For a program with matrices M₁, M₂, ..., Mₙ:

```python
height = 1  # Row 0: instructions

for matrix in matrices:
    height += calculate_matrix_rows(matrix.cols, matrix.rows, width)
```

**Recommendation**: Use power-of-2 dimensions (256, 512, 1024, 2048, 4096) for efficiency on GPU.

---

## Pixel-LLM Example

**Model**: 2-layer MLP (128-dim hidden, 1024 vocab)

### Weights

- `W_hidden`: 128×128
- `b_hidden`: 128 (as 1×128 vector)
- `W_out`: 128×1024
- `b_out`: 1024 (as 1×1024 vector)

### Layout (256×256 image)

| Rows | Name | Shape | Description |
|------|------|-------|-------------|
| 0 | Instructions | 6 ops | MATMUL, ADD, RELU, MATMUL, ADD, HALT |
| 1 | `h_in` | 1×128 | Input embedding (filled at runtime) |
| 2-67 | `W_hidden` | 128×128 | Hidden layer weights (66 rows) |
| 68-68 | `b_hidden` | 1×128 | Hidden bias (2 rows) |
| 69-69 | `h` | 1×128 | Hidden activations (placeholder, 2 rows) |
| 70-598 | `W_out` | 128×1024 | Output weights (529 rows) |
| 599-602 | `b_out` | 1×1024 | Output bias (5 rows) |
| 603-606 | `logits` | 1×1024 | Final output (placeholder, 5 rows) |

**Total**: 607 rows (fits in 256×1024 or 1024×1024 image)

### Instructions

```python
Row 0:
  (0,0): (4, 1,  2,  4)   # MATMUL: h = h_in @ W_hidden
  (1,0): (2, 4,  68, 4)   # ADD: h += b_hidden
  (2,0): (3, 4,  0,  0)   # RELU: h = relu(h)
  (3,0): (4, 4,  70, 603) # MATMUL: logits = h @ W_out
  (4,0): (2, 603, 599, 603) # ADD: logits += b_out
  (5,0): (0, 0,  0,  0)   # HALT
```

**Result**: Complete neural network encoded in 89KB PNG.

---

## Validation Rules

### Required Checks

1. **Image format**: Must be RGBA (4 channels)
2. **HALT instruction**: Row 0 must contain `OP_HALT`
3. **Matrix headers**: All referenced rows must have valid headers
4. **Shape compatibility**: MATMUL requires `cols_A == rows_B`
5. **Bounds**: All row references must be within image height

### Recommended Checks

1. **Instruction sequence**: Verify operations are in logical order
2. **Data initialization**: Warn if input buffers contain zeros
3. **Unused rows**: Flag unreferenced data regions

---

## Executor Requirements

Any pxVM executor (CPU, GPU, ASIC, etc.) must:

1. **Preserve format**: Output .pxi must be valid input to another executor
2. **Byte-identical semantics**: Same input → same output across platforms
3. **In-place mutation**: Modify image directly; no external I/O
4. **PC behavior**: Execute row 0 left-to-right until HALT

---

## Future Extensions (Non-Breaking)

### v0.1.0 (Planned)
- Float16/Float32 support (via new opcodes)
- Sparse matrix encoding (header flag)
- Conditional execution (OP_BRANCH)

### v0.2.0 (Research)
- Multi-channel data (RGB all used)
- Compression (LZ77 in unused alpha channel)
- Encrypted weights (AES in metadata)

---

## References

- **Specification**: This document
- **Implementation**: `pxvm/core/interpreter.py` (reference CPU executor)
- **GPU Implementation**: `pxvm/gpu/interpreter.wgsl`
- **Test Suite**: `pxvm/tests/test_*.py`
- **Utilities**: `pxvm/utils/layout.py`, `pxvm/utils/validation.py`

---

## Appendix: Decision Rationale

### Why RGBA PNG?
- Universal support (every platform, every language)
- Lossless compression built-in
- Viewable as image (debugging aid)
- GPU texture format (zero-copy upload)

### Why R-channel-only data?
- Leaves GBA for future extensions
- Simplifies initial implementation
- Clear visual debugging (grayscale = data)

### Why row-major?
- Matches C/Python numpy default
- Efficient for row-oriented operations
- Natural for GPU horizontal scan

### Why uint8?
- Hardware native (fastest)
- 256 levels sufficient for quantized neural nets
- Predictable overflow behavior
- Visual intuition (0=black, 255=white)

### Why in-place mutation?
- No I/O abstraction (pixels ARE computation)
- Simplifies GPU implementation (no buffer copies)
- Forces thinking in terms of memory layout
- Natural for neural nets (buffers reused)

---

**End of Specification v0.0.3**
