# pxIR - High-Level Intermediate Representation

**Version**: 1.0
**Status**: Production
**Purpose**: Typed, SSA-based compiler IR with first-class matrix support

---

## Overview

**pxIR** is the high-level intermediate representation for pxOS compilation. It sits **above** PXI Assembly in the IR hierarchy and provides:

- **Static Single Assignment (SSA)** form
- **Explicit typing** (i8, i16, i32, f32, mat, vec, ptr)
- **Matrix operations** as first-class citizens
- **Block-structured control flow** (CFG)
- **Optimization target** for compiler passes

---

## Two-Level IR Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Source Languages (Python, C, Binary lifters)            │
└────────────────────┬──────────────────────────────────────┘
                     │ Emit semantic operations
                     ▼
┌──────────────────────────────────────────────────────────┐
│  pxIR (High-Level IR) ◄── THIS LAYER                     │
│  - SSA form                                               │
│  - Typed (i8, i16, i32, f32, mat, vec, ptr)              │
│  - Matrix operations (MATMUL, RELU, SOFTMAX)             │
│  - Block structure (CFG)                                  │
│  - Optimization passes (const fold, DCE, CSE)            │
└────────────────────┬──────────────────────────────────────┘
                     │ Lowering
                     ▼
┌──────────────────────────────────────────────────────────┐
│  PXI Assembly (Low-Level IR)                             │
│  - Assembly-level instructions                            │
│  - Register operations                                    │
│  - Jump/branch                                            │
└────────────────────┬──────────────────────────────────────┘
                     │ Mechanical translation
                     ▼
┌──────────────────────────────────────────────────────────┐
│  Primitives (WRITE/DEFINE)                               │
└────────────────────┬──────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│  Binary (pxos.bin)                                        │
└──────────────────────────────────────────────────────────┘
```

### Why Two IRs?

**pxIR (High-Level):**
- Semantic, portable operations
- Type system enables optimization
- Matrix operations for AI workloads
- LLM-friendly representation

**PXI Assembly (Low-Level):**
- Direct assembly instruction mapping
- Register allocation
- x86-specific optimizations
- Machine code generation

---

## Core Data Structures

### Type System

```python
from pxir import Type

# Primitive types
i8 = Type.i8()         # 8-bit integer
i16 = Type.i16()       # 16-bit integer
i32 = Type.i32()       # 32-bit integer
f32 = Type.f32()       # 32-bit float
bool_t = Type.bool_type()  # Boolean

# Composite types
vec = Type.vec(Type.f32(), 128)  # Vector of 128 floats
mat = Type.mat(Type.f32(), 64, 128)  # 64x128 matrix of floats

# Quantized matrix
qmat = Type.mat(
    Type.i8(), 64, 128,
    quant_scheme="linear",
    quant_scale=0.01,
    quant_offset=127
)

# Pointer types
ptr = Type.ptr("mem", Type.i8())  # Pointer to i8 in main memory
fb_ptr = Type.ptr("framebuffer", Type.i8())  # Framebuffer pointer
```

### SSA Values

```python
from pxir import Value, Type

# SSA values are typed temporaries
v0 = Value("v0", Type.i32())  # %v0:i32
v1 = Value("v1", Type.f32())  # %v1:f32
h = Value("h", Type.mat(Type.f32(), 1, 128))  # %h:mat<f32,1,128>
```

### Operations

```python
from pxir import Op, Value

# Arithmetic
add_op = Op("ADD", [v0, v1], result=v2)  # %v2 = ADD(%v0, %v1)
mul_op = Op("MUL", [v0, v1], result=v3)  # %v3 = MUL(%v0, %v1)

# Matrix operations
matmul_op = Op("MATMUL", [h, W], result=logits)  # %logits = MATMUL(%h, %W)
relu_op = Op("RELU", [x], result=y)              # %y = RELU(%x)

# Memory operations
load_op = Op("LOAD", [ptr], result=value)        # %value = LOAD(%ptr)
store_op = Op("STORE", [ptr, value], has_side_effects=True)

# Control flow
br_op = Op("BR", [cond, "then_block", "else_block"])  # Conditional branch
jmp_op = Op("JMP", ["target_block"])                  # Unconditional jump
ret_op = Op("RET", [result])                          # Return
```

### Basic Blocks

```python
from pxir import Block, Op

block = Block("entry")
block.add_op(Op("ADD", [v0, v1], result=v2))
block.add_op(Op("MUL", [v2, v3], result=v4))
block.set_terminator(Op("RET", [v4]))

# Block structure:
# entry:
#   %v2 = ADD(%v0, %v1)
#   %v4 = MUL(%v2, %v3)
#   RET(%v4)
```

### Programs

```python
from pxir import Program, Block

program = Program()
program.add_block(entry_block)
program.add_block(loop_block)
program.entry_block = "entry"

# Serialize to JSON
json_str = program.to_json()
```

---

## Operation Reference

### Arithmetic

| Operation | Description | Example |
|-----------|-------------|---------|
| `ADD` | Addition | `%v2 = ADD(%v0, %v1)` |
| `SUB` | Subtraction | `%v2 = SUB(%v0, %v1)` |
| `MUL` | Multiplication | `%v2 = MUL(%v0, %v1)` |
| `DIV` | Division | `%v2 = DIV(%v0, %v1)` |
| `MOD` | Modulo | `%v2 = MOD(%v0, %v1)` |
| `NEG` | Negation | `%v1 = NEG(%v0)` |
| `ABS` | Absolute value | `%v1 = ABS(%v0)` |

### Matrix Operations

| Operation | Description | Example |
|-----------|-------------|---------|
| `MATMUL` | Matrix multiply | `%C = MATMUL(%A, %B)` |
| `RELU` | ReLU activation | `%y = RELU(%x)` |
| `SOFTMAX` | Softmax | `%probs = SOFTMAX(%logits)` |
| `ARGMAX` | Argmax | `%idx = ARGMAX(%vec)` |
| `SAMPLE` | Sample from distribution | `%token = SAMPLE(%probs, temp=0.8)` |

### Memory Operations

| Operation | Description | Example |
|-----------|-------------|---------|
| `LOAD` | Load from memory | `%val = LOAD(%ptr)` |
| `STORE` | Store to memory | `STORE(%ptr, %val)` |
| `ALLOC` | Allocate memory | `%ptr = ALLOC(size=1024)` |

### Control Flow

| Operation | Description | Example |
|-----------|-------------|---------|
| `BR` | Conditional branch | `BR(%cond, "then", "else")` |
| `JMP` | Unconditional jump | `JMP("target")` |
| `RET` | Return | `RET(%value)` |
| `CALL` | Function call | `%ret = CALL("func", %arg1, %arg2)` |

### Comparison

| Operation | Description | Example |
|-----------|-------------|---------|
| `EQ` | Equal | `%cmp = EQ(%v0, %v1)` |
| `NE` | Not equal | `%cmp = NE(%v0, %v1)` |
| `LT` | Less than | `%cmp = LT(%v0, %v1)` |
| `LE` | Less or equal | `%cmp = LE(%v0, %v1)` |
| `GT` | Greater than | `%cmp = GT(%v0, %v1)` |
| `GE` | Greater or equal | `%cmp = GE(%v0, %v1)` |

### I/O Operations (pxOS-specific)

| Operation | Description | Example |
|-----------|-------------|---------|
| `PRINT_STR` | Print string | `PRINT_STR(%ptr, %len)` |
| `DRAW_GLYPH` | Draw glyph | `DRAW_GLYPH(%glyph, %x, %y)` |
| `BLIT` | Blit pixels | `BLIT(%src, %dst, %w, %h)` |
| `READ_KEY` | Read keyboard | `%key = READ_KEY()` |

---

## IR Builder API

The `IRBuilder` class provides a convenient API for constructing pxIR programs:

```python
from pxir import IRBuilder, Type

builder = IRBuilder()

# Create entry block
entry = builder.create_block("entry")
builder.set_insert_point(entry)

# Build operations
x = builder.fresh_value(Type.f32(), "x")
y = builder.fresh_value(Type.f32(), "y")

# %result = ADD(%x, %y)
result = builder.add(x, y)

# %result2 = MUL(%result, %result)
result2 = builder.mul(result, result)

# RET(%result2)
builder.ret(result2)

# Get program
program = builder.program
print(program)
```

---

## Frontends

### Python Frontend

```python
from pxir.python_frontend import compile_python

source = """
def compute(x, y):
    result = x + y * 2
    return result
"""

program = compile_python(source, "compute")
print(program)
```

**Supported Python subset:**
- Function definitions
- Arithmetic operations (+, -, *, /, @)
- Function calls
- Return statements
- Constants (int, float, string)

**Coming soon:**
- Variables and assignments
- Loops (for, while)
- Conditionals (if/else)
- NumPy operations

### Future Frontends

- **C to pxIR** - Compile C subset to pxIR
- **Binary lifter** - Lift x86/ARM binaries to pxIR
- **LLM thoughts** - Direct semantic generation

---

## Backends

### PXI Assembly Backend

```python
from pxir.codegen_pxi import generate_pxi

# Generate PXI Assembly from pxIR
pxi_json = generate_pxi(program)

# Write to file
with open("output.pxi.json", "w") as f:
    json.dump(pxi_json, f, indent=2)

# Compile with ir_compiler.py
# $ ir_compiler.py output.pxi.json -o primitives.txt
```

### Future Backends

- **pxVM Backend** - Generate pxVM bytecode directly from pxIR
- **ARM Backend** - Generate ARM assembly
- **RISC-V Backend** - Generate RISC-V assembly
- **Interpreter** - Execute pxIR directly for testing

---

## Optimization Passes (Future)

pxIR enables standard compiler optimizations:

### Planned Passes

1. **Constant Folding**
   ```
   %v0 = ADD(10, 20) → %v0 = 30
   ```

2. **Dead Code Elimination**
   ```
   %unused = ADD(%x, %y)  → (removed)
   ```

3. **Common Subexpression Elimination**
   ```
   %v1 = ADD(%x, %y)
   %v2 = ADD(%x, %y)  → %v2 = %v1
   ```

4. **Loop Invariant Code Motion**
5. **Strength Reduction**
6. **Inlining**

---

## Complete Example

```python
from pxir import IRBuilder, Type

# Build a simple program: compute = x + y * 2
builder = IRBuilder()

entry = builder.create_block("entry")
builder.set_insert_point(entry)

# Parameters
x = Value("x", Type.f32())
y = Value("y", Type.f32())

# Constant 2
two = builder.fresh_value(Type.f32(), "const")
two.metadata = {"constant": 2.0}

# y * 2
y_times_2 = builder.mul(y, two)

# x + (y * 2)
result = builder.add(x, y_times_2)

# Return result
builder.ret(result)

# Print IR
program = builder.program
print("pxIR Program:")
print(program)

# Generate PXI Assembly
from pxir.codegen_pxi import generate_pxi
pxi = generate_pxi(program)
print("\nPXI Assembly:")
print(json.dumps(pxi, indent=2))
```

---

## Integration with Existing System

pxIR fits into the existing compilation pipeline:

```
Old Pipeline:
  Python → PXI Assembly → Primitives → Binary

New Pipeline:
  Python → pxIR → PXI Assembly → Primitives → Binary
             ↑
         Optimization
         passes here
```

Both `pxpyc2.py` (old) and `pxir.python_frontend` (new) can coexist. The new pipeline enables:
- Better optimization (at pxIR level)
- Multiple backends (pxVM, ARM, etc.)
- Matrix operations
- Type checking

---

## Future Work

### Near-Term

- [ ] Complete Python language support (loops, conditionals, variables)
- [ ] Implement optimization passes (constant folding, DCE)
- [ ] Add type inference
- [ ] Full matrix operation support

### Mid-Term

- [ ] pxVM backend (pxIR → bytecode)
- [ ] Binary lifter (x86/ELF → pxIR)
- [ ] C frontend
- [ ] Source maps for debugging

### Long-Term

- [ ] JIT compilation from pxIR
- [ ] Multiple architecture backends (ARM, RISC-V)
- [ ] Advanced optimizations (loop unrolling, vectorization)
- [ ] Neural architecture search at IR level

---

## References

- [spec.py](spec.py) - Operation and type specifications
- [ir.py](ir.py) - Core IR data structures
- [python_frontend.py](python_frontend.py) - Python → pxIR compiler
- [codegen_pxi.py](codegen_pxi.py) - pxIR → PXI Assembly backend
- [pxir_demo.py](../pxir_demo.py) - Demo and examples
- [PXI Assembly IR](../../ir/README.md) - Low-level IR documentation

---

**pxIR is the semantic foundation for pxOS compilation!** 🚀
