# GPU Primitive Specification v0.1

## Overview

This document defines the primitive syntax for GPU programming in the pxOS heterogeneous computing system.

**Design Goals:**
- Simple, readable syntax
- Maps cleanly to CUDA/PTX/OpenCL
- Compatible with existing pxOS CPU primitives
- LLM-friendly (clear intent)

---

## Primitive Commands

### GPU_KERNEL - Define GPU Kernel

```
GPU_KERNEL <kernel_name>
```

Starts a GPU kernel definition. Must be followed by parameters and thread code.

**Example:**
```
GPU_KERNEL vector_add
```

---

### GPU_PARAM - Kernel Parameter

```
GPU_PARAM <name> <type>
```

Defines an input/output parameter for the kernel.

**Supported Types:**
- `int` - 32-bit integer
- `float` - 32-bit float
- `int[]` - Integer array (pointer)
- `float[]` - Float array (pointer)
- `int*` - Integer pointer (explicit)

**Example:**
```
GPU_PARAM input_a float[]
GPU_PARAM input_b float[]
GPU_PARAM output float[]
GPU_PARAM size int
```

---

### GPU_SHARED - Shared Memory

```
GPU_SHARED <name> <type> <size>
```

Declares shared memory visible to all threads in a block.

**Example:**
```
GPU_SHARED cache float 256
```

Maps to: `__shared__ float cache[256];`

---

### GPU_THREAD_CODE - Thread Execution Body

```
GPU_THREAD_CODE:
    <operations>
GPU_END
```

Defines what each GPU thread executes.

---

## Operations (Inside GPU_THREAD_CODE)

### THREAD_ID - Get Thread Index

```
THREAD_ID → <variable>
```

Gets the global thread ID.

**Example:**
```
THREAD_ID → tid
```

Maps to: `int tid = blockIdx.x * blockDim.x + threadIdx.x;`

---

### BLOCK_ID - Get Block Index

```
BLOCK_ID → <variable>
```

Gets the block ID.

**Example:**
```
BLOCK_ID → bid
```

Maps to: `int bid = blockIdx.x;`

---

### LOCAL_ID - Get Thread ID Within Block

```
LOCAL_ID → <variable>
```

Gets the thread ID within its block.

**Example:**
```
LOCAL_ID → ltid
```

Maps to: `int ltid = threadIdx.x;`

---

### LOAD - Read from Memory

```
LOAD <source> → <destination>
```

Loads a value from global memory into a register.

**Example:**
```
LOAD input_a[tid] → val_a
```

Maps to: `float val_a = input_a[tid];`

---

### STORE - Write to Memory

```
STORE <source> → <destination>
```

Stores a register value to global memory.

**Example:**
```
STORE result → output[tid]
```

Maps to: `output[tid] = result;`

---

### Arithmetic Operations

```
ADD <a> <b> → <result>
SUB <a> <b> → <result>
MUL <a> <b> → <result>
DIV <a> <b> → <result>
MOD <a> <b> → <result>
```

**Example:**
```
ADD val_a val_b → sum
MUL sum 2 → doubled
```

---

### Comparison Operations

```
COMPARE <a> <op> <b> → <result>
```

**Operators:** `<`, `>`, `==`, `!=`, `<=`, `>=`

**Example:**
```
COMPARE tid < size → in_bounds
```

Maps to: `bool in_bounds = (tid < size);`

---

### Control Flow

```
IF <condition>:
    <operations>
ENDIF

IF <condition>:
    <operations>
ELSE:
    <operations>
ENDIF
```

**Example:**
```
IF in_bounds:
    LOAD input[tid] → val
    STORE val → output[tid]
ENDIF
```

---

### Synchronization

```
SYNC_THREADS
```

Synchronizes all threads in a block.

Maps to: `__syncthreads();`

**Example:**
```
STORE local_data → shared_mem[ltid]
SYNC_THREADS
LOAD shared_mem[255 - ltid] → reversed_data
```

---

## Kernel Launch

### GPU_LAUNCH - Execute Kernel

```
GPU_LAUNCH <kernel_name> BLOCKS <num_blocks> THREADS <threads_per_block>
```

**Example:**
```
GPU_LAUNCH vector_add BLOCKS 256 THREADS 256
```

Maps to: `vector_add<<<256, 256>>>(...);`

---

### GPU_SYNC - Wait for Completion

```
GPU_SYNC
```

Waits for all GPU operations to complete.

Maps to: `cudaDeviceSynchronize();`

---

## Complete Example: Vector Addition

```
# Define kernel
GPU_KERNEL vector_add
GPU_PARAM a float[]
GPU_PARAM b float[]
GPU_PARAM c float[]
GPU_PARAM n int

GPU_THREAD_CODE:
    # Get thread ID
    THREAD_ID → tid

    # Bounds check
    COMPARE tid < n → in_bounds

    IF in_bounds:
        # Load inputs
        LOAD a[tid] → val_a
        LOAD b[tid] → val_b

        # Compute
        ADD val_a val_b → sum

        # Store result
        STORE sum → c[tid]
    ENDIF
GPU_END

# Launch kernel
GPU_LAUNCH vector_add BLOCKS 256 THREADS 256
GPU_SYNC
```

---

## CUDA Mapping

The above primitives map to this CUDA C code:

```c
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool in_bounds = (tid < n);

    if (in_bounds) {
        float val_a = a[tid];
        float val_b = b[tid];
        float sum = val_a + val_b;
        c[tid] = sum;
    }
}

// Launch
vector_add<<<256, 256>>>(a, b, c, n);
cudaDeviceSynchronize();
```

---

## Integration with CPU Primitives

CPU and GPU primitives coexist in the same file:

```
# CPU code (pxOS style)
WRITE 0x7C00 0xB4    COMMENT mov ah, 0x0E
CALL setup_gpu_driver

# GPU code (new)
GPU_KERNEL render_text
GPU_PARAM text char[]
GPU_THREAD_CODE:
    THREAD_ID → tid
    LOAD text[tid] → char
    # ... render logic
GPU_END

# Launch from CPU
GPU_LAUNCH render_text BLOCKS 32 THREADS 32
```

---

## Future Extensions

### Memory Management
```
GPU_ALLOC <name> <type> <size>
GPU_COPY_TO_GPU <cpu_ptr> → <gpu_ptr> <size>
GPU_COPY_FROM_GPU <gpu_ptr> → <cpu_ptr> <size>
GPU_FREE <name>
```

### Advanced Operations
```
ATOMIC_ADD <address> <value>
WARP_REDUCE SUM <value> → <result>
BLOCK_REDUCE MAX <value> → <result>
```

### Multiple GPUs
```
GPU_SELECT 0
GPU_LAUNCH kernel BLOCKS 100 THREADS 256
```

---

## Design Rationale

### Why Arrow Notation `→`?

```
LOAD array[i] → value     # Clear: array goes INTO value
STORE value → array[i]    # Clear: value goes INTO array
```

Alternative considered:
```
value = array[i]          # Looks like assignment, but it's not
array[i] = value          # Confusing - is this CPU or GPU?
```

**Arrow makes data flow explicit and readable.**

### Why Uppercase Commands?

- Consistent with pxOS CPU primitives (`WRITE`, `DEFINE`)
- Visually distinct from variables
- Easy for parser to tokenize
- LLM-friendly (clear command identification)

### Why Explicit Types?

```
GPU_PARAM data float[]    # Clear: array of floats
```

Helps:
- Code generation (know what CUDA type to use)
- LLM understanding (knows data structure)
- Error checking (type mismatches caught early)

---

## Version History

- **v0.1** (2025-11-19): Initial specification
  - Basic kernel definition
  - Thread operations
  - Arithmetic and control flow
  - Kernel launch syntax

---

**Status:** Draft for implementation
**Next Step:** Build parser for these primitives
