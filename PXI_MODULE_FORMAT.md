# PXI Module Format

**Pixel-native modules for self-hosting pxOS**

## Overview

The PXI Module Format allows Python code to be compiled into pixel-native modules that run directly on PXI_CPU with no host dependencies.

**Goal**: Eliminate Python from the runtime. Everything runs as pixels.

## Module Structure

### Module Header (First row of pixels)

```
Pixel 0: MODULE_MAGIC = RGBA(0x50, 0x58, 0x49, 0x4D)  # "PXIM"
Pixel 1: Module version (major, minor, patch, flags)
Pixel 2-3: Entry point address (16-bit)
Pixel 4-5: Module size in pixels (16-bit)
Pixel 6-15: Reserved for future use
```

### Module Table (Pixel row 1)

List of function entry points within the module:

```
Each entry = 4 pixels:
  Pixel N+0: Function ID (0-255)
  Pixel N+1: Name hash (for debugging)
  Pixel N+2-3: Entry point address (16-bit)
```

Example:
```
Function 0 (compress):   ID=0, Entry=256
Function 1 (decompress): ID=1, Entry=512
Function 2 (hash):       ID=2, Entry=768
```

### Module Code (Pixel row 16+)

Actual PXI instructions starting at row 16.

Each instruction = 1 pixel (RGBA = opcode, arg1, arg2, arg3)

## Calling Convention

### Function Call Protocol

```assembly
; Caller:
LOAD R0, arg1          ; First argument
LOAD R1, arg2          ; Second argument
CALL function_addr     ; Push return address, jump
; Return value in R0

; Callee (function):
PUSH R2                ; Save registers we'll use
PUSH R3
; ... function body ...
LOAD R0, return_value  ; Return value
POP R3                 ; Restore registers
POP R2
RET                    ; Pop return address, jump back
```

### Register Usage

| Register | Purpose | Preserved? |
|----------|---------|------------|
| R0 | Return value / First arg | No |
| R1 | Second arg | No |
| R2 | Third arg | No |
| R3-R7 | Caller-saved (scratch) | No |
| R8-R15 | Callee-saved | Yes |

### Stack

The stack grows downward in memory. CALL/RET use the CPU's internal stack.

For local variables, use PUSH/POP:

```assembly
PUSH R8    ; Save R8
LOAD R8, local_value
; ... use R8 ...
POP R8     ; Restore R8
```

## Standard Modules

### Module 0: Bootstrap

Entry point functions:
- `boot_from_pixel` - Main entry point
- `decompress_god_pixel` - Decompress a God Pixel
- `hash_compute` - Compute SHA256 hash

### Module 1: Compression (Future)

- `compress` - Compress data
- `decompress` - Decompress data

### Module 2: Registry (Future)

- `lookup_world` - Find world by color
- `register_world` - Add world to registry

### Module 3: Oracle Router (Future)

- `route_oracle_request` - Handle organism requests
- `call_llm` - Wrapper for SYS_LLM

## Python → PXI Compilation

### Supported Python Subset (v1)

```python
# Arithmetic
a = b + c
a = b - c
a = b * c  # (mult via repeated addition)

# Control flow
if condition:
    ...
else:
    ...

while condition:
    ...

# Functions
def add(a, b):
    return a + b

# Variables (16 max, map to registers)
x = 10
y = 20
```

### Example Compilation

Python:
```python
def add_two_numbers(a, b):
    return a + b
```

PXI:
```assembly
; Function: add_two_numbers
; Entry point: 256
; Args: R0=a, R1=b
; Returns: R0=result

add_two_numbers:
    ADD R0, R0, R1    ; R0 = R0 + R1
    RET               ; Return
```

As pixels:
```
Pixel 256: (0x20, 0, 0, 1)    ; ADD R0, R0, R1
Pixel 257: (0x33, 0, 0, 0)    ; RET
```

## Compiler Pipeline

```
Python source
   ↓
Python AST (ast.parse)
   ↓
Simplified IR (variables → registers)
   ↓
PXI assembly
   ↓
PXI bytecode (pixels)
   ↓
Module image (PNG)
   ↓
Compress → God Pixel
```

## Module Loader

When PXI_CPU boots a module image:

1. Check MODULE_MAGIC at pixel 0
2. Read entry point from pixels 2-3
3. Parse module table (row 1)
4. Jump to entry point
5. Module can call other functions via CALL

## Self-Hosting Path

### Phase 1 (Current): Python host + PXI guests
- Python: bootloader, compression, registry
- PXI: universes, organisms

### Phase 2: Mixed host
- Python: bootloader, compiler
- PXI: compression, registry, oracle router

### Phase 3: Minimal Python stub
- Python: 50-line bootstrap only
- PXI: everything else

### Phase 4: Pure PXI
- Python: nothing (or optional tooling)
- PXI: entire OS, including compiler

## Example: Full Module

```python
# create_compression_module.py

from PIL import Image
from pxi_cpu import *

def create_compression_module():
    """Create a PXI module for compression"""

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 255))

    def set_pixel(pc, r, g, b, a=0):
        x = pc % 64
        y = pc // 64
        img.putpixel((x, y), (r, g, b, a))

    # Header
    set_pixel(0, 0x50, 0x58, 0x49, 0x4D)  # "PXIM" magic
    set_pixel(1, 1, 0, 0, 0)               # Version 1.0.0
    set_pixel(2, 0, 16, 0, 0)              # Entry point = 16
    set_pixel(3, 0, 64, 0, 0)              # Module size = 64 pixels

    # Module table
    # Function 0: compress, entry=256
    set_pixel(64, 0, 0, 0, 0)              # ID=0
    set_pixel(65, 0xAB, 0xCD, 0, 0)        # Name hash
    set_pixel(66, 0, 1, 0, 0)              # Entry = 256

    # Code starts at pixel 256
    # Simple RLE compression stub
    pc = 256
    # ... (compression logic) ...
    set_pixel(pc, OP_RET, 0, 0, 0)

    return img

module = create_compression_module()
module.save("module_compression.pxi.png")
```

## Benefits

1. **No external dependencies** - Pure pixel execution
2. **Universal** - Any PXI_CPU can run any module
3. **Compressible** - Modules are images → God Pixels
4. **Eternal** - Modules are data, not code
5. **Self-hosting** - Compiler itself becomes a module

## Roadmap

- ✅ Phase 3a: CALL/RET/PUSH/POP opcodes
- 🔄 Phase 3b: Module format specification (this doc)
- ⏳ Phase 3c: Python → PXI translator (basic)
- ⏳ Phase 3d: Compression module in PXI
- ⏳ Phase 3e: Registry module in PXI
- ⏳ Phase 3f: Bootloader module in PXI

**End goal**: Boot pixel resurrects bootloader module → bootloader module resurrects everything else.

All in pixels. Forever.

---

**The pixels will compile themselves.**
