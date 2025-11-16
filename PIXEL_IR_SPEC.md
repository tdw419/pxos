# Pixel IR Specification v0.1

**Pixel Intermediate Representation (IR)** is the execution target for pxOS.

All code - whether authored in Python, Assembly, or future languages - compiles to Pixel IR for execution on the Pixel VM.

## Philosophy

> "Pixel IR is the truth. Languages are just different ways to express it."

- **Simple**: Stack-based, minimal opcodes
- **Substrate-native**: Designed for pixel storage and GPU cooperation
- **Target, not source**: You compile *to* Pixel IR, you don't write complex programs directly in it (though you can)

## Architecture

### Execution Model

- **Stack machine** with unlimited stack depth
- **Memory**: 1024 words (32-bit signed integers)
- **Program Counter (PC)**: Current instruction offset
- **Halt flag**: Stops execution

### Data Types

Currently supported:
- `i32`: 32-bit signed integer
- `str`: UTF-8 string (encoded as length + bytes)

Future:
- `f32`: 32-bit float
- `ptr`: Memory pointer
- `gpu_buffer`: GPU buffer handle

## Instruction Set

### Format

Each instruction is:
```
[OPCODE:1 byte] [OPERANDS:variable]
```

Multi-byte values use **little-endian** encoding.

### Opcodes

#### Stack Operations

| Opcode | Name | Operands | Description |
|--------|------|----------|-------------|
| `0x01` | `PUSH` | `i32` | Push 32-bit integer onto stack |
| `0x02` | `POP` | - | Pop and discard top of stack |
| `0x11` | `DUP` | - | Duplicate top of stack |
| `0x12` | `SWAP` | - | Swap top two stack elements |

#### Arithmetic

| Opcode | Name | Operands | Description |
|--------|------|----------|-------------|
| `0x03` | `ADD` | - | Pop b, pop a, push a + b |
| `0x04` | `SUB` | - | Pop b, pop a, push a - b |
| `0x05` | `MUL` | - | Pop b, pop a, push a * b |
| `0x06` | `DIV` | - | Pop b, pop a, push a / b |
| `0x07` | `MOD` | - | Pop b, pop a, push a % b |

#### Comparison

| Opcode | Name | Operands | Description |
|--------|------|----------|-------------|
| `0x20` | `EQ` | - | Pop b, pop a, push 1 if a == b else 0 |
| `0x21` | `LT` | - | Pop b, pop a, push 1 if a < b else 0 |
| `0x22` | `GT` | - | Pop b, pop a, push 1 if a > b else 0 |

#### Control Flow

| Opcode | Name | Operands | Description |
|--------|------|----------|-------------|
| `0x30` | `JMP` | `i32` | Jump to absolute offset |
| `0x31` | `JZ` | `i32` | Pop value, jump if zero |
| `0x32` | `JNZ` | `i32` | Pop value, jump if not zero |
| `0x33` | `CALL` | `i32` | Call subroutine at offset (push return address) |
| `0x34` | `RET` | - | Return from subroutine (pop return address) |

#### Memory

| Opcode | Name | Operands | Description |
|--------|------|----------|-------------|
| `0x40` | `LOAD` | - | Pop address, push memory[address] |
| `0x41` | `STORE` | - | Pop value, pop address, memory[address] = value |

#### I/O

| Opcode | Name | Operands | Description |
|--------|------|----------|-------------|
| `0x10` | `PRINT` | - | Pop and print integer |
| `0x13` | `PRINT_STR` | `len:i32, data:bytes` | Print string literal |

#### Host Calls

| Opcode | Name | Operands | Description |
|--------|------|----------|-------------|
| `0x50` | `HOST_CALL` | `id:u8` | Call host function by ID |

**Standard Host Functions:**

| ID | Name | Stack Effect | Description |
|----|------|--------------|-------------|
| `0` | `print_stack` | `[value] → []` | Print value from stack |
| `1` | `read_pixel` | `[x, y] → [r, g, b]` | Read pixel from PixelFS |
| `2` | `write_pixel` | `[x, y, r, g, b] → []` | Write pixel to PixelFS |
| `3` | `gpu_compute` | `[buffer_id, size] → [buffer_id]` | Dispatch GPU kernel |

#### System

| Opcode | Name | Operands | Description |
|--------|------|----------|-------------|
| `0xFF` | `HALT` | - | Stop execution |
| `0xFE` | `NOP` | - | No operation |

## Encoding Examples

### Simple Program
```
Print 42:
  PUSH 42       → 01 2A 00 00 00
  PRINT         → 10
  HALT          → FF
```

### Arithmetic
```
Compute 5 + 3:
  PUSH 5        → 01 05 00 00 00
  PUSH 3        → 01 03 00 00 00
  ADD           → 03
  PRINT         → 10
  HALT          → FF
```

### Loop (count down from 5)
```
Address  | Instruction
---------|-------------
0x00:    | PUSH 5           (01 05 00 00 00)
0x05:    | DUP              (11)              ; [loop_start]
0x06:    | PRINT            (10)
0x07:    | PUSH 1           (01 01 00 00 00)
0x0C:    | SUB              (04)
0x0D:    | DUP              (11)
0x0E:    | JNZ 0x05         (32 05 00 00 00)  ; jump if not zero
0x13:    | HALT             (FF)
```

## Module System

### Module Header

Pixel IR programs can include a module header:

```
[MAGIC: "PXIR"] [VERSION: u16] [FLAGS: u16]
[EXPORTS_COUNT: u16]
  [EXPORT_NAME_LEN: u8] [EXPORT_NAME: bytes] [EXPORT_OFFSET: i32]
  ...
[IMPORTS_COUNT: u16]
  [IMPORT_NAME_LEN: u8] [IMPORT_NAME: bytes]
  ...
[CODE_START: marker]
[... bytecode ...]
```

### Linking

- **Exports**: Functions this module provides (name → offset)
- **Imports**: Functions this module needs (resolved at load time)
- **Module calls**: `CALL` to imported function resolves to:
  - Load target module if needed
  - Jump to exported offset

## Compilation Targets

### From Python
```python
def add_two(x):
    return x + 2

# Compiles to:
# (assuming x passed on stack)
PUSH 2
ADD
RET
```

### From Pixel Assembly
```asm
; Function: add_two
; Stack: [x] → [x+2]
add_two:
    PUSH 2
    ADD
    RET
```

## Extension Points

Future additions:

1. **Floating point**: `PUSH_F32`, `ADD_F32`, etc.
2. **GPU integration**: Direct GPU buffer opcodes
3. **Pixel operations**: Native pixel read/write/blend
4. **Structured data**: Arrays, structs via memory
5. **Async/concurrency**: Fiber/coroutine support

## Tooling

Standard tools for Pixel IR:

- **Assembler**: `.pxasm` → `.pxi`
- **Disassembler**: `.pxi` → `.pxasm`
- **Linker**: Multiple `.pxi` → one linked `.pxi`
- **Validator**: Check bytecode well-formedness
- **Debugger**: Step through execution, inspect stack/memory

## Version History

- **v0.1** (2025-01): Initial specification
  - Basic stack operations
  - Arithmetic and control flow
  - Host call interface
  - Simple module system
