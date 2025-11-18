# PXI Assembly - pxOS Intermediate Representation

**Version**: 1.0
**Status**: Production

---

## Overview

**PXI Assembly** is the universal intermediate representation (IR) for pxOS. It serves as the abstraction layer between high-level source languages and low-level primitives.

### The Critical Layer

```
┌──────────────────────────────────────────────────────────┐
│  Source Languages (Python, C, etc.)                       │
└────────────────────┬──────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│  PXI Assembly IR (THIS LAYER)                            │
│  - Human-readable                                         │
│  - Debuggable                                             │
│  - Architecture-agnostic semantics                        │
│  - LLM-friendly                                           │
└────────────────────┬──────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│  ir_compiler.py (Mechanical translation)                  │
│  - ONLY module that knows x86 opcodes                     │
│  - Provable correctness                                   │
│  - Address resolution                                     │
└────────────────────┬──────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│  pxOS Primitives (WRITE/DEFINE commands)                 │
└────────────────────┬──────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│  Binary (pxos.bin)                                        │
└──────────────────────────────────────────────────────────┘
```

---

## Why PXI Assembly Exists

### The Problem Without IR

In the v1 Python compiler, we went directly from Python AST → x86 opcodes:

```python
# BAD: Compiler knows about x86 opcodes
def gen_print_text(self, text):
    addr = self.ctx.emit_bytes(addr, [0xBE, low, high], "MOV SI")
    # What if the opcode is wrong?
    # What if address calculation is off?
    # How do you debug it?
```

Issues:
- ❌ **Hard to debug** - opcode generation bugs look like logic bugs
- ❌ **Not portable** - tied to x86
- ❌ **Every language needs x86 knowledge** - C compiler, assembly, all need opcodes
- ❌ **LLMs generate wrong opcodes** - hard to verify

### The Solution: IR Layer

With PXI Assembly:

```python
# GOOD: Compiler emits semantic operations
def gen_print_text(self, text):
    self.emit_instr("MOV", dst="SI", src="str_label")
    self.emit_instr("CALL", target="print_string")
    # ↑ Human-readable, debuggable, verifiable
```

Benefits:
- ✅ **Debuggable** - can inspect IR, verify logic
- ✅ **Portable** - IR is semantic, not arch-specific
- ✅ **Separation of concerns** - language compiler vs. code generator
- ✅ **LLM-friendly** - can generate/analyze IR directly
- ✅ **Mechanical backend** - ir_compiler.py is provably correct

---

## PXI Assembly Specification

See [pxi_asm_v1.json](pxi_asm_v1.json) for the complete specification.

### Basic Structure

PXI Assembly programs consist of:

1. **Instructions** - assembly-like operations
2. **Labels** - code and data labels
3. **Data** - strings, bytes, words

### Example Program

```assembly
; Hello World in PXI Assembly
ORG 0x7C00

FUNC main
  SYSCALL clear_screen
  MOV SI, str_hello
  CALL print_string
  JMP halt
ENDFUNC

FUNC print_string
.loop:
  LODSB
  OR AL, AL
  JZ .done
  MOV AH, 0x0E
  INT 0x10
  JMP .loop
.done:
  RET
ENDFUNC

halt:
  JMP halt

str_hello:
  STRING "Hello from PXI Assembly!"
```

### JSON Format

IR can also be represented as JSON for programmatic generation:

```json
{
  "source": "hello.py",
  "ir_version": "1.0",
  "origin": 31744,
  "instructions": [
    {
      "op": "MOV",
      "operands": {"dst": "AX", "src": 47104},
      "comment": "Set ES to VGA buffer"
    },
    {
      "op": "MOV",
      "operands": {"dst": "ES", "src": "AX"},
      "comment": ""
    }
  ],
  "data": [
    {
      "label": "str_0",
      "type": "string",
      "value": "Hello World!"
    }
  ]
}
```

---

## Tools

### 1. Python Compiler v2 (`pxpyc2.py`)

Compiles Python → PXI Assembly IR:

```bash
pxpyc2.py hello.py --show-ir   # Show IR
pxpyc2.py hello.py --build     # Compile to binary
```

**Architecture**:
- Parses Python AST
- Generates semantic PXI Assembly instructions
- NO opcode knowledge
- NO address calculation
- Just emits what the program does

### 2. IR Compiler (`ir_compiler.py`)

Compiles PXI Assembly IR → primitives:

```bash
ir_compiler.py program.pxi.json -o primitives.txt
```

**Architecture**:
- Parses IR (JSON or text format)
- Resolves label addresses
- Generates x86 opcodes
- ONLY module with x86 knowledge

### 3. Pixel Cartridges (Future)

Cartridges will store IR + binary:

```
.pxcart.png:
  ├── Header (metadata)
  ├── IR section (PXI Assembly JSON)
  ├── Binary section (compiled code)
  └── Checksum
```

This makes cartridges truly portable - the IR can be recompiled for any target architecture.

---

## Instruction Reference

### Data Movement

| Instruction | Description | Example |
|-------------|-------------|---------|
| `MOV dst, src` | Move data | `MOV AX, 0x1234` |
| `PUSH src` | Push to stack | `PUSH AX` |
| `POP dst` | Pop from stack | `POP BX` |
| `LEA dst, src` | Load effective address | `LEA SI, [BP+4]` |

### Arithmetic

| Instruction | Description | Example |
|-------------|-------------|---------|
| `ADD dst, src` | Add | `ADD AX, 10` |
| `SUB dst, src` | Subtract | `SUB CX, 5` |
| `INC dst` | Increment | `INC CX` |
| `DEC dst` | Decrement | `DEC BX` |
| `MUL src` | Multiply | `MUL BX` |
| `DIV src` | Divide | `DIV CX` |

### Logical

| Instruction | Description | Example |
|-------------|-------------|---------|
| `AND dst, src` | Bitwise AND | `AND AX, 0xFF` |
| `OR dst, src` | Bitwise OR | `OR AL, AL` |
| `XOR dst, src` | Bitwise XOR | `XOR DI, DI` |
| `NOT dst` | Bitwise NOT | `NOT AX` |
| `SHL dst, count` | Shift left | `SHL AX, 2` |
| `SHR dst, count` | Shift right | `SHR BX, 1` |

### Comparison

| Instruction | Description | Example |
|-------------|-------------|---------|
| `CMP op1, op2` | Compare | `CMP AL, 0` |
| `TEST op1, op2` | Test (AND without storing) | `TEST AL, AL` |

### Control Flow

| Instruction | Description | Example |
|-------------|-------------|---------|
| `JMP target` | Unconditional jump | `JMP main_loop` |
| `JE/JZ target` | Jump if equal/zero | `JE .done` |
| `JNE/JNZ target` | Jump if not equal/zero | `JNE .loop` |
| `JG target` | Jump if greater (signed) | `JG .positive` |
| `JL target` | Jump if less (signed) | `JL .negative` |
| `CALL target` | Call subroutine | `CALL print_string` |
| `RET` | Return | `RET` |

### String Operations

| Instruction | Description | Example |
|-------------|-------------|---------|
| `LODSB` | Load byte at DS:SI to AL | `LODSB` |
| `STOSB` | Store AL at ES:DI | `STOSB` |
| `STOSW` | Store AX at ES:DI | `STOSW` |
| `REP` | Repeat next instruction CX times | `REP STOSW` |

### System

| Instruction | Description | Example |
|-------------|-------------|---------|
| `INT num` | Software interrupt | `INT 0x10` |
| `CLI` | Disable interrupts | `CLI` |
| `STI` | Enable interrupts | `STI` |
| `HLT` | Halt CPU | `HLT` |
| `NOP` | No operation | `NOP` |

---

## Workflow Examples

### Python → PXI → Primitives → Binary

```bash
# Step 1: Python → PXI Assembly IR
pxpyc2.py hello.py -o hello.pxi.json

# Step 2: PXI → Primitives
ir_compiler.py hello.pxi.json -o primitives.txt

# Step 3: Primitives → Binary
build_pxos.py

# Step 4: Test
qemu-system-i386 -fda pxos.bin
```

### Or, all at once:

```bash
pxpyc2.py hello.py --run
```

### Debugging IR

```bash
# Generate and inspect IR
pxpyc2.py hello.py --show-ir

# You'll see clean, readable JSON:
{
  "instructions": [
    {"op": "MOV", "operands": {"dst": "SI", "src": "str_0"}},
    {"op": "CALL", "operands": {"target": "print_string"}}
  ]
}

# Now you can verify:
# - Is the logic correct? ✓
# - Are labels referenced properly? ✓
# - Is the control flow right? ✓
```

---

## Benefits of the IR Layer

### 1. **Debuggability**

**Without IR**:
```
Triple fault! Is it:
  - Wrong opcode?
  - Bad address calculation?
  - Logic error in Python?
  - Stack corruption?
→ NO IDEA!
```

**With IR**:
```
1. Check IR: Does the logic make sense? YES
2. Check primitives: Are opcodes correct? NO! MOV is wrong
3. Fix ir_compiler.py
→ PROBLEM IDENTIFIED!
```

### 2. **Portability**

PXI Assembly is semantic, not tied to x86:

```json
{"op": "MOV", "operands": {"dst": "SI", "src": "str_hello"}}
```

This can compile to:
- **x86**: `MOV SI, 0x7E00` → `BE 00 7E`
- **ARM**: `LDR R0, =str_hello` → Different opcodes
- **RISC-V**: `LA a0, str_hello` → Different again
- **pxVM**: `LOAD_ADDR r0, str_hello` → Bytecode

### 3. **LLM-Friendly**

LLMs can generate IR directly:

```
User: "Write a program that clears the screen"

LLM generates:
{
  "instructions": [
    {"op": "SYSCALL", "operands": {"name": "clear_screen"}}
  ]
}
```

No need to know x86 opcodes!

### 4. **Multiple Languages**

All languages target the same IR:

```
Python  ──┐
C       ──┼──→  PXI Assembly  ──→  ir_compiler.py  ──→  Primitives
Assembly──┘
```

### 5. **Pixel Cartridges**

Store IR in cartridges for true portability:

```
Binary (ELF/PE/etc.) → Lifter → PXI Assembly IR → Store in cartridge

Later:
Cartridge → Extract IR → Compile for pxOS → Native code
```

---

## Implementation Notes

### Address Resolution

`ir_compiler.py` uses a two-pass approach:

**Pass 1**: Calculate addresses
- Iterate through instructions
- Calculate size of each instruction
- Assign addresses to all labels

**Pass 2**: Generate opcodes
- Resolve label references to addresses
- Generate final x86 opcodes
- Emit primitives

### Jump Offset Calculation

```python
# For: CALL print_string
target_addr = label_addresses["print_string"]  # e.g., 0x7C20
current_addr = 0x7C10
next_addr = current_addr + 3  # CALL is 3 bytes
offset = target_addr - next_addr  # Relative offset
opcodes = [0xE8, offset & 0xFF, (offset >> 8) & 0xFF]
```

### Label Types

1. **Code labels**: Function entry points, loop labels
2. **Data labels**: Strings, constants
3. **Local labels**: Start with `.` (e.g., `.loop`, `.done`)

---

## Future Extensions

### 1. Optimization Passes

```
IR → Dead Code Elimination → Register Allocation → Constant Folding → Optimized IR
```

### 2. Multiple Backends

```
                      ┌→ x86 Backend → x86 primitives
PXI Assembly IR  ──→  ├→ ARM Backend → ARM code
                      └→ pxVM Backend → Bytecode
```

### 3. Source Maps

Link IR back to original source for debugging:

```json
{
  "op": "MOV",
  "source_line": 15,
  "source_file": "hello.py",
  "source_code": "print_text('Hello')"
}
```

### 4. Type System

Add type annotations to IR:

```json
{
  "op": "MOV",
  "operands": {"dst": {"reg": "SI", "type": "ptr<char>"}, "src": "str_0"}
}
```

---

## See Also

- [pxi_asm_v1.json](pxi_asm_v1.json) - Complete specification
- [../tools/pxpyc2.py](../tools/pxpyc2.py) - Python → IR compiler
- [../tools/ir_compiler.py](../tools/ir_compiler.py) - IR → Primitives compiler
- [../PYTHON_ROADMAP.md](../PYTHON_ROADMAP.md) - Development roadmap

---

**The IR layer is the foundation of pxOS's portability and debuggability!** 🎯
