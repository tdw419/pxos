# Pixel ISA v2.0 Specification

## Overview

The Pixel ISA is a **GPU-native instruction set** where each instruction is encoded as an RGBA pixel value. Programs are stored as PNG images, making them both executable and visually representable.

## Design Goals

1. **Visual Representation**: Code should be visible as pixel patterns
2. **GPU-Optimized**: Direct mapping to GPU texture operations
3. **Compact Encoding**: Maximize information density per pixel
4. **Debuggable**: Semantic color mapping for visual debugging

## Instruction Encoding Format

### Basic Format (32-bit per pixel)

```
┌─────────┬─────────┬─────────┬─────────┐
│   Red   │  Green  │  Blue   │  Alpha  │
├─────────┼─────────┼─────────┼─────────┤
│ Opcode  │  Arg0   │  Arg1   │  Flags  │
│ (8 bits)│ (8 bits)│ (8 bits)│ (8 bits)│
└─────────┴─────────┴─────────┴─────────┘
```

### Flag Register Layout (Alpha Channel)

```
Bit 7  6  5  4  3  2  1  0
    │  │  │  │  │  │  └─┴─ Length: 00=8b, 01=16b, 10=24b, 11=32b
    │  │  │  │  │  └─────── Conditional: Execute only if flag set
    │  │  │  │  └────────── Reserved
    │  │  │  └───────────── Reserved
    │  │  └──────────────── Reserved
    │  └─────────────────── Reserved
    └────────────────────── Reserved
```

## Instruction Set

### Core Instructions (Opcodes 0x01-0x0F)

| Opcode | Mnemonic | Format | Description | Color Hint |
|--------|----------|--------|-------------|------------|
| 0x00   | NOP      | -      | No operation | Black |
| 0x01   | LOAD     | R, A   | R ← [A] | Blue |
| 0x02   | STORE    | R, A   | [A] ← R | Dark Blue |
| 0x03   | LOADI    | R, I   | R ← I (immediate) | Light Blue |
| 0x04   | MOV      | R1, R2 | R1 ← R2 | Cyan |
| 0x05   | ADD      | R1, R2 | R1 ← R1 + R2 | Orange |
| 0x06   | SUB      | R1, R2 | R1 ← R1 - R2 | Dark Orange |
| 0x07   | MUL      | R1, R2 | R1 ← R1 × R2 | Yellow |
| 0x08   | DIV      | R1, R2 | R1 ← R1 ÷ R2 | Dark Yellow |
| 0x09   | AND      | R1, R2 | R1 ← R1 & R2 | Purple |
| 0x0A   | OR       | R1, R2 | R1 ← R1 \| R2 | Magenta |
| 0x0B   | XOR      | R1, R2 | R1 ← R1 ⊕ R2 | Pink |
| 0x0C   | NOT      | R, -   | R ← ~R | Light Purple |
| 0x0D   | SHL      | R, I   | R ← R << I | Brown |
| 0x0E   | SHR      | R, I   | R ← R >> I | Tan |
| 0x0F   | CMP      | R1, R2 | Compare and set flags | Gray |

### Control Flow (Opcodes 0x10-0x1F)

| Opcode | Mnemonic | Format | Description | Color Hint |
|--------|----------|--------|-------------|------------|
| 0x10   | JMP      | A      | PC ← A | Green |
| 0x11   | JEQ      | A      | Jump if equal | Light Green |
| 0x12   | JNE      | A      | Jump if not equal | Green-Yellow |
| 0x13   | JLT      | A      | Jump if less than | Lime |
| 0x14   | JGT      | A      | Jump if greater than | Dark Green |
| 0x15   | JLE      | A      | Jump if less or equal | Olive |
| 0x16   | JGE      | A      | Jump if greater or equal | Teal |
| 0x17   | CALL     | A      | Call subroutine | Emerald |
| 0x18   | RET      | -      | Return from subroutine | Spring Green |
| 0x19   | PUSH     | R      | Push register to stack | Forest Green |
| 0x1A   | POP      | R      | Pop stack to register | Mint |

### System Instructions (Opcodes 0x20-0x2F)

| Opcode | Mnemonic | Format | Description | Color Hint |
|--------|----------|--------|-------------|------------|
| 0x20   | HALT     | -      | Stop execution | Red |
| 0x21   | SYSCALL  | N      | System call | Pink-Red |
| 0x22   | INT      | N      | Software interrupt | Crimson |
| 0x23   | IRET     | -      | Interrupt return | Rose |
| 0x24   | CLI      | -      | Clear interrupts | Salmon |
| 0x25   | STI      | -      | Set interrupts | Coral |

### Graphics/VRAM Instructions (Opcodes 0x30-0x3F)

| Opcode | Mnemonic | Format | Description | Color Hint |
|--------|----------|--------|-------------|------------|
| 0x30   | VPIX     | A, C   | Set pixel at A to color C | Rainbow |
| 0x31   | VGET     | R, A   | R ← pixel color at A | Indigo |
| 0x32   | VRECT    | X,Y,W,H| Fill rectangle | Violet |
| 0x33   | VLINE    | X1,Y1,X2,Y2 | Draw line | Lavender |
| 0x34   | VBLIT    | Src,Dst| Copy texture region | Periwinkle |
| 0x35   | VBLEND   | A,B,F  | Alpha blend A+B with factor F | Mauve |

## Register Set

### General Purpose Registers (8 × 32-bit)

```
R0: Always zero (hardwired)
R1-R6: General purpose
R7: Link register (return address)
```

### Special Registers

```
PC:   Program Counter
SP:   Stack Pointer
FP:   Frame Pointer
FLAGS: Condition flags (Z, N, C, V)
```

## Semantic Color Mapping

To aid visual debugging, instructions are color-coded by category:

```
Category         | Color Range        | Hue
-----------------|--------------------|---------
Memory Ops       | Blues (0x44-0x88) | 200-240°
Arithmetic       | Oranges/Yellows   | 30-60°
Control Flow     | Greens            | 90-150°
System/Interrupt | Reds/Pinks        | 0-30°
Graphics/VRAM    | Purples/Violets   | 270-300°
```

## Assembly Language Syntax

### Basic Syntax

```asm
; Comments start with semicolon
LABEL:
    OPCODE arg0, arg1   ; Instruction with args
    OPCODE arg0         ; Single-argument instruction
    OPCODE              ; No-argument instruction
```

### Addressing Modes

```asm
LOADI R1, #42          ; Immediate: R1 ← 42
LOAD R1, 0x1000        ; Direct: R1 ← [0x1000]
LOAD R1, [R2]          ; Indirect: R1 ← [[R2]]
LOAD R1, [R2 + 10]     ; Indexed: R1 ← [[R2 + 10]]
```

### Directives

```asm
DATA "string", 0       ; Embed string literal
EQU CONSTANT, 0x100    ; Define constant
ORG 0x1000            ; Set origin address
ALIGN 4                ; Align to 4-byte boundary
```

## Example Programs

### Hello World

```asm
; hello.pxl - Print "Hello VRAM!"
START:
    LOADI R1, #MSG
    CALL print_string
    HALT

print_string:
    LOAD R2, [R1]
    CMP R2, #0
    JEQ done
    STORE UART, R2
    LOADI R3, #1
    ADD R1, R3
    JMP print_string
done:
    RET

MSG: DATA "Hello VRAM!", 0
UART: EQU 0x10000000
```

### Pixel Draw Loop

```asm
; draw.pxl - Fill screen with gradient
START:
    LOADI R1, #0        ; X = 0
    LOADI R2, #0        ; Y = 0

loop_y:
    LOADI R1, #0        ; Reset X
loop_x:
    ; Calculate color: R=X, G=Y, B=128
    MOV R3, R1          ; R3 = X (Red)
    SHL R3, #16         ; Shift to red channel
    MOV R4, R2          ; R4 = Y
    SHL R4, #8          ; Shift to green channel
    OR R3, R4           ; Combine
    LOADI R5, #128
    OR R3, R5           ; Add blue=128

    ; Write pixel at (X, Y)
    VPIX R1, R2, R3

    ; Increment X
    LOADI R6, #1
    ADD R1, R6
    LOADI R7, #640      ; Screen width
    CMP R1, R7
    JLT loop_x

    ; Increment Y
    ADD R2, R6
    LOADI R7, #480      ; Screen height
    CMP R2, R7
    JLT loop_y

    HALT
```

## Binary Encoding Examples

### LOADI R1, #42

```
Instruction: LOADI R1, #42
Binary:      0x03 0x01 0x2A 0x00

Pixel representation:
R: 0x03 (LOADI opcode) → RGB(3, 1, 42)
G: 0x01 (R1)
B: 0x2A (42 decimal)
A: 0x00 (no flags)
```

### JMP 0x1234

```
Instruction: JMP 0x1234
Binary:      0x10 0x12 0x34 0x00

Pixel representation:
R: 0x10 (JMP opcode) → RGB(16, 18, 52)
G: 0x12 (address high byte)
B: 0x34 (address low byte)
A: 0x00 (no flags)
```

## PNG File Format

### Structure

```
┌──────────────────────────────────┐
│  PNG Header (8 bytes)             │
├──────────────────────────────────┤
│  IHDR Chunk (Image metadata)     │
├──────────────────────────────────┤
│  IDAT Chunk (Pixel data)         │
│  ┌────────────────────────────┐  │
│  │ Row 0: Program metadata    │  │
│  │ Row 1+: Instructions       │  │
│  └────────────────────────────┘  │
├──────────────────────────────────┤
│  IEND Chunk (End marker)         │
└──────────────────────────────────┘
```

### Metadata Row (Row 0)

```
Pixel 0: Magic number 0x50584C00 ("PXL\0")
Pixel 1: Version (0x02000000 = v2.0)
Pixel 2: Entry point offset
Pixel 3: Program size (instructions)
Pixel 4: Required VRAM size
Pixel 5-7: Reserved
Pixel 8+: Program icon/thumbnail data
```

## Performance Characteristics

| Operation | GPU Cycles | Latency |
|-----------|-----------|----------|
| LOAD/STORE | 1 | ~1µs |
| ALU ops | 1 | <100ns |
| Control flow | 2-3 | ~200ns |
| VPIX | 1 | <100ns |

## Toolchain

- **pxlas.py**: Assembler (`.pxl` → `.px` PNG)
- **pxldis.py**: Disassembler (`.px` PNG → `.pxl`)
- **pxlemu.py**: Python reference emulator
- **pxldbg**: Interactive debugger
- **pxlprof**: Performance profiler

## Future Extensions

### v2.1 (Planned)
- 16-bit opcodes for extended instruction set
- SIMD vector instructions
- Hardware floating-point support

### v3.0 (Research)
- Multi-threading primitives
- Atomic operations
- Compute shader optimization hints

---

**Pixel ISA v2.0**: Making code visible, executable, and beautiful.
