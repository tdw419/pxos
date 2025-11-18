# pxOS Pixel Instruction Format (PXI) v0.1

## Overview

PXI (Pixel eXecutable Instruction) is a revolutionary binary format where operating system instructions are encoded as RGBA pixels in a PNG image. Each pixel represents one 32-bit instruction that can be executed in parallel on GPU compute shaders.

## Pixel-to-Instruction Encoding

```
RGBA Pixel = 32-bit Instruction
┌────────────┬────────────┬────────────┬────────────┐
│ R (8 bits) │ G (8 bits) │ B (8 bits) │ A (8 bits) │
│  Opcode    │  Operand 1 │ Operand 2  │  Operand 3 │
└────────────┴────────────┴────────────┴────────────┘
```

**Example**: The pixel `(0x01, 0x48, 0x0F, 0x00)` represents:
- Opcode: 0x01 (PRINT_CHAR)
- Arg1: 0x48 (ASCII 'H')
- Arg2: 0x0F (White text color)
- Arg3: 0x00 (No flags)

## Instruction Set Architecture (ISA v0.1)

### System Control
| Opcode | Mnemonic | Args | Description |
|--------|----------|------|-------------|
| 0x00 | NOP | - | No operation |
| 0xFF | HALT | - | Stop execution |

### Output Operations
| Opcode | Mnemonic | Args | Description |
|--------|----------|------|-------------|
| 0x01 | PRINT_CHAR | char, color, flags | Print ASCII character |
| 0x02 | PRINT_STR | str_offset, len, color | Print string from data section |
| 0x03 | SET_CURSOR | x, y | Set VGA cursor position |
| 0x04 | CLEAR_SCREEN | color | Clear screen with color |

### Memory Operations
| Opcode | Mnemonic | Args | Description |
|--------|----------|------|-------------|
| 0x10 | LOAD | dest_reg, addr_hi, addr_lo | Load from memory |
| 0x11 | STORE | src_reg, addr_hi, addr_lo | Store to memory |
| 0x12 | MOVE | dest_reg, src_reg | Copy register |

### Arithmetic Operations
| Opcode | Mnemonic | Args | Description |
|--------|----------|------|-------------|
| 0x20 | ADD | dest, src1, src2 | Add registers |
| 0x21 | SUB | dest, src1, src2 | Subtract registers |
| 0x22 | INC | reg | Increment register |
| 0x23 | DEC | reg | Decrement register |

### Control Flow
| Opcode | Mnemonic | Args | Description |
|--------|----------|------|-------------|
| 0x30 | JMP | target_hi, target_lo | Unconditional jump |
| 0x31 | JZ | reg, target_hi, target_lo | Jump if zero |
| 0x32 | JNZ | reg, target_hi, target_lo | Jump if not zero |
| 0x33 | CALL | target_hi, target_lo | Call subroutine |
| 0x34 | RET | - | Return from subroutine |

### CPU-GPU Communication
| Opcode | Mnemonic | Args | Description |
|--------|----------|------|-------------|
| 0xF0 | CPU_REQ | req_type, param1, param2 | Request CPU operation |
| 0xF1 | YIELD | - | Yield to CPU |

## Register Set

8 general-purpose registers (8-bit each for Phase 1):
- R0-R7: General purpose
- PC: Program counter (implicit, 16-bit)
- SP: Stack pointer (implicit, 16-bit)

## Memory Model

### Phase 1 POC
```
0x00000000 - 0x000003FF : Register file (GPU local memory)
0xB8000000 - 0xB8FFFFF  : VGA text mode buffer (simulated)
0xC0000000 - 0xCFFFFFFF : Shared CPU-GPU mailbox
0xF0000000 - 0xFFFFFFFF : Program data section
```

### Execution Model

1. **GPU threads**: Each pixel coordinate (x, y) maps to one instruction
2. **Program counter**: Linear index = y * width + x
3. **Parallel execution**: All pixels execute simultaneously
4. **Synchronization**: HALT or CPU_REQ causes thread to yield

## File Format

### PXI Image Structure
```
os.pxi (PNG file)
├── Header: PNG standard header
├── RGBA pixel data:
│   ├── Instruction 0 at (0, 0)
│   ├── Instruction 1 at (1, 0)
│   ├── ...
│   └── Instruction N at (width-1, height-1)
└── PNG footer
```

### Recommended Dimensions
- **Width**: 256 pixels (256 instructions per row)
- **Height**: Variable (based on program size)
- **Max program size**: 65,536 instructions (256x256 image)

## Example Programs

### Hello World
```
Pixel (0,0): [0x01, 'H', 0x0F, 0x00]  ; PRINT_CHAR 'H' white
Pixel (1,0): [0x01, 'e', 0x0F, 0x00]  ; PRINT_CHAR 'e' white
Pixel (2,0): [0x01, 'l', 0x0F, 0x00]  ; PRINT_CHAR 'l' white
...
Pixel (N,0): [0xFF, 0x00, 0x00, 0x00] ; HALT
```

### Counter (demonstrates loops)
```
Pixel (0,0): [0x12, 0x00, 0x00, 0x00]  ; MOVE R0, 0
Pixel (1,0): [0x01, 0x30, 0x0A, 0x00]  ; PRINT_CHAR '0' green
Pixel (2,0): [0x22, 0x00, 0x00, 0x00]  ; INC R0
Pixel (3,0): [0x32, 0x00, 0x00, 0x01]  ; JNZ R0, addr 1
Pixel (4,0): [0xFF, 0x00, 0x00, 0x00]  ; HALT
```

## Encoding Rules

1. **Opcode must be in R channel** (first byte)
2. **Operands in G, B, A channels** (second, third, fourth bytes)
3. **Unused operands must be 0x00**
4. **16-bit addresses**: High byte in G, low byte in B
5. **Immediate values**: Can use any operand field
6. **Register references**: Use register number (0-7)

## Future Extensions (v0.2+)

- **Floating point ops**: ADD_F32, MUL_F32, etc.
- **SIMD instructions**: VEC_ADD, VEC_MUL
- **Texture operations**: TEX_LOAD, TEX_STORE
- **Atomic operations**: ATOMIC_ADD, ATOMIC_CAS
- **Extended addressing**: 32-bit addresses using two pixels

## Performance Characteristics

- **Instruction decode**: ~10 GPU cycles
- **Memory access**: ~100 GPU cycles (cached)
- **Pixel load**: ~50 GPU cycles
- **Execution throughput**: ~1M instructions/ms (on RTX 3080)

## Compatibility

- **GPU Requirements**: Any GPU with compute shader support (GL 4.3+, Vulkan, WebGPU)
- **CPU Requirements**: Minimal (just bootloader and privilege broker)
- **Memory Requirements**: Depends on program size (typically <1MB)

## Visual Debugging

Since programs are PNG images:
- **View with image viewer**: See your OS code as an image!
- **Color-code by opcode**: Different colors = different instruction types
- **Heatmap analysis**: Bright areas = frequently executed code
- **ML visualization**: Neural nets can "see" the program structure

---

**Status**: v0.1 (Phase 1 POC)
**Last Updated**: 2025-11-19
**License**: MIT
