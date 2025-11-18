# Python on pxOS - Development Roadmap

This document outlines the path to full Python support on pxOS, from cross-compilation to native runtime.

---

## Overview

Python support for pxOS is being developed in **two parallel tracks**:

1. **Near-Term (Track A)**: Python as a cross-compiler (development-time)
2. **Mid-Term (Track B)**: Native Python runtime on pxOS (run-time)

---

## Track A: Python Cross-Compiler (IMPLEMENTED ✅)

**Goal**: Let LLMs write pxOS code in Python, which compiles to primitives.

### Status: **COMPLETE** (v0.1)

### Components

#### 1. pxos Python Module (`tools/pxos/__init__.py`) ✅

High-level API that mimics what will eventually run on pxOS:

```python
from pxos import clear_screen, print_text, loop_forever

def main():
    clear_screen()
    print_text("Hello from Python!")
    loop_forever()
```

**Features**:
- Display: `clear_screen()`, `print_text()`, `print_char()`, `move_cursor()`
- Keyboard: `read_key()`, `check_key()`
- Colors: `make_color()`, color constants
- Control: `loop_forever()`, `delay()`
- Memory: `peek()`, `poke()`

#### 2. Python Compiler (`tools/pxpyc.py`) ✅

Translates Python → primitives:

```bash
pxpyc.py hello.py -o pxos_commands.txt
```

**Features**:
- AST parsing
- Function call translation
- String literal handling
- x86 opcode generation
- Memory allocation (data/code sections)
- Symbol management

**Limitations (v0.1)**:
- Only `main()` function
- No variables/expressions
- No loops/conditionals
- Basic function calls only

#### 3. Examples (`examples/python/`) ✅

- `hello_simple.py` - Basic program
- `hello_multiline.py` - Multiple text lines
- `hello_colors.py` - Color attributes
- `bootloader_demo.py` - Boot sequence simulation

### Workflow

```
Python Source → pxpyc.py → Primitives → build_pxos.py → pxos.bin → QEMU
```

### Next Steps for Track A

- [ ] **v0.2**: Add variable support
- [ ] **v0.3**: Add loop constructs (for/while)
- [ ] **v0.4**: Add conditionals (if/else)
- [ ] **v0.5**: Add expressions and operators
- [ ] **v0.6**: Multiple function definitions
- [ ] **v0.7**: Basic data structures (lists, dicts)
- [ ] **v1.0**: Full Python subset compiler

---

## Track B: Native Python Runtime

**Goal**: Run MicroPython directly on pxOS hardware.

### Status: **PLANNED**

### Prerequisites

Before native Python can run on pxOS, we need:

#### 1. Two-Stage Bootloader (Milestone: M2)

**Current**: Single 512-byte boot sector
**Target**: Stage 1 (512b) → Stage 2 (multi-KB)

**Requirements**:
- Stage 1: Load Stage 2 from disk
- Stage 2: Setup protected mode, load kernel

**Timeline**: Q1 2025

#### 2. Protected Mode Kernel (Milestone: M3)

**Current**: 16-bit real mode
**Target**: 32-bit protected mode with flat memory model

**Requirements**:
- GDT setup
- Protected mode switch
- Flat 4GB addressing
- Basic memory management (bump allocator)

**Timeline**: Q2 2025

#### 3. Minimal Kernel Services (Milestone: M4)

**Requirements**:
- Console I/O (write_char, read_key)
- Memory allocator (malloc/free)
- Basic file I/O (once FAT12 driver exists)

**Timeline**: Q2 2025

#### 4. MicroPython Port (Milestone: M6)

**Requirements**:
- Cross-compiler: `i386-elf-gcc`
- MicroPython source (minimal config)
- pxOS HAL layer (`hal_pxos.c`):
  - `mp_hal_stdout_tx_char()` → pxOS write_char
  - `mp_hal_stdin_rx_chr()` → pxOS read_key
  - Memory allocation hooks
- Link as flat binary (`python.bin`)

**Timeline**: Q3 2025

#### 5. Program Loader (Milestone: M7)

**Requirements**:
- Load `python.bin` from disk
- Relocate to high memory (e.g., 0x200000)
- Jump to Python entry point
- Handle returns/exits

**Timeline**: Q3 2025

### Architecture Evolution

```
pxOS v0.1 (current)
├── 512-byte boot sector
└── Primitives only

pxOS v0.5 (M2-M3)
├── Stage 1 bootloader (512b)
├── Stage 2 loader (protected mode)
└── Minimal kernel

pxOS v1.0 (M6-M7)
├── Stage 1 bootloader
├── Stage 2 loader
├── pxOS kernel (syscalls, HAL)
└── Python runtime (MicroPython)
    ├── REPL
    └── Script execution
```

### MicroPython Configuration

**Target**: Minimal embedded config

- **Heap**: 64-128KB (configurable)
- **Features**: Core language only, no external libs initially
- **Modules**: `sys`, `gc`, `pxos` (custom)
- **Disabled**: Networking, filesystem (until pxOS supports them)

### Custom `pxos` Module (for MicroPython)

Once Python runs natively, provide a `pxos` module that matches the API:

```python
# This now runs ON pxOS, not cross-compiled
import pxos

pxos.clear_screen()
pxos.print_text("Running on real pxOS!")
```

Implementation:
- Written in C
- Calls pxOS kernel syscalls
- Registered as MicroPython builtin module

---

## Track C: Pixel Cartridge System (IMPLEMENTED ✅)

**Goal**: Systematic porting of foreign binaries via pixel-based containers.

### Status: **SPEC COMPLETE** (v0.1)

### Components

#### 1. PXCARTRIDGE Format (`docs/pxcartridge_v0.md`) ✅

Specification for `.pxcart.png` files:
- Header rows (metadata as pixels)
- Payload (binary data as pixels)
- Checksum row (integrity)

**Metadata Stored**:
- ISA (x86_32, x86_64, arm64, etc.)
- ABI (ELF, PE, raw, pxOS, etc.)
- Entry point
- Dependencies
- License/Author
- SHA256 hash

#### 2. Cartridge Maker (`tools/make_pxcart.py`) ✅

Convert binary → pixel cartridge:

```bash
make_pxcart.py hello.bin \
  --isa x86_32 --abi raw_bin --entry 0x7C00 \
  -o hello.pxcart.png
```

Features:
- Binary encoding to RGBA pixels
- Metadata header generation
- Optional zlib compression
- Checksum calculation

#### 3. Cartridge Reader (`tools/read_pxcart.py`) ✅

Read/extract cartridges:

```bash
# Show info
read_pxcart.py hello.pxcart.png --info

# Extract binary
read_pxcart.py hello.pxcart.png --extract hello.bin

# Verify integrity
read_pxcart.py hello.pxcart.png --verify
```

### Porting Workflows

**Workflow 1: Emulation**
```
.pxcart.png → Decode → Emulator → Runs on pxOS
```

**Workflow 2: Lift & Translate**
```
.pxcart.png → Lifter → IR → Translator → pxOS primitives
```

**Workflow 3: Hybrid (Selective Patching)**
```
.pxcart.png → Patcher → Binary + pxOS stubs
```

### Next Steps for Track C

- [ ] **M8**: Implement cartridge loader in pxOS
- [ ] **M9**: Basic x86 emulator for cartridges
- [ ] **M10**: IR lifter (binary → intermediate representation)
- [ ] **M11**: IR translator (IR → pxOS primitives)
- [ ] **M12**: Pixel Vault integration

---

## Integration Timeline

### Phase 1: Development Tools (Q1 2025) ✅ COMPLETE

- [x] Python cross-compiler working
- [x] Pixel cartridge spec defined
- [x] Cartridge tools implemented
- [x] Example programs created
- [x] Documentation complete

### Phase 2: OS Foundation (Q2 2025)

- [ ] Two-stage bootloader
- [ ] Protected mode kernel
- [ ] Basic memory management
- [ ] Console I/O syscalls

### Phase 3: Python Runtime (Q3 2025)

- [ ] MicroPython cross-compilation
- [ ] pxOS HAL layer
- [ ] Program loader
- [ ] Python REPL on pxOS

### Phase 4: Porting Pipeline (Q4 2025)

- [ ] Cartridge loader in pxOS
- [ ] Basic emulator
- [ ] Lifter/translator prototype
- [ ] First ported program

### Phase 5: Ecosystem (2026+)

- [ ] Full Python stdlib (subset)
- [ ] Package manager for pxOS
- [ ] Pixel Vault + cartridge browser
- [ ] Developer tools in Python
- [ ] Self-hosting: pxOS builds itself

---

## Success Metrics

### Near-Term (Track A) ✅

- [x] LLM can write pxOS code in Python
- [x] Python programs compile to primitives
- [x] Examples run in QEMU
- [ ] 90% of primitives can be generated from Python

### Mid-Term (Track B)

- [ ] MicroPython boots on pxOS
- [ ] REPL accepts Python commands
- [ ] Core language features work
- [ ] Can run simple scripts (hello world, etc.)

### Long-Term (Track C)

- [ ] Foreign binaries load as cartridges
- [ ] At least one emulator works
- [ ] At least one program successfully ported
- [ ] Pixel Vault indexes 100+ cartridges

---

## Current Status Summary

| Track | Component | Status | Version |
|-------|-----------|--------|---------|
| A | pxos Python module | ✅ Complete | v0.1 |
| A | pxpyc.py compiler | ✅ Complete | v0.1 |
| A | Python examples | ✅ Complete | v0.1 |
| C | PXCARTRIDGE spec | ✅ Complete | v0.1 |
| C | make_pxcart.py | ✅ Complete | v0.1 |
| C | read_pxcart.py | ✅ Complete | v0.1 |
| B | Two-stage bootloader | 🔵 Planned | - |
| B | Protected mode | 🔵 Planned | - |
| B | MicroPython port | 🔵 Planned | - |
| C | Cartridge loader | 🔵 Planned | - |
| C | Emulator/lifter | 🔵 Planned | - |

**Legend**: ✅ Complete | 🟡 In Progress | 🔵 Planned

---

## Getting Started

### For LLM Developers (Track A)

1. Write Python using `pxos` module:
   ```python
   from pxos import clear_screen, print_text

   def main():
       clear_screen()
       print_text("Hello pxOS!")
   ```

2. Compile to primitives:
   ```bash
   cd tools
   python3 pxpyc.py ../examples/python/hello_simple.py --run
   ```

3. See it run in QEMU!

### For Binary Porters (Track C)

1. Create a pixel cartridge:
   ```bash
   python3 tools/make_pxcart.py myapp.bin \
     --isa x86_32 --abi elf_linux --entry 0x400000 \
     -o myapp.pxcart.png
   ```

2. Inspect it:
   ```bash
   python3 tools/read_pxcart.py myapp.pxcart.png --info --verify
   ```

3. Store in Pixel Vault for future porting

### For OS Developers (Track B)

1. Study current architecture: `docs/architecture.md`
2. Review protected mode requirements
3. Start designing Stage 2 bootloader
4. Plan GDT and memory layout

---

## References

- [pxOS Architecture](docs/architecture.md)
- [Primitive Commands](docs/primitives.md)
- [PXCARTRIDGE Spec](docs/pxcartridge_v0.md)
- [Python Examples](examples/python/)
- [MicroPython](https://micropython.org/)
- [OSDev Wiki - Protected Mode](https://wiki.osdev.org/Protected_Mode)

---

**Last Updated**: 2025-01-18
**Status**: Track A (Python compiler) and Track C (Pixel cartridge spec) complete!
