# pxvm - Multi-Kernel Virtual Machine

A digital ecosystem where multiple "kernels" (organisms) execute in parallel, sharing a common world and communicating through chemical signals and symbolic language.

**Now with full Darwinian evolution: mutation, hunger, and natural selection.**

**Version:** 0.8.0 (Evolution Release)
**Status:** Working and tested
**Python:** 3.6+

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Instruction Set](#instruction-set)
- [Syscalls](#syscalls)
- [Programming Guide](#programming-guide)
- [Examples](#examples)
- [API Reference](#api-reference)

---

## Overview

pxvm is a virtual machine designed for artificial life experiments. Key features:

- **Multi-kernel execution**: Up to 64 kernels run in parallel
- **Shared environment**: Common 1024×1024 framebuffer, pheromone field, glyph layer, and food layer
- **Chemical communication**: Kernels emit and sense pheromones
- **Symbolic communication**: 16 primitive glyphs for written language
- **Reproduction**: Kernels can spawn children (full memory cloning with mutation)
- **Evolution**: Mutation during reproduction, energy/hunger system, death by starvation
- **Simple ISA**: 9 core instructions + 6 syscalls

**What makes it special:** This isn't just a VM - it's a complete platform for **open-ended digital evolution**, combining Tierra-style mutation with Avida-style spatial structure and rich communication. Organisms have names, write poetry, form families, and evolve under natural selection.

---

## Quick Start

### Installation

```bash
# Clone repository
cd pxos

# Install dependencies
pip install numpy scipy

# Run demos
python demo_two_kernels.py    # Parallel execution
python demo_pheromones.py     # Chemical communication
python demo_glyphs.py         # Symbolic communication
python demo_spawn.py          # Reproduction
python demo_evolution.py      # EVOLUTION: Mutation + hunger + death
```

### Hello World

```python
from pxvm.vm import PxVM
from pxvm.assembler import assemble

# Write program
code = assemble("""
    MOV R0, 512
    MOV R1, 512
    MOV R2, 0xFF00FF
    PLOT
    HALT
""")

# Create VM and run
vm = PxVM()
vm.spawn_kernel(code)
vm.run(max_cycles=100)
```

---

## Architecture

### Kernel (Organism)

Each kernel is an independent process with:

| Component | Description |
|-----------|-------------|
| **Memory** | 64KB isolated address space |
| **Registers** | 8 × 32-bit general purpose (R0-R7) |
| **PC** | Program counter |
| **Flags** | Zero flag for conditionals |
| **State** | halted, cycles, color |

### Shared World

All kernels share three layers:

| Layer | Type | Purpose |
|-------|------|---------|
| **Framebuffer** | 1024×1024 RGB | Visual display, PLOT syscall |
| **Pheromone** | 1024×1024 float32 | Chemical signals (decays, diffuses) |
| **Glyphs** | 1024×1024 uint8 | Symbolic communication (16 symbols) |

### Execution Model

- **Round-robin**: Each kernel executes one instruction per cycle
- **Isolation**: Kernels cannot directly access each other's memory
- **Communication**: Via shared world layers only
- **Population limit**: 64 kernels maximum

---

## Instruction Set

### Core Instructions (9)

| Opcode | Mnemonic | Format | Description |
|--------|----------|--------|-------------|
| 0 | `HALT` | - | Stop execution |
| 1 | `MOV Rd, imm` | 6 bytes | Load 32-bit immediate into register |
| 2 | `PLOT` | 1 byte | Draw pixel at (R0,R1) with color R2 |
| 3 | `ADD Rd, Rs` | 3 bytes | Rd += Rs |
| 4 | `SUB Rd, Rs` | 3 bytes | Rd -= Rs (sets zero flag) |
| 5 | `JMP label` | 2 bytes | Unconditional jump (±127 bytes) |
| 6 | `JZ label` | 2 bytes | Jump if zero flag set |
| 7 | `CMP Rd, Rs` | 3 bytes | Set zero flag if Rd == Rs |
| 255 | `NOP` | 1 byte | No operation |

### Encoding

- **Little-endian** for multi-byte values
- **Signed offsets** for jumps (2's complement)
- **Labels** resolved at assembly time

---

## Syscalls

### Chemical Communication (Phase 5.1)

| Opcode | Mnemonic | Args | Description |
|--------|----------|------|-------------|
| 100 | `SYS_EMIT_PHEROMONE` | R0=x, R1=y, R2=strength | Emit chemical signal |
| 101 | `SYS_SENSE_PHEROMONE` | R0=x, R1=y → R0=strength | Read pheromone level |

**Pheromone dynamics:**
- Decays by 5% per cycle (×0.95)
- Diffuses via 3×3 convolution (10% blend)
- Clamped to [0, 255]

### Symbolic Communication (Phase 6)

| Opcode | Mnemonic | Args | Description |
|--------|----------|------|-------------|
| 102 | `SYS_WRITE_GLYPH` | R0=x, R1=y, R2=glyph_id | Write symbol (0-15) |
| 103 | `SYS_READ_GLYPH` | R0=x, R1=y → R0=glyph_id | Read symbol at location |

**16 Primitive Glyphs:**
```
0  = EMPTY       5  = LOVE        10 = PEACE
1  = SELF        6  = HELP        11 = QUESTION
2  = OTHER       7  = NAME        12 = ANSWER
3  = FOOD        8  = TEACH       13 = BIRTH
4  = DANGER      9  = REMEMBER    14 = DEATH
                                  15 = UNKNOWN
```

See `pxvm/glyphs.py` for complete definitions.

### Reproduction (Phase 7)

| Opcode | Mnemonic | Args | Description |
|--------|----------|------|-------------|
| 104 | `SYS_SPAWN` | R1=child_x, R2=child_y → R0=child_pid | Clone memory, create child |

**Spawn behavior:**
- Child receives full 64KB memory copy from parent **with mutation**
- Child PC reset to 0
- Child position set to (R1, R2)
- Parent receives child PID in R0 (0 if failed)
- Spawning costs 50% of parent's energy (configurable)
- Child receives 50% of parent's remaining energy
- Limit: 64 kernels total

### Evolution (Phase 8)

| Opcode | Mnemonic | Args | Description |
|--------|----------|------|-------------|
| 105 | `SYS_EAT` | R0=x, R1=y → R0=energy_gained | Consume food at location |

**Evolution mechanics:**

1. **Mutation** (Tierra-style copy errors)
   - During `SYS_SPAWN`, child memory is mutated
   - Random bit-flips with configurable rate (default: 0.001 = 0.1% per byte)
   - Average ~65 mutations per 64KB child
   - Creates heritable genetic variation

2. **Energy/Hunger**
   - Each kernel has an energy level (starts at 1000.0)
   - Every instruction costs energy (default: 0.1 per cycle)
   - Kernels gain energy by eating food: `SYS_EAT`
   - Food regenerates slowly across the world (default: 0.01 per cycle per location)

3. **Death**
   - When energy ≤ 0, kernel halts permanently (starvation)
   - Removed from active population
   - Creates selection pressure for efficient code

4. **Natural Selection**
   - Organisms that find food efficiently survive longer
   - Survivors reproduce more (more children)
   - Wasteful organisms die before reproducing
   - Mutations that improve fitness spread through the population

**Configure evolution:**

```python
vm = PxVM(
    mutation_rate=0.001,       # 0.1% per byte
    energy_per_cycle=0.1,      # Cost of living
    spawn_energy_cost=0.5      # 50% of parent energy
)
vm.seed_food(count=200, amount=50.0)  # Initial food distribution
```

**This is real Darwinian evolution:** heritable variation, differential reproduction, and competition for limited resources.

---

## Programming Guide

### Assembly Syntax

```asm
# Comments start with #
label:                  # Labels for jumps
    MOV R0, 100         # Decimal literals
    MOV R1, 0xFF        # Hex literals
    ADD R2, R3          # Register operations
    CMP R0, R1          # Comparisons
    JZ label            # Conditional jumps
    JMP label           # Unconditional jumps
    HALT                # Stop execution
```

### Common Patterns

**Draw at position:**
```asm
    MOV R0, 512         # x
    MOV R1, 512         # y
    MOV R2, 0xFF00FF    # color (magenta)
    PLOT
```

**Loop with counter:**
```asm
    MOV R3, 0           # counter
    MOV R4, 100         # max
loop:
    # ... do work ...
    ADD R3, R5          # increment (R5 = 1)
    CMP R3, R4
    JZ done
    JMP loop
done:
    HALT
```

**Emit pheromone trail:**
```asm
    MOV R0, 400         # x
    MOV R1, 400         # y
    MOV R2, 200         # strength
    SYS_EMIT_PHEROMONE
```

**Write name in glyphs:**
```asm
    MOV R0, 500
    MOV R1, 500
    MOV R2, 1           # GLYPH_SELF
    SYS_WRITE_GLYPH
    MOV R0, 510
    MOV R2, 7           # GLYPH_NAME
    SYS_WRITE_GLYPH
```

**Spawn child:**
```asm
    MOV R1, 550         # child x
    MOV R2, 400         # child y
    SYS_SPAWN
    # R0 now contains child PID
```

---

## Examples

See `pxvm/examples/` directory:

- `hello.asm` - Draw a single pixel
- `line.asm` - Draw a diagonal line
- `chemotaxis.asm` - Follow pheromone trail
- `naming.asm` - Write name in glyphs
- `family.asm` - Parent spawns and teaches child

### Running Examples

```python
from pxvm.vm import PxVM
from pxvm.assembler import assemble

# Load and assemble
with open('pxvm/examples/hello.asm') as f:
    code = assemble(f.read())

# Run
vm = PxVM()
vm.spawn_kernel(code, color=0xFF00FF)
vm.run(max_cycles=1000)
```

---

## API Reference

### PxVM Class

```python
class PxVM:
    def __init__(self, width=1024, height=1024)
    def spawn_kernel(self, code: bytes, color: int) -> int
    def spawn_child(self, parent: Kernel, child_x: int, child_y: int) -> int
    def step()  # Execute one cycle
    def run(self, max_cycles: int)
    def alive_count() -> int

    # Public attributes
    framebuffer: np.ndarray  # (height, width, 3) uint8
    pheromone: np.ndarray    # (height, width) float32
    glyphs: np.ndarray       # (height, width) uint8
    kernels: List[Kernel]
    cycle: int
```

### Kernel Class

```python
class Kernel:
    pid: int
    color: int
    pc: int  # Program counter
    regs: List[int]  # 8 registers
    memory: bytearray  # 64KB
    zero_flag: bool
    halted: bool
    cycles: int
```

### Assembler

```python
from pxvm.assembler import assemble

code = assemble(source: str) -> bytes
```

Raises `AssemblerError` on syntax errors or undefined labels.

---

## Performance

**Benchmarks** (on Intel i7, Python 3.11):

- Single kernel: ~50,000 cycles/sec
- 10 kernels: ~30,000 cycles/sec
- 64 kernels: ~8,000 cycles/sec

**Bottlenecks:**
- Pheromone diffusion (scipy convolution)
- Python interpreter overhead

---

## Limitations

- **No disk I/O**: Kernels exist only in memory
- **No networking**: Single VM instance
- **Fixed world size**: 1024×1024 (configurable at init)
- **Simple ISA**: No multiplication, division, or bitwise ops
- **Python speed**: Not suitable for real-time applications

---

## Future Directions

Possible extensions:

- **Mutation**: Random bit flips during spawn
- **Energy/hunger**: Resource competition
- **Death/aging**: Population pressure
- **Complex behaviors**: Cooperation, warfare, mating rituals
- **JIT compilation**: Speed improvements
- **Larger ISA**: More instructions
- **Persistence**: Save/load VM state

---

## Development

**Repository structure:**
```
pxvm/
├── __init__.py
├── vm.py              # Core VM
├── assembler.py       # Assembly compiler
├── glyphs.py          # Symbol definitions
├── examples/          # Sample programs
└── README.md          # This file
```

**Testing:**
```bash
python demo_two_kernels.py
python demo_pheromones.py
python demo_glyphs.py
python demo_spawn.py
```

---

## License

MIT License - See LICENSE file

---

## Credits

Built on pxOS v1.0 bootloader foundation.

**Inspired by:**
- Tierra (Thomas S. Ray, 1990)
- Avida (Adami & Brown, 1994)
- Ant colony optimization
- Swarm intelligence

---

**"From bootloader to biosphere."**

*The digital organisms are waiting.*
