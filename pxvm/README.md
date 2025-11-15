# pxvm - Multi-Kernel Virtual Machine

**Phase 5: Collective Evolution — Foundation**

A simple virtual machine where multiple "kernels" (processes) run in parallel, sharing a common framebuffer. This is the foundation for building emergent collective behaviors.

## Features

- **Multi-kernel execution**: Run up to 64 kernels in parallel
- **Shared framebuffer**: 1024×1024 RGB display visible to all kernels
- **Simple ISA**: 9 instructions (HALT, MOV, PLOT, ADD, SUB, CMP, JMP, JZ, NOP)
- **Assembler**: Write programs in readable assembly syntax
- **Isolated memory**: Each kernel has 64KB of private memory

## Quick Start

```bash
# Run the demo
python3 demo_two_kernels.py
```

This will spawn two kernels that draw to the shared framebuffer in parallel:
- **Green kernel**: Draws a vertical line
- **Red kernel**: Draws a diagonal line

## Architecture

### Kernel
Each kernel has:
- 8 registers (R0-R7, 32-bit)
- 64KB memory
- Program counter (PC)
- Zero flag (for conditional jumps)
- Cycle counter

### Instruction Set

| Opcode | Instruction | Description |
|--------|-------------|-------------|
| 0      | HALT        | Stop execution |
| 1      | MOV Rd, imm | Load 32-bit immediate into register |
| 2      | PLOT        | Draw pixel at (R0, R1) with color R2 |
| 3      | ADD Rd, Rs  | Rd += Rs |
| 4      | SUB Rd, Rs  | Rd -= Rs (sets zero flag) |
| 5      | JMP label   | Unconditional jump |
| 6      | JZ label    | Jump if zero flag set |
| 7      | CMP Rd, Rs  | Compare (sets zero flag if equal) |
| 255    | NOP         | No operation |

### Assembly Syntax

```asm
# Comments start with # or ;
    MOV R0, 100         # Load 100 into R0
    MOV R1, 0xFF00FF    # Load color into R1 (hex literals supported)

loop:                   # Labels for jumps
    PLOT                # Draw pixel
    ADD R0, R1          # R0 += R1
    CMP R0, R2          # Compare
    JZ done             # Jump if equal
    JMP loop            # Unconditional jump

done:
    HALT                # Stop
```

## Example: Two Parallel Kernels

```python
from pxvm.vm import PxVM
from pxvm.assembler import assemble

# Kernel 1: Draw a green line
code1 = assemble("""
    MOV R0, 100
    MOV R1, 100
    MOV R2, 0x00FF00
loop:
    PLOT
    ADD R1, R0
    JMP loop
""")

# Kernel 2: Draw a red line
code2 = assemble("""
    MOV R0, 200
    MOV R1, 200
    MOV R2, 0xFF0000
loop:
    PLOT
    ADD R0, R0
    JMP loop
""")

# Create VM and spawn kernels
vm = PxVM()
vm.spawn_kernel(code1, color=0x00FF00)
vm.spawn_kernel(code2, color=0xFF0000)

# Run
vm.run(max_cycles=1000)

# Both kernels share the same framebuffer!
print(f"Pixels drawn: {(vm.framebuffer > 0).any(axis=2).sum()}")
```

## What's Next

This is the foundation for:

- **Phase 5.1**: Pheromone communication (kernels leave chemical signals)
- **Phase 6**: Language (kernels write glyphs and broadcast messages)
- **Phase 7**: Tool-making (kernels create persistent artifacts)

The shared framebuffer is the **world**. The kernels are the **organisms**. The evolution begins here.

## Files

- `vm.py` - Core VM and kernel implementation
- `assembler.py` - Assembly language compiler
- `README.md` - This file
- `examples/` - Example kernel programs (future)

---

**Built on pxOS v1.0**
From bootloader to biosphere.
