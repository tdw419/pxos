# Pure Pixel Kernel - Python Prototype

**State == Color | RAM == Display**

A complete operating system where pixels are the fundamental unit of computation, memory, and storage.

## Overview

The Pure Pixel Kernel (PPK) is a visual operating system where:
- **Memory is visible**: The screen IS the RAM
- **Processes are sprites**: Each process is a colored 16×16 block
- **Registers are pixels**: RGB values encode data
- **Instructions are colors**: Piet-inspired visual programming
- **Files are images**: File system as texture atlas

## Features

- ✅ **Pixel Memory Management**: Guillotine packing with Best Area Fit
- ✅ **Visual Processes**: Processes as visible sprites with RGB registers
- ✅ **Round-Robin Scheduler**: Fair time-sharing with scanline visualization
- ✅ **Heatmap Overlays**: See which pixels are "hot" (frequently accessed)
- ✅ **Game of Life Demo**: Each process runs Conway's Game of Life
- 🚧 **Piet Interpreter**: Color-based instruction execution (coming soon)
- 🚧 **Pixel File System**: Hilbert curve storage (coming soon)
- 🚧 **Color-Coded Syscalls**: Visual system calls (coming soon)

## Requirements

```
python >= 3.8
pygame >= 2.0.0
numpy >= 1.20.0
```

## Installation

```bash
cd pixel-os
pip install -r requirements.txt
```

## Running the Kernel

```bash
python kernel/pixel_kernel.py
```

## Controls

- **SPACE**: Pause/Resume execution
- **N**: Spawn a new process
- **D**: Print debug information
- **ESC**: Shutdown the kernel
- **Click**: Spawn a process at mouse position

## Architecture

### Memory Layout

```
┌────────────────────────────────────┐
│  Process 1 (16×16)                │
│  ┌────────────────────────┐       │
│  │ Row 0: Header         │       │
│  │ Row 1: RGB Registers  │       │
│  │ Rows 2-15: Stack      │       │
│  └────────────────────────┘       │
├────────────────────────────────────┤
│  Process 2 (16×16)                │
├────────────────────────────────────┤
│  Free Memory (Black)              │
└────────────────────────────────────┘
```

### Process Header (Row 0)

| Pixel | Field | Description |
|-------|-------|-------------|
| (0,0) | PID | Process ID encoded as RGB |
| (1,0) | Status | Color-coded state (Green=Running, Red=Terminated, etc.) |
| (2,0) | IP_X | Instruction Pointer X coordinate |
| (3,0) | IP_Y | Instruction Pointer Y coordinate |
| (4,0) | SP | Stack Pointer |

### RGB Registers (Row 1)

| Pixel | Register | Purpose |
|-------|----------|---------|
| (0,1) | R0 | Accumulator |
| (1,1) | R1 | Counter |
| (2,1) | R2 | Index |

## File Structure

```
pixel-os/
├── kernel/
│   ├── pixel_kernel.py           # Main kernel loop
│   ├── pixel_memory_manager.py   # 2D memory allocator
│   ├── pixel_process.py          # Process control blocks
│   └── pixel_scheduler.py        # Round-robin scheduler
├── fs/                           # File system (coming soon)
├── drivers/                      # Device drivers (coming soon)
├── userland/                     # User programs (coming soon)
├── lib/                          # Standard library (coming soon)
├── tools/                        # Development tools (coming soon)
├── tests/                        # Unit tests (coming soon)
└── docs/                         # Documentation (coming soon)
```

## Performance

On a typical modern system (2025):
- **60 FPS** rendering
- **1000+ processes** supported
- **2 million pixels** of memory (1920×1080)
- **Real-time** memory visualization

## How It Works

### Memory Management

The Pixel Memory Manager (PMM) uses the **Guillotine packing algorithm** to allocate 16×16 pixel blocks:

1. Maintain a list of free rectangles
2. Find the smallest rectangle that fits (Best Area Fit)
3. Place the block in the corner
4. Cut the remaining space into new free rectangles
5. Merge adjacent rectangles to reduce fragmentation

### Process Execution

Each process runs a simple instruction cycle:

1. **Fetch**: Read current pixel at Instruction Pointer
2. **Decode**: Interpret color as instruction (Piet-style)
3. **Execute**: Modify registers and stack
4. **Update**: Write results back to pixels

Currently, processes run Conway's Game of Life as a demonstration.

### Scheduling

The scheduler uses **Round-Robin** with a visual scanline:

1. Each process gets a fixed time slice (100 frames)
2. When time expires, context switch to next process
3. Skip BLOCKED or TERMINATED processes
4. Visual scanline shows which process is active

## Debugging

All state is visible:
- **Black regions**: Free memory
- **Colored blocks**: Active processes
- **Bright pixels**: High register values / alive cells
- **Dark pixels**: Low values / dead cells
- **Green status**: Process is running
- **Yellow status**: Process is blocked

## Future Directions

- **Piet Interpreter**: Full color-based instruction execution
- **Visual Syscalls**: Fork, exec, read, write via color codes
- **Hilbert File System**: Space-filling curve storage
- **IPC via Color Mixing**: Shared memory with visual blending
- **GPU Acceleration**: Native execution on compute shaders

## References

- [Pure Pixel Kernel Architecture Paper](../PURE_PIXEL_KERNEL_ARCHITECTURE.md)
- [Native GPU Architecture](../NATIVE_PIXEL_OS_ARCHITECTURE.md)
- [Development Roadmap](../PIXEL_OS_ROADMAP.md)

## License

MIT License - See [LICENSE](LICENSE) file

## Credits

Built with inspiration from:
- Conway's Game of Life
- Piet esoteric programming language
- Visual6502 transistor-level simulation
- OSDev community

---

**"The Blue Screen of Death is just a process that decided to paint itself blue."**
