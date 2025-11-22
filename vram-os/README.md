# VRAM OS - GPU-Native Operating System

**VRAM OS** is a revolutionary operating system architecture where **programs are pixels** and **execution happens on the GPU**.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     VRAM OS Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────┐        ┌──────────────────┐             │
│  │  Pixel ISA    │───────▶│  Window Manager  │             │
│  │  Programs     │        │  (pxl_wm.px)     │             │
│  │  (.px files)  │        └──────────────────┘             │
│  └───────────────┘                 │                         │
│         │                          │                         │
│         ▼                          ▼                         │
│  ┌────────────────────────────────────────────┐             │
│  │       VRAM (Texture Memory)                │             │
│  │  ┌──────────────┬──────────────────────┐  │             │
│  │  │ Window Table │ Program Memory       │  │             │
│  │  │ (Metadata)   │ (Instructions+Data)  │  │             │
│  │  └──────────────┴──────────────────────┘  │             │
│  └────────────────────────────────────────────┘             │
│         │                          │                         │
│         ▼                          ▼                         │
│  ┌─────────────────┐      ┌──────────────────┐             │
│  │  HAC Compositor │      │  RISC-V Emulator │             │
│  │  (WGSL Shader)  │      │  (WGSL Shader)   │             │
│  └─────────────────┘      └──────────────────┘             │
│         │                          │                         │
│         ▼                          ▼                         │
│  ┌──────────────────────────────────────────┐               │
│  │          WebGPU Runtime                   │               │
│  └──────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### 1. Programs as Pixels
- Each `.px` file is a **valid PNG image**
- Instructions are **encoded as RGBA pixel values**
- Visual representation = executable code

### 2. VRAM as Memory
- All memory is **GPU texture storage**
- Direct pixel read/write = memory access
- Zero CPU-to-GPU transfer overhead

### 3. Dual Execution Paths

#### Path A: Pixel ISA (Simulated/Educational)
- Custom 8-bit instruction set
- WGSL compute shader interpreter
- Fast prototyping and visualization

#### Path B: RISC-V (Real Linux)
- Full rv32ima ISA implementation
- Boot actual Linux kernel
- WGSL-based CPU emulation

## Directory Structure

```
vram-os/
├── README.md                 # This file
├── specs/                    # Specifications and documentation
│   ├── pixel-isa-v2.md      # Enhanced Pixel ISA specification
│   ├── window-table.md      # Window Table format
│   └── vram-layout.md       # Memory map and layout
├── pixel-isa/               # Pixel ISA tools
│   ├── pxlas.py            # Assembler (assembly → pixel binary)
│   ├── pxldis.py           # Disassembler (pixels → assembly)
│   └── pxlemu.py           # Python reference emulator
├── window-manager/          # Window management system
│   ├── pxl_wm.px           # Window manager (Pixel ISA)
│   ├── window_table.py     # Window Table encoder/decoder
│   └── examples/           # Example window programs
├── compositor/              # Hardware Abstraction Compositor
│   ├── hac.wgsl            # Main compositor shader
│   ├── blend.wgsl          # Alpha blending shader
│   └── decorations.wgsl    # Window decorations shader
├── emulator/                # RISC-V emulator
│   ├── rv32ima.wgsl        # RISC-V instruction decoder
│   ├── mmu.wgsl            # Memory management unit
│   └── uart.wgsl           # I/O device emulation
└── frontend/                # React UI components
    ├── BootSimulator.tsx    # Dual-boot interface
    ├── VRAMVisualizer.tsx   # Real-time VRAM viewer
    └── PixelInspector.tsx   # Pixel debugging tool
```

## Quick Start

### 1. Build a Pixel ISA Program

```bash
# Write assembly
cat > hello.pxl << 'EOF'
; hello.pxl - Print "Hello VRAM!"
START:
    LOAD R1, #MSG_ADDR
    CALL print_string
    HALT

print_string:
    LOAD R2, [R1]
    CMP R2, #0
    JEQ done
    STORE #UART_ADDR, R2
    ADD R1, #1
    JMP print_string
done:
    RET

MSG_ADDR: DATA "Hello VRAM!", 0
UART_ADDR: EQU 0x10000000
EOF

# Assemble to pixels
python3 vram-os/pixel-isa/pxlas.py hello.pxl -o hello.px

# The output hello.px is a valid PNG!
file hello.px
# hello.px: PNG image data, 64 x 8, 8-bit/color RGBA
```

### 2. Run in Browser

```bash
cd vram-os/frontend
npm install
npm run dev
```

Open http://localhost:5173 and:
1. Click "Load Program"
2. Select `hello.px`
3. Watch it execute on the GPU!

## Technical Specifications

### Pixel Encoding Format (v2)

Each pixel encodes one 32-bit instruction:

```
┌────────┬────────┬────────┬────────┐
│ Red    │ Green  │ Blue   │ Alpha  │
├────────┼────────┼────────┼────────┤
│ Opcode │ Arg0   │ Arg1   │ Flags  │
│ (8bit) │ (8bit) │ (8bit) │ (8bit) │
└────────┴────────┴────────┴────────┘

Alpha Flags:
  Bit 0-1: Length (00=8b, 01=16b, 10=24b, 11=32b)
  Bit 2:   Conditional flag
  Bit 3-7: Reserved
```

### Window Table Format

Located at VRAM address `0x100000`:

```
Each window entry = 4 pixels (128 bits):

Pixel 0: [R=Window ID] [G=X-pos] [B=Y-pos] [A=Flags]
Pixel 1: [R=Width] [G=Height] [B=Z-order] [A=State]
Pixel 2: [R=Title ptr high] [G=Title ptr low] [B=Reserved] [A=Reserved]
Pixel 3: [R=Content ptr high] [G=Content ptr low] [B=Reserved] [A=Reserved]

Max windows: 32
Total size: 128 pixels (512 bytes)
```

### VRAM Memory Map

```
Address Range       | Size    | Purpose
--------------------|---------|----------------------------------
0x00000000-0x000FFF | 4KB     | Interrupt Vector Table
0x00001000-0x0FFFFF | 1020KB  | Reserved
0x00100000-0x001001FF| 512B   | Window Table (32 entries × 16B)
0x00100200-0x00FFFFFF| 15.9MB | Program Memory
0x01000000-0x01FFFFFF| 16MB   | Display Buffer (1920×1080×4)
0x02000000-0x02FFFFFF| 16MB   | Shared Memory / IPC
0x03000000-0x03FFFFFF| 16MB   | Disk I/O Buffer
0x10000000-0x100000FF| 256B   | UART / Serial Console
0x10000100-0x100001FF| 256B   | Keyboard Input Buffer
0x10000200-0x100002FF| 256B   | Mouse Input Buffer
```

## Development Workflow

### Option 1: Pure Pixel ISA (Educational)
1. Write programs in Pixel ISA assembly
2. Assemble to PNG
3. Load in browser → runs on GPU
4. See visual representation of code

### Option 2: Real Linux Boot
1. Build RISC-V Linux kernel
2. Encode as pixel data
3. Load in RISC-V emulator (WGSL)
4. Boot real Linux → all on GPU!

### Option 3: Hybrid (Recommended)
1. Use Pixel ISA for system apps
2. RISC-V for Linux kernel
3. Best of both worlds

## Performance Targets

| Metric | Target | Actual |
|--------|--------|--------|
| Boot Time (Pixel ISA) | < 100ms | TBD |
| Boot Time (Linux) | < 5s | TBD |
| Window Composite @ 60fps | < 16ms | TBD |
| Instruction Throughput | > 1M ops/sec | TBD |
| VRAM Access Latency | < 1ms | TBD |

## Roadmap

### ✅ Phase 1: Foundation (Current)
- [x] pxOS bootloader (can boot Linux)
- [ ] Pixel ISA v2 specification
- [ ] Basic assembler (pxlas.py)
- [ ] WGSL interpreter

### 🚧 Phase 2: Window Manager
- [ ] Window Table implementation
- [ ] pxl_wm.px (window manager)
- [ ] HAC compositor (WGSL)
- [ ] Basic UI toolkit

### 🔮 Phase 3: RISC-V Emulator
- [ ] rv32ima decoder (WGSL)
- [ ] MMU implementation
- [ ] UART device
- [ ] Boot Tiny Core Linux

### 🌟 Phase 4: System Apps
- [ ] pxl_desktop.px (launcher)
- [ ] pxl_term.px (terminal)
- [ ] pxl_paint.px (paint app)
- [ ] pxl_browser.px (web browser)

## Contributing

We welcome contributions! Focus areas:
- 🎨 Pixel encoding optimizations
- 🖼️ Window manager features
- ⚡ WGSL performance tuning
- 🐧 Linux kernel integration
- 📚 Documentation

## Resources

- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [WGSL Specification](https://www.w3.org/TR/WGSL/)
- [RISC-V ISA Manual](https://riscv.org/technical/specifications/)
- [Linux x86 Boot Protocol](https://www.kernel.org/doc/html/latest/x86/boot.html)

## License

MIT License - See LICENSE file

---

**VRAM OS**: Where pixels are programs, and the GPU is the CPU.

*"The future of computing is visual, parallel, and runs on the GPU."*
