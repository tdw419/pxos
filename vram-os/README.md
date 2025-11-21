# VRAM OS - Pixel-Based Operating System

**VRAM OS** is a novel operating system architecture where programs and data are stored directly as **pixels in a 2D memory grid**, with instructions encoded in **RGBA color channels**. It's designed for GPU-native execution and spatial code representation.

---

## 🎯 **Key Concept**

Traditional OS:
```
Memory: [0x00, 0x01, 0x02, ...] (1D byte array)
Instructions: Sequential byte codes
```

VRAM OS:
```
Memory: Pixel grid (X, Y) coordinates
Instructions: RGBA values (R=opcode, G=operand_type, B=operand1, A=operand2)
```

Every pixel is a potential instruction, data value, or display element.

---

## ✅ **Status: Working Foundation**

- ✅ VRAMState storage system
- ✅ PXL-ISA v0.1 instruction set
- ✅ CPU-based interpreter
- ✅ All tests passing
- 🚧 Bootloader in progress
- 🚧 GPU shader execution (planned)

---

## 🏗️ **Architecture**

### **VRAMState** - Unified Memory Space

```python
VRAMState (1024 x 1024 pixels = 4MB RGBA memory)

Memory Map:
┌─────────────────────────────────────────┐
│ Bootloader    (0,0)     [64 x 1 pixels] │
│ Kernel        (0,1)     [1024 x 256]    │
│ Registers     (0,257)   [16 x 1]        │
│ Stack         (0,258)   [256 x 256]     │
│ Heap          (256,258) [768 x 768]     │
└─────────────────────────────────────────┘
```

### **PXL-ISA v0.1** - Pixel Instruction Set

Each instruction is encoded as an RGBA pixel:

```
R channel (8-bit): Opcode
G channel (8-bit): Operand type / flags
B channel (8-bit): Operand 1
A channel (8-bit): Operand 2
```

**Example:**
```
LOAD R0, #42  →  RGBA(16, 0, 0, 42)
  R=16  (LOAD opcode)
  G=0   (IMMEDIATE mode)
  B=0   (Destination: R0)
  A=42  (Value: 42)
```

### **Supported Instructions**

| Opcode | Instruction | Description |
|--------|-------------|-------------|
| 0x00 | NOP | No operation |
| 0x01 | JMP | Jump to address |
| 0x02 | JMP_IF | Conditional jump |
| 0x03 | CALL | Call subroutine |
| 0x04 | RET | Return |
| 0x10 | LOAD | Load value to register |
| 0x11 | STORE | Store register to memory |
| 0x12 | MOV | Move between registers |
| 0x20 | ADD | Addition |
| 0x21 | SUB | Subtraction |
| 0x30 | AND | Bitwise AND |
| 0x31 | OR | Bitwise OR |
| 0x50 | PIXEL_READ | Read pixel from (X,Y) |
| 0x51 | PIXEL_WRITE | Write pixel to (X,Y) |
| 0xFF | HALT | Stop execution |

---

## 🚀 **Quick Start**

### **1. Test the Core Components**

```bash
cd vram-os

# Test VRAMState
python3 core/vram_state.py

# Test PXL-ISA encoder/decoder
python3 core/pxl_isa.py

# Test interpreter (runs a program!)
python3 test_interpreter.py
```

### **2. Write Your First Program**

```python
from core.vram_state import VRAMState
from core.pxl_isa import PXLAssembler, Register
from core.interpreter import PXLInterpreter

# Create VRAM
vram = VRAMState(1024, 1024)

# Write program
asm = PXLAssembler()
program = [
    asm.assemble_load_imm(Register.R0, 10),   # R0 = 10
    asm.assemble_load_imm(Register.R1, 32),   # R1 = 32
    asm.assemble_add(Register.R0, Register.R1),  # R0 += R1
    asm.assemble_halt()
]

# Write to bootloader region
for i, instr in enumerate(program):
    r, g, b, a = instr.to_pixel()
    vram.write_pixel(i, 0, r, g, b, a)

# Execute!
interpreter = PXLInterpreter(vram)
interpreter.run(start_x=0, start_y=0, debug=True)

# R0 now equals 42
```

---

## 📊 **Test Results**

```
=== PXL-ISA Interpreter Test ===
Starting execution at (0, 0)...

Execution trace:
[0000,0000] LOAD R0, #10
[0001,0000] LOAD R1, #32
[0002,0000] ADD R0, R1
[0003,0000] PIXEL_WRITE (244, 244)
[0004,0000] HALT

Program halted after 5 instructions.

Final register state:
  R0 = 42  ✓
  R1 = 32  ✓

✓ Test PASSED!
```

---

## 🎨 **Why Pixel-Based?**

### **Advantages**

1. **GPU-Native Execution**
   - Programs stored as textures
   - Massive parallelism potential
   - Direct VRAM access

2. **Visual Debugging**
   - "See" your code as colored pixels
   - Execution flow visible as cursor movement
   - Data patterns emerge visually

3. **High Density**
   - 32 bits per pixel (RGBA)
   - Alpha channel fully utilized (not wasted on transparency)
   - Compact instruction encoding

4. **Spatial Encoding**
   - Code layout has semantic meaning
   - (X,Y) coordinates are addresses
   - Natural 2D data structures

### **Challenges**

- Non-traditional tooling required
- GPU programming complexity
- Debugging requires new paradigms
- Limited by display resolution for VRAM size

---

## 📁 **Project Structure**

```
vram-os/
├── core/
│   ├── vram_state.py      # Unified memory system
│   ├── pxl_isa.py         # Instruction set
│   └── interpreter.py     # Execution engine
├── tools/                 # Build utilities (coming soon)
├── programs/              # Example programs (coming soon)
├── docs/                  # Architecture docs (coming soon)
└── test_interpreter.py    # Integration tests
```

---

## 🔬 **Technical Details**

### **Memory Model**

- **Address Space**: 2D grid (X, Y coordinates)
- **Word Size**: 32-bit (RGBA pixel)
- **Endianness**: Little-endian (A is LSB, R is MSB for multi-byte values)
- **Alignment**: No alignment requirements (pixel-addressed)

### **Execution Model**

- **Program Counter**: (X, Y) coordinate pair
- **Sequential Flow**: PC increments X, wraps to next row when X >= width
- **Jumps**: Set PC to new (X, Y) coordinate
- **Registers**: 16 general-purpose (R0-R15) + special (PC, SP, FLAGS)

### **Data Types**

| Type | Size | Encoding |
|------|------|----------|
| Byte | 8-bit | Single channel (R, G, B, or A) |
| Word | 32-bit | Full pixel (RGBA) |
| Coordinate | 16-bit | Two channels (X in B, Y in A) |
| Instruction | 32-bit | RGBA (opcode + operands + flags) |

---

## 🛣️ **Roadmap**

### **Phase 1: Foundation** ✅ (Current)
- [x] VRAMState storage
- [x] PXL-ISA v0.1
- [x] CPU interpreter
- [x] Basic tests

### **Phase 2: Bootloader** 🚧 (In Progress)
- [ ] Minimal bootloader program
- [ ] String display primitives
- [ ] Input handling
- [ ] PNG-based program distribution

### **Phase 3: GPU Execution** (Planned)
- [ ] WebGPU compute shader implementation
- [ ] WGSL translation layer
- [ ] GPU-accelerated execution
- [ ] Performance benchmarks

### **Phase 4: Operating System** (Planned)
- [ ] Process management
- [ ] Memory allocation
- [ ] I/O system
- [ ] Standard library

---

## 🤝 **Comparison to pxOS x86**

This repository contains **two different OS projects**:

| Feature | **pxOS x86** | **VRAM OS** |
|---------|-------------|-------------|
| **Architecture** | x86 real-mode | Pixel-based |
| **Memory** | Linear (0x7C00...) | 2D grid (X,Y) |
| **Instructions** | x86 opcodes | RGBA pixels |
| **Execution** | CPU (BIOS interrupts) | GPU/CPU interpreter |
| **Encoding** | Byte primitives (WRITE) | Pixel assembler |
| **Status** | ✅ Bootable on real hardware | 🚧 Interpreter works |

Both use primitive-based build systems but target completely different platforms.

---

## 📚 **Learn More**

- [PXL-ISA Specification](docs/pxl-isa-spec.md) (coming soon)
- [VRAM Memory Layout](docs/memory-layout.md) (coming soon)
- [GPU Execution Model](docs/gpu-execution.md) (coming soon)

---

## 🧪 **Contributing**

Ideas for contributions:
- Additional PXL-ISA instructions
- GPU shader implementation
- Example programs
- Optimization techniques
- Visual debugging tools
- Documentation

---

## 📜 **License**

MIT License

---

## 🎓 **Educational Purpose**

VRAM OS is an educational project exploring:
- Alternative OS architectures
- GPU computing (GPGPU)
- Spatial code representation
- Visual programming paradigms
- Pixel-level computation

*"What if pixels weren't just for display, but for execution?"*

---

**Built with research from spatial vs. sequential encoding, pixel density optimization, and GPU acceleration principles.**
