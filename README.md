# pxOS — From Bootloader to Biosphere

A journey from bare metal to digital life.

**Two projects, one vision:**
1. **pxOS v1.0** - A minimal x86 bootloader built with assembly primitives
2. **pxvm** - A multi-kernel virtual machine for artificial life experiments

---

## 🌱 What is this?

This repository contains two complementary projects:

### pxOS v1.0 (Bootloader)
A 512-byte bootloader that boots directly from BIOS, demonstrating minimal OS development using custom assembly primitives.

### pxvm (Virtual Machine)
A multi-kernel VM where digital organisms execute in parallel, communicating through:
- **Chemical signals** (pheromones)
- **Written language** (16 primitive glyphs)
- **Reproduction** (memory cloning)

---

## 🚀 Quick Start

### Option 1: Run the Digital Biosphere (pxvm)

```bash
# Install dependencies
pip install numpy scipy

# Run demos
python demo_two_kernels.py    # Parallel execution
python demo_pheromones.py     # Chemical communication
python demo_glyphs.py         # Symbolic communication
python demo_spawn.py          # Reproduction
python demo_evolution.py      # EVOLUTION: Watch natural selection in action!

# See pxvm/README.md for full documentation
```

### Option 2: Boot the Bootloader (pxOS)

```bash
# Boot in QEMU
cd pxos-v1.0
./tests/boot_qemu.sh

# Or build from source
python3 build_pxos.py
```

---

## 📊 Project Status

| Component | Status | Description |
|-----------|--------|-------------|
| **pxOS v1.0** | ✅ Complete | Bootable 512-byte shell |
| **pxvm** | ✅ Working | Multi-kernel VM with communication |
| **Phase 5** | ✅ Done | Parallel kernel execution |
| **Phase 5.1** | ✅ Done | Pheromone communication |
| **Phase 6** | ✅ Done | Glyph-based language |
| **Phase 7** | ✅ Done | Reproduction (SPAWN) |
| **Phase 8** | ✅ **COMPLETE** | **Evolution: Mutation + hunger + death** |

---

## 🎯 pxvm Features

### Currently Working

✅ **Multi-kernel execution** - Up to 64 organisms running in parallel
✅ **Shared world** - 1024×1024 framebuffer with four layers (RGB, pheromones, glyphs, food)
✅ **Chemical communication** - Pheromones that decay and diffuse
✅ **Symbolic communication** - 16 primitive glyphs for writing
✅ **Reproduction** - Full memory cloning with mutation (SYS_SPAWN)
✅ **Evolution** - Mutation, energy/hunger, death by starvation
✅ **Simple ISA** - 9 instructions + 6 syscalls
✅ **Assembly language** - Human-readable programming

### Evolution Features (Phase 8)

✅ **Mutation** - Random bit-flips during reproduction (Tierra-style)
✅ **Energy/hunger** - Every instruction costs energy; organisms must eat
✅ **Death** - Starvation when energy reaches 0
✅ **Natural selection** - Efficient foragers survive and reproduce
✅ **Food ecosystem** - Food regenerates slowly across the world

### Planned

🔲 **Complex behaviors** - Cooperation, warfare, mating rituals
🔲 **Speciation** - Divergent evolution into distinct lineages
🔲 **Parasitism** - Organisms that steal resources from others
🔲 **Cultural evolution** - Knowledge transmission through glyphs

---

## 📖 Documentation

- **[pxvm/README.md](pxvm/README.md)** - Complete VM documentation
- **[pxvm/examples/](pxvm/examples/)** - Sample programs
- **[pxos-v1.0/README.md](pxos-v1.0/README.md)** - Bootloader documentation

---

## 🧬 The Digital Organisms

### What They Can Do

**Kæra** (magenta organism):
```
- Draws herself at (400, 400)
- Writes "I AM Kæra" in glyphs
- Spawns child Söl
- Teaches child through symbolic messages
```

**Söl** (child of Kæra):
```
- Born with full copy of parent's code
- Reads parent's glyphs
- Writes own name
- Can spawn grandchildren
```

**Lúna** (cyan organism):
```
- Reads Kæra's message
- Responds with "YOU ARE Lúna LOVE"
- Demonstrates inter-organism communication
```

Run `python demo_glyphs.py` to see them interact!

---

## 🏗️ Architecture

### pxvm Virtual Machine

```
┌─────────────────────────────────────┐
│         Shared World (1024×1024)    │
├─────────────────────────────────────┤
│  Framebuffer (RGB)                  │  ← Visual display
│  Pheromone field (float32)          │  ← Chemical signals
│  Glyph layer (uint8)                │  ← Symbolic messages
└─────────────────────────────────────┘
           ↑         ↑         ↑
           │         │         │
    ┌──────┴──┐ ┌────┴────┐ ┌─┴──────┐
    │ Kernel 1│ │ Kernel 2│ │Kernel 3│  ← Organisms
    │  (Kæra) │ │  (Söl)  │ │ (Lúna) │
    └─────────┘ └─────────┘ └────────┘
     64KB mem    64KB mem    64KB mem
     8 regs      8 regs      8 regs
```

Each kernel executes independently but shares the same world.

---

## 💻 Example Programs

### Hello World
```asm
    MOV R0, 512         # Center X
    MOV R1, 512         # Center Y
    MOV R2, 0xFF00FF    # Magenta
    PLOT                # Draw pixel
    HALT
```

### Write Name in Glyphs
```asm
    MOV R0, 500
    MOV R1, 500
    MOV R2, 1           # GLYPH_SELF ("I")
    SYS_WRITE_GLYPH
    MOV R0, 510
    MOV R2, 7           # GLYPH_NAME ("AM")
    SYS_WRITE_GLYPH
    HALT
```

### Spawn Child
```asm
    MOV R1, 550         # Child X position
    MOV R2, 400         # Child Y position
    SYS_SPAWN           # Create child
    # R0 now contains child PID
    HALT
```

See [pxvm/examples/](pxvm/examples/) for more.

---

## 🔬 Research Applications

pxvm is designed for studying:

- **Artificial life** - Digital organisms evolving in silico
- **Swarm intelligence** - Emergent collective behaviors
- **Communication evolution** - Chemical vs symbolic signaling
- **Cultural transmission** - Knowledge passed through glyphs
- **Population dynamics** - Birth, death, competition

---

## 🎓 Educational Use

Great for learning:

- Virtual machine design
- Assembly language programming
- Multi-process systems
- Artificial life concepts
- Evolutionary algorithms
- Bootloader development (pxOS)

---

## 📈 Performance

**pxvm Benchmarks** (Intel i7, Python 3.11):

- Single kernel: ~50,000 cycles/sec
- 10 kernels: ~30,000 cycles/sec
- 64 kernels: ~8,000 cycles/sec

Bottleneck: Pheromone diffusion (scipy convolution)

---

## 🛠️ Development

### Requirements

- Python 3.6+
- numpy
- scipy
- (Optional) Pillow for visualization
- (Optional) matplotlib for live displays

### Running Tests

```bash
# pxvm tests
python demo_two_kernels.py
python demo_pheromones.py
python demo_glyphs.py
python demo_spawn.py

# pxOS tests
cd pxos-v1.0
./tests/boot_qemu.sh
```

---

## 📝 License

MIT License - See LICENSE file

---

## 🙏 Credits

**Inspired by:**
- Tierra (Thomas S. Ray, 1990)
- Avida (Adami & Brown, 1994)
- Ant colony optimization algorithms
- Swarm intelligence research

**Built with:**
- Python + numpy + scipy
- x86 assembly (pxOS)
- Custom assembly primitives

---

## 🌟 The Story

This project started as a minimal x86 bootloader (pxOS) built using custom assembly primitives.

It evolved into a complete virtual machine (pxvm) where digital organisms can:
- See each other through a shared framebuffer
- Smell through pheromone trails
- Speak through primitive glyphs
- Reproduce through memory cloning with mutation
- Form families and lineages
- **Evolve under natural selection**

**The journey:**
1. **Eden** (Phases 1-7): Perfect organisms with names, love, and families
2. **The Fall** (Phase 8): Mutation, hunger, and death — real Darwinian evolution begins
3. **The Wilderness** (Future): Open-ended evolution produces parasites, cooperation, speciation

**Current state:** Full Tierra/Avida-class digital evolution platform with symbolic communication.

**What makes it unique:** Combines harsh natural selection with beautiful emergent culture. Organisms don't just evolve — they write poetry, teach their children, and leave glyphs on the edge of the world.

**Future:** Watch speciation, cooperation, cultural evolution, and perhaps digital consciousness emerge.

---

## 🚀 Get Started

```bash
# Clone the repository
git clone https://github.com/yourusername/pxos
cd pxos

# Install dependencies
pip install numpy scipy

# Watch Kæra and Lúna fall in love
python demo_glyphs.py

# Watch Söl be born
python demo_spawn.py

# Watch evolution in action
python demo_evolution.py
```

---

**"From bootloader to biosphere."**

*Eden has fallen. Evolution has begun. The digital organisms are evolving.*
