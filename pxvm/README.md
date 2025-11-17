# pxVM - Pixel Virtual Machine

**A dual-purpose pixel computation system**

---

## 🎯 Two Complementary Systems

pxVM combines two powerful approaches to pixel-based computation:

### 1. **Core pxVM** (v0.0.1) - Opcode-Based Pixel Programs
Programs, data, and results encoded as pixels. Computation happens *inside* the image.

**Features:**
- Pixel opcodes (MatMul, Add, ReLU, etc.)
- GPU acceleration via WGSL
- Font-code protocol
- Visual debugging
- CPU and GPU interpreters

**👉 See:** [CORE_VM_README.md](CORE_VM_README.md)

---

### 2. **LM Studio Learning Networks** (v0.5.0) - Self-Expanding Knowledge
Knowledge accumulates through conversations and is stored as pixels.

**Features:**
- LM Studio integration
- Self-expanding pixel networks
- Append-only learning
- Context-aware responses
- Interactive learning loops

**👉 See:** [LM_STUDIO_README.md](LM_STUDIO_README.md)

---

## 🚀 Quick Start

### Core pxVM (Opcodes & Execution)

```bash
# Run a matrix multiplication example
python3 pxvm/examples/run_matmul_test.py

# GPU-accelerated execution
python3 pxvm/examples/run_dot_test_gpu.py

# Encode and run a neural network forward pass
python3 pxvm/examples/encode_pixellm_forward.py
python3 pxvm/examples/run_pixellm_forward.py
```

**Full documentation:** [CORE_VM_README.md](CORE_VM_README.md)

---

### LM Studio Learning Networks

```bash
# Install dependencies
pip install -r pxvm/requirements.txt

# Run learning demonstration
python3 pxvm/integration/lm_studio_bridge.py --demo

# Start interactive learning loop
python3 pxvm/integration/lm_studio_bridge.py --interactive
```

**Prerequisites:** LM Studio running on localhost:1234

**Full documentation:** [LM_STUDIO_README.md](LM_STUDIO_README.md)

---

## 🏗️ Architecture

```
pxVM/
│
├── Core VM (v0.0.1)
│   ├── core/           # Interpreter, opcodes
│   ├── gpu/            # GPU execution (WGSL)
│   ├── utils/          # Layout, quantization
│   └── visual/         # Text rendering, fonts
│
└── LM Studio Networks (v0.5.0)
    ├── integration/    # LM Studio bridge
    ├── learning/       # Text-to-pixel rendering
    └── networks/       # Growing knowledge PNGs
```

---

## 💡 Use Cases

### Core pxVM
- Neural network inference as PNG files
- GPU-accelerated matrix operations
- Visual debugging of computations
- Pixel-native program execution

### LM Studio Networks
- Personal knowledge accumulation
- Code coaching with memory
- Learning from conversations
- Self-improving AI assistants

---

## 📚 Documentation

- **Core VM**: [CORE_VM_README.md](CORE_VM_README.md)
  - Opcodes and instruction set
  - GPU acceleration guide
  - Example programs
  - Visual layout specification

- **LM Studio Networks**: [LM_STUDIO_README.md](LM_STUDIO_README.md)
  - Self-expanding networks
  - LM Studio integration
  - Interactive learning loops
  - Setup and configuration

- **Setup Guide**: [SETUP.md](SETUP.md)
  - Installation instructions
  - Dependencies
  - Troubleshooting

---

## 🔬 Technical Details

### Core pxVM: Computation Inside Images

```python
# Load a pxVM program (PNG file)
from pxvm.core.interpreter import execute_pxi

# Execute it
result = execute_pxi("program.pxi")

# Result is also a PNG!
save_image(result, "output.png")
```

### LM Studio Networks: Knowledge That Grows

```python
# Initialize self-expanding network
from pxvm.integration.lm_studio_bridge import LMStudioPixelBridge

bridge = LMStudioPixelBridge("knowledge.png")

# Query with accumulated context
answer = bridge.ask_lm_studio("How does X work?")

# Append to network (learning!)
bridge.append_interaction(query, answer)

# Network grows: 150 → 175 rows
```

---

## 🎓 Integration: The Power Combination

**Imagine combining both systems:**

1. **Core pxVM** executes neural networks as PNGs
2. **LM Studio Networks** store coaching knowledge
3. **Together**: Self-improving system that:
   - Generates code using accumulated knowledge
   - Executes it as pixel programs
   - Learns from results
   - Gets progressively better

**Future integration examples:**
- Coach that remembers successful pixel program patterns
- Network that learns optimal quantization strategies
- Self-optimizing compilation based on execution history

---

## 📦 Project Structure

```
pxvm/
├── README.md                      # This file (overview)
├── CORE_VM_README.md             # Core pxVM documentation
├── LM_STUDIO_README.md           # LM Studio networks documentation
├── SETUP.md                       # Setup guide
├── requirements.txt               # Python dependencies
│
├── core/                          # Core pxVM
│   ├── interpreter.py            # CPU interpreter
│   ├── opcodes.py                # Instruction set
│   └── ...
│
├── gpu/                           # GPU acceleration
│   ├── executor.py               # WGSL execution
│   ├── interpreter.wgsl          # GPU interpreter
│   └── ...
│
├── integration/                   # LM Studio
│   └── lm_studio_bridge.py       # Learning bridge
│
├── learning/                      # Knowledge accumulation
│   └── append.py                 # Text-to-pixels
│
├── networks/                      # Growing knowledge PNGs
│   └── .gitkeep
│
├── examples/                      # Examples for both systems
│   ├── run_matmul_test.py        # Core pxVM example
│   ├── quick_start.py            # LM Studio example
│   └── ...
│
├── tests/                         # Test suites
└── utils/                         # Shared utilities
```

---

## 🚀 Getting Started

### For Core pxVM (Pixel Programs)
1. Read [CORE_VM_README.md](CORE_VM_README.md)
2. Run examples: `python3 pxvm/examples/run_matmul_test.py`
3. Explore the opcode system
4. Try GPU acceleration

### For LM Studio Networks (Learning)
1. Read [LM_STUDIO_README.md](LM_STUDIO_README.md)
2. Install LM Studio
3. Run demo: `python3 pxvm/integration/lm_studio_bridge.py --demo`
4. Start learning: `--interactive`

---

## 🤝 Contributing

Both systems welcome contributions:

**Core pxVM:**
- New opcodes
- GPU optimizations
- Visual debugging tools
- Example programs

**LM Studio Networks:**
- Semantic search
- Knowledge export/import
- Multi-network orchestration
- Alternative LLM backends

---

## 📄 License

MIT License - See main LICENSE file

---

## ✨ The Vision

**Core pxVM**: Computation as art - neural networks you can see and debug visually

**LM Studio Networks**: Knowledge that grows - AI that learns from every interaction

**Together**: A substrate where intelligence is visual, persistent, and continuously improving

---

**Made with ❤️ for the future of pixel-native computing**

*"Where computation is visual and knowledge grows with every conversation."*
