# 🚀 pxOS + pxVM: Self-Expanding Learning Systems

**A revolutionary approach to AI knowledge accumulation**

---

## 🎯 The Big Picture

This repository contains **two complementary systems**:

### 1. **pxOS v1.0** - Minimal Bootable OS
   - Location: `pxos-v1.0/`
   - A primitive-built x86 bootloader with interactive shell
   - Educational foundation for OS development
   - See: [pxos-v1.0/README.md](pxos-v1.0/README.md)

### 2. **pxVM v0.5.0** - Self-Expanding Pixel Networks ⭐ NEW
   - Location: `pxvm/`
   - LM Studio integration for local AI learning loops
   - **Neural networks as PNG files**
   - **Knowledge that grows with every conversation**
   - See: [pxvm/README.md](pxvm/README.md)

---

## 🌟 What Makes pxVM Special?

### Traditional LLMs
```
User Query → LLM → Response
                   ↓
                 [Lost forever]
```
- **Stateless**: Every conversation starts fresh
- **Generic**: No specialization
- **Expensive**: Cloud API costs
- **Black box**: Can't inspect knowledge

### pxVM Self-Expanding Networks
```
User Query → [Read Pixel Network] → LLM → Response
                    ↑                        ↓
                    └────── [Append as Pixels]
```
- ✅ **Persistent**: Remembers every conversation
- ✅ **Growing**: Gets smarter with use
- ✅ **Local**: No cloud dependency
- ✅ **Visual**: Open PNG to see what it learned
- ✅ **Shareable**: Export/import trained networks

---

## 🚀 Quick Start: Self-Expanding Learning Loop

### Prerequisites

1. **Install LM Studio**
   - Download: https://lmstudio.ai/
   - Load any model (Mistral, Llama, Phi, etc.)
   - Start server on `localhost:1234`

2. **Install dependencies**
   ```bash
   pip install -r pxvm/requirements.txt
   ```

### Run Your First Learning Loop

```bash
# Demo the learning improvement
python3 pxvm/integration/lm_studio_bridge.py --demo

# Start interactive mode
python3 pxvm/integration/lm_studio_bridge.py --interactive
```

### What You'll See

```
🧑 You: What is pxOS?
📖 Reading pixel context...
🤖 LLM: [Answer]
💾 Appending to pixel network...
   ✅ Network expanded: 150 → 175 rows (+25)

🧑 You: How does it work?
📖 Reading pixel context...
🤖 LLM: [Answer with context from previous Q&A!]
💾 Appending to pixel network...
   ✅ Network expanded: 175 → 205 rows (+30)

💡 Network has learned from 2 conversations!
```

---

## 📊 The Self-Expanding Loop

```
Month 1:  200 rows   → Basic knowledge
Month 3:  2,000 rows → Getting smart
Month 6:  5,000 rows → Domain expert
Year 1:   20,000 rows → True specialist
```

The network becomes **your personal AI that grows with you**.

---

## 🏗️ Project Structure

```
pxos/
├── README.md                          # Main project README
├── PXVM_OVERVIEW.md                   # This file
│
├── pxos-v1.0/                         # Bootable OS (v1.0)
│   ├── README.md                      # OS documentation
│   ├── build_pxos.py                  # Build system
│   ├── pxos_commands.txt              # Primitive source code
│   ├── pxos.bin                       # Bootable binary
│   ├── tests/                         # Boot tests
│   ├── docs/                          # OS documentation
│   └── examples/                      # OS examples
│
└── pxvm/                              # Self-expanding networks (v0.5.0)
    ├── README.md                      # pxVM documentation
    ├── SETUP.md                       # Setup guide
    ├── requirements.txt               # Python dependencies
    │
    ├── integration/
    │   ├── __init__.py
    │   └── lm_studio_bridge.py       # LM Studio bridge ⭐
    │
    ├── learning/
    │   ├── __init__.py
    │   └── append.py                 # Text-to-pixel rendering
    │
    ├── networks/
    │   └── learning_network.png      # Growing network file
    │
    └── examples/
        ├── __init__.py
        └── quick_start.py            # Usage examples
```

---

## 🎯 Use Cases

### 1. **Personal Knowledge Base**
```bash
# Accumulate your research notes
bridge.append_interaction("What is X?", "X is...")
bridge.append_interaction("How does Y work?", "Y works by...")

# Future queries have access to all past notes!
```

### 2. **Code Assistant**
```bash
# Teach it your codebase
bridge.append_interaction(
    "What's our API structure?",
    "REST endpoints at /api/v1/..."
)

# Now it knows your API for future questions!
```

### 3. **Learning Companion**
```bash
# Each study session builds on the last
bridge.conversational_loop()

# "What did we cover last time?"
# LLM can reference previous sessions!
```

### 4. **Team Knowledge Sharing**
```bash
# Export your trained network
bridge.export_knowledge("team_knowledge.png")

# Team members import it
other_bridge.import_knowledge("team_knowledge.png")

# Everyone benefits from accumulated knowledge!
```

---

## 💡 Key Innovations

### 1. **Append-Only Learning**
- Knowledge is never lost
- Network only grows, never shrinks
- Complete audit trail of learning

### 2. **Visual Knowledge Storage**
- Open the PNG to literally see what it learned
- Inspectable and debuggable
- Can extract/audit stored knowledge

### 3. **Local-First Architecture**
- No cloud dependency
- Complete privacy
- No API costs

### 4. **Shareable Intelligence**
- Export trained networks
- Share with colleagues
- Build specialized network libraries

---

## 🔬 Technical Deep Dive

### How It Works

1. **Text → Pixels**
   ```python
   text = "Q: What is pxOS?\nA: It's a pixel-based OS."
   pixels = render_text_to_rows(text, width=1024)
   # Result: RGBA numpy array (rows × 1024 × 4)
   ```

2. **Append to Network**
   ```python
   existing = load_network("network.png")  # 150 rows
   new_pixels = render_interaction(q, a)    # 25 rows
   expanded = vstack([existing, new_pixels]) # 175 rows
   save_network(expanded, "network.png")
   ```

3. **Read Context**
   ```python
   context = read_pixel_context("network.png")
   # Extract accumulated knowledge from pixel rows
   ```

4. **Query with Context**
   ```python
   messages = [
       {"role": "system", "content": context},
       {"role": "user", "content": query}
   ]
   response = lm_studio.query(messages)
   ```

### The Magic: Contextual Learning

**First Conversation:**
- Network: 150 rows (generic knowledge)
- Query: "What is pxOS?"
- LLM: Generic answer
- **Network grows to 175 rows**

**Second Conversation:**
- Network: 175 rows (includes previous pxOS Q&A)
- Query: "How does quantization work?"
- LLM: **More informed answer** (references pxOS context!)
- **Network grows to 200 rows**

**Tenth Conversation:**
- Network: 350 rows (9 previous conversations!)
- Query: "Debug my shader"
- LLM: **Expert answer** using accumulated knowledge
- **Network is now a domain specialist**

---

## 📚 Documentation

### pxVM (Self-Expanding Networks)
- **Main Documentation**: [pxvm/README.md](pxvm/README.md)
- **Setup Guide**: [pxvm/SETUP.md](pxvm/SETUP.md)
- **Quick Start**: `python3 pxvm/examples/quick_start.py`

### pxOS (Bootable OS)
- **OS Documentation**: [pxos-v1.0/README.md](pxos-v1.0/README.md)
- **Architecture**: `pxos-v1.0/docs/architecture.md`
- **Extensions**: `pxos-v1.0/docs/extensions.md`

---

## 🎓 Learning Path

### For Beginners
1. Start with pxOS v1.0 (understand primitives)
2. Read pxVM concepts (pixels as knowledge)
3. Run pxVM examples (see learning in action)
4. Experiment with your own networks

### For Advanced Users
1. Implement semantic search (v0.5.0 full)
2. Build multi-network orchestration
3. Create specialized domain networks
4. Contribute to the project!

---

## 🚀 Future Roadmap

### pxVM v0.6.0 (Planned)
- [ ] Semantic search over pixel rows
- [ ] Knowledge export/import utilities
- [ ] Multi-network orchestration
- [ ] OCR for text extraction from pixels
- [ ] Network visualization tools

### pxVM v1.0 (Vision)
- [ ] GPU-accelerated pixel networks
- [ ] Neural networks executing as PNG files
- [ ] Font-code protocol (ASCII opcodes)
- [ ] Per-matrix quantization
- [ ] True GPU-native computation

---

## 🤝 Contributing

We welcome contributions to both systems!

**pxVM Ideas:**
- Semantic search implementation
- Alternative LLM backends (Ollama, etc.)
- Knowledge visualization tools
- Network compression algorithms
- Export/import utilities

**pxOS Ideas:**
- Command parser implementation
- New primitive commands
- Module system design
- Protected mode support
- FAT12 filesystem driver

---

## 📄 License

MIT License - See [LICENSE](pxos-v1.0/LICENSE)

---

## 🌟 Why This Matters

Traditional AI systems are **stateless and generic**. You pay for the same generic knowledge every time.

With pxVM, you're building **your own specialized AI** that:
- Remembers every conversation
- Gets smarter with use
- Runs completely local
- Costs nothing after setup
- Can be shared and collaborated on

**It's not just an AI assistant. It's a growing knowledge base that becomes uniquely yours.**

---

## 🎉 Get Started Now!

### pxVM Self-Expanding Networks
```bash
# Install dependencies
pip install -r pxvm/requirements.txt

# Start LM Studio (localhost:1234)

# Run interactive learning loop
python3 pxvm/integration/lm_studio_bridge.py --interactive
```

### pxOS Bootable System
```bash
# Build and boot
cd pxos-v1.0
python3 build_pxos.py
./tests/boot_qemu.sh
```

---

## 📖 Key Resources

- **pxVM README**: [pxvm/README.md](pxvm/README.md) - Complete pxVM documentation
- **Setup Guide**: [pxvm/SETUP.md](pxvm/SETUP.md) - Step-by-step setup
- **pxOS README**: [pxos-v1.0/README.md](pxos-v1.0/README.md) - OS documentation
- **LM Studio**: https://lmstudio.ai/ - Local LLM runtime

---

## 💬 Questions?

**For pxVM:**
- See [pxvm/README.md](pxvm/README.md)
- Check [pxvm/SETUP.md](pxvm/SETUP.md)
- Run examples: `python3 pxvm/examples/quick_start.py`

**For pxOS:**
- See [pxos-v1.0/README.md](pxos-v1.0/README.md)
- Check `pxos-v1.0/docs/` directory
- Run tests: `./tests/boot_qemu.sh`

---

**Made with ❤️ for the future of computing**

*"Where pixels are primitives and knowledge grows with every conversation."*

---

🚀 **Start your learning loop today!**
