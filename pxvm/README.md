# pxVM: Self-Expanding Pixel Networks

**Version 0.5.0 - The Learning Loop**

---

## 🎯 The Vision

pxVM creates a **self-contained learning loop** where:

1. **LM Studio LLM generates knowledge**
2. **Knowledge gets pixelated and stored in the network** (as PNG files)
3. **Network reads its own pixels for context**
4. **System gets smarter with every interaction**

This is a closed system where the network becomes progressively more valuable over time.

---

## 🚀 Quick Start

### Prerequisites

1. **LM Studio** running locally
   - Download from: https://lmstudio.ai/
   - Load any model (Mistral, Llama, Phi, etc.)
   - Start the server (should run on `localhost:1234`)

2. **Python 3.7+** with dependencies:
   ```bash
   pip install numpy pillow requests
   ```

### Run Your First Learning Loop

```bash
# 1. Demo the learning improvement
python3 pxvm/integration/lm_studio_bridge.py --demo

# 2. Start interactive mode
python3 pxvm/integration/lm_studio_bridge.py --interactive

# 3. Run examples
python3 pxvm/examples/quick_start.py basic
python3 pxvm/examples/quick_start.py teaching
python3 pxvm/examples/quick_start.py interactive
```

---

## 🧠 How It Works

### The Self-Expanding Loop

```
┌─────────────────────────────────────────────────────┐
│  USER QUERY                                         │
│  "What is pxOS?"                                    │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  PIXEL NETWORK (PNG file)                           │
│  • Read accumulated knowledge                       │
│  • Extract context from previous Q&As               │
│  • 150 rows of learned conversations                │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  LM STUDIO (Local LLM)                              │
│  • Receives: query + pixel context                  │
│  • Generates: informed answer                       │
│  • Returns: response text                           │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  APPEND TO NETWORK                                  │
│  • Render Q&A as pixels (text → rows)               │
│  • Append to existing PNG                           │
│  • Network grows: 150 → 175 rows                    │
└─────────────────────────────────────────────────────┘
```

### The Magic: It Gets Smarter!

**Conversation 1:**
- Network: 150 rows (generic knowledge)
- Query: "What is pxOS?"
- LLM: Generic answer
- **Network grows to 170 rows**

**Conversation 2:**
- Network: 170 rows (includes previous pxOS Q&A)
- Query: "How does quantization work?"
- LLM: More informed answer (references pxOS context!)
- **Network grows to 195 rows**

**Conversation 10:**
- Network: 350 rows (9 previous conversations as context!)
- Query: "Debug my shader"
- LLM: Expert answer using accumulated knowledge
- **Network is now a specialist in your domain**

---

## 📁 Project Structure

```
pxvm/
├── README.md                      # This file
├── integration/
│   └── lm_studio_bridge.py       # Main LM Studio integration
├── learning/
│   └── append.py                 # Text-to-pixel rendering
├── networks/
│   └── learning_network.png      # Self-expanding network (grows over time)
└── examples/
    └── quick_start.py            # Usage examples
```

---

## 🔧 Usage Examples

### Basic Interaction

```python
from pxvm.integration.lm_studio_bridge import LMStudioPixelBridge

# Initialize
bridge = LMStudioPixelBridge(
    network_path="my_network.png"
)

# Query with context
answer = bridge.ask_lm_studio("What is machine learning?")

# Append to network (learning!)
bridge.append_interaction("What is machine learning?", answer)

# Network has grown and will inform future queries!
```

### Teaching the Network

```python
# Teach specialized knowledge
bridge.append_interaction(
    "What is pxOS?",
    "pxOS is a GPU-native OS where pixels are computational primitives."
)

# Now the LLM knows about pxOS for future queries!
answer = bridge.ask_lm_studio("Explain pxOS architecture")
# Answer will include context from what you taught it!
```

### Interactive Learning Loop

```python
# Start conversational mode
bridge.conversational_loop()

# Each conversation:
# 1. User asks question
# 2. LLM answers with accumulated context
# 3. Q&A appended to pixel network
# 4. Network grows and gets smarter!
```

---

## 💡 Key Features

### ✅ Persistent Learning
- Unlike stateless LLMs, the network remembers every conversation
- Knowledge accumulates over time
- Sessions build on previous sessions

### ✅ Visual & Inspectable
- Open the PNG file to literally see what it learned
- Network size (rows) = learning progress
- Can extract/audit stored knowledge

### ✅ Fully Local
- No cloud dependency
- Complete privacy
- No API costs

### ✅ Shareable Knowledge
- Export your trained network
- Share with others
- Collaborate on specialized networks

---

## 🎓 Advanced Features (Coming in v0.6.0)

### Semantic Search
```python
def read_pixel_context_semantic(self, query: str) -> str:
    """Find most relevant past conversations."""
    # Extract all Q&As from pixels
    conversations = extract_all_conversations(self.network_path)

    # Semantic similarity matching
    relevant = find_top_k_similar(query, conversations, k=5)

    return "\n".join(relevant)
```

### Knowledge Export/Import
```python
# Export your learned network
bridge.export_knowledge("pxos_expert.png")

# Someone else imports it
other_bridge.import_knowledge("pxos_expert.png")
# Now they benefit from YOUR accumulated knowledge!
```

### Multi-Network Orchestration
```python
# Combine multiple specialized networks
orchestrator = NetworkOrchestrator([
    "pxos_expert.png",      # pxOS specialist
    "python_expert.png",    # Python specialist
    "ml_expert.png"         # ML specialist
])

# Query routes to most relevant network(s)
answer = orchestrator.ask("How do I implement a neural network in pxOS?")
# Uses pxos_expert.png + ml_expert.png contexts!
```

---

## 🔬 Technical Details

### Text-to-Pixel Rendering

The system renders text as monospaced pixel rows:

```python
from pxvm.learning.append import render_text_to_rows

text = "Q: What is pxOS?\nA: It's a pixel-based OS."
pixels = render_text_to_rows(text, width=1024, max_lines=20)

# Result: RGBA numpy array (20 × 1024 × 4)
# Can be appended to existing network
```

### Network Growth

```python
# Load existing network
img_array = np.array(Image.open("network.png"))
old_height = img_array.shape[0]  # e.g., 150 rows

# Render new interaction
new_pixels = render_text_to_rows(interaction, width=1024)

# Append vertically
expanded = np.vstack([img_array, new_pixels])
new_height = expanded.shape[0]  # e.g., 175 rows

# Save expanded network
Image.fromarray(expanded).save("network.png")
```

### LM Studio API Integration

```python
# Query with context
response = requests.post(
    "http://localhost:1234/v1/chat/completions",
    json={
        "model": "local-model",
        "messages": [
            {"role": "system", "content": pixel_context},
            {"role": "user", "content": query}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
)

answer = response.json()["choices"][0]["message"]["content"]
```

---

## 📊 Growth Trajectory

```
Month 1:  200 rows   → Basic knowledge
Month 3:  2,000 rows → Getting smart
Month 6:  5,000 rows → Domain expert
Year 1:   20,000 rows → True specialist
```

The network becomes progressively more valuable. It's **your personal AI that grows with you**.

---

## 🎯 Why This Is Powerful

### Traditional LLM
- ❌ Stateless (forgets between sessions)
- ❌ Generic (no specialization)
- ❌ Expensive (cloud API costs)
- ❌ Black box (can't inspect knowledge)

### Self-Expanding Pixel Network
- ✅ **Persistent**: Remembers every conversation
- ✅ **Growing**: Gets smarter with use
- ✅ **Local**: No cloud dependency
- ✅ **Visual**: You can see the learning (PNG grows)
- ✅ **Inspectable**: Open PNG, see what it learned
- ✅ **Shareable**: Export/import trained networks

---

## 🔧 Configuration

### Custom Network Path

```bash
python3 pxvm/integration/lm_studio_bridge.py \
    --network ~/my_networks/custom.png \
    --interactive
```

### Custom LM Studio URL

```bash
python3 pxvm/integration/lm_studio_bridge.py \
    --url http://localhost:8080/v1 \
    --interactive
```

---

## 🐛 Troubleshooting

### "Cannot connect to LM Studio"

**Solution:**
1. Verify LM Studio is running
2. Check it's on port 1234
3. Test with: `curl http://localhost:1234/v1/models`

### "No pixels generated"

**Solution:**
- Install Pillow: `pip install pillow`
- Check font availability: `/usr/share/fonts/truetype/dejavu/`

### Network not growing

**Solution:**
- Verify file permissions on network PNG
- Check disk space
- Ensure `append_interaction()` is called after queries

---

## 🚀 Getting Started Checklist

- [ ] Install LM Studio
- [ ] Load a model in LM Studio
- [ ] Start LM Studio server (localhost:1234)
- [ ] Install Python dependencies: `pip install numpy pillow requests`
- [ ] Run demo: `python3 pxvm/integration/lm_studio_bridge.py --demo`
- [ ] Try interactive mode: `python3 pxvm/integration/lm_studio_bridge.py --interactive`
- [ ] Watch your network grow! 🌱

---

## 📚 Examples Gallery

### Example 1: Code Assistant

```python
# Teach it about your codebase
bridge.append_interaction(
    "What's the API structure?",
    "The API uses REST endpoints at /api/v1/..."
)

# Future queries will know your API structure!
```

### Example 2: Research Assistant

```python
# Accumulate research notes
for paper in papers:
    bridge.append_interaction(
        f"Summarize {paper.title}",
        paper.summary
    )

# Network becomes a research knowledge base
```

### Example 3: Learning Companion

```python
# Each study session builds on the last
bridge.conversational_loop()

# "What did we discuss last time?"
# LLM can reference previous sessions!
```

---

## 🎉 Start Your Learning Loop!

```bash
# Clone and run
cd pxos/pxvm
python3 integration/lm_studio_bridge.py --interactive

# Watch the magic happen
🧑 You: What is pxOS?
🤖 LLM: [Answer]
💾 Appending to pixel network...
   ✅ Network expanded: 150 → 175 rows (+25)

🧑 You: How does it work?
🤖 LLM: [Answer with context from previous Q&A!]
💾 Appending to pixel network...
   ✅ Network expanded: 175 → 205 rows (+30)

💡 Network has learned from 2 conversations!
```

---

## 📖 Documentation

- **Integration Guide**: `pxvm/integration/lm_studio_bridge.py` (docstrings)
- **API Reference**: See class `LMStudioPixelBridge`
- **Examples**: `pxvm/examples/quick_start.py`

---

## 🤝 Contributing

Ideas for contributions:
- Semantic search implementation
- Multi-network orchestration
- Knowledge export/import utilities
- OCR for extracting text from pixels
- Visualization tools (show network growth)
- Alternative LLM backends (Ollama, etc.)

---

## 📄 License

MIT License - See main LICENSE file

---

## ✨ Credits

**pxVM** - Where pixels are computational primitives and knowledge grows with every conversation.

*"The network that learns from itself."*

---

**Ready to start?**

```bash
python3 pxvm/integration/lm_studio_bridge.py --interactive
```

🚀 **Let your network grow!**
