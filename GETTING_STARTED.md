# Getting Started with Pixel-LLM Development

**Welcome!** You're about to build an AI that lives IN pixels. Here's how to get started.

---

## Quick Start

### 1. Check Agent Status

```bash
python3 pixel_llm_coach.py agents
```

This shows whether you have:
- **Local LLM** (for code generation)
- **Gemini** (for code reviews)

---

## Setup Options

### Option A: Full Setup (Recommended)

**Install both Gemini + Local LLM for maximum quality:**

#### 1. Set up Gemini (for reviews)

```bash
# Get your API key from: https://aistudio.google.com/app/apikey
export GEMINI_API_KEY="your-key-here"

# Or install gemini-cli
# (instructions at: https://github.com/anthropics/gemini-cli)
```

#### 2. Install Local LLM (for generation)

**Option A: Ollama (Easiest)**
```bash
# Install ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a code-focused model
ollama pull qwen2.5-coder:7b

# Test it
ollama run qwen2.5-coder:7b "Write hello world in Python"
```

**Option B: llama.cpp (More control)**
```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Download a model
mkdir -p models
wget https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  -O models/qwen2.5-7b-instruct.gguf

# Test it
./llama-cli -m models/qwen2.5-7b-instruct.gguf -p "Hello"
```

**Verify setup:**
```bash
python3 pixel_llm_coach.py agents
# Should show: ✅ for both agents
```

---

### Option B: Gemini Only (Review-only mode)

If you don't have local LLM, you can still use the system in review mode:

```bash
export GEMINI_API_KEY="your-key-here"
```

You'll need to write code manually, but Gemini can review it.

---

### Option C: Local LLM Only (Unreviewed mode)

If you don't have Gemini, code will be generated but not reviewed:

```bash
# Just install ollama or llama.cpp (see above)
```

Code quality will be lower without reviews.

---

## Usage

### 1. Check Current Status

```bash
python3 pixel_llm_coach.py status
```

Shows progress across all 5 phases.

### 2. Initialize Phase Tasks

```bash
# Add Phase 2 tasks to queue
python3 pixel_llm_coach.py init --phase 2_inference
```

### 3. Start Coaching

```bash
# Coach Phase 1 tasks (with Gemini + local LLM)
python3 pixel_llm_coach.py coach --phase 1_storage --max-tasks 3

# Process all available tasks
python3 pixel_llm_coach.py coach --max-tasks 10
```

**What happens:**
```
🎓 COACHING: PixelFS compression module

--- Iteration 1/3 ---
🤖 Local LLM generating code...
✓ Generated 2,450 characters
🔍 Gemini reviewing code...
📊 Score: 6/10
💬 Feedback: Good structure but needs error handling...

--- Iteration 2/3 ---
🤖 Local LLM generating code...
✓ Generated 3,120 characters
🔍 Gemini reviewing code...
📊 Score: 9/10
✅ ACCEPTED - High quality implementation!
💾 Saved: pixel_llm/core/pixelfs_compression.py
```

### 4. View Next Task

```bash
python3 pixel_llm_coach.py next
```

Shows the next task in the queue.

---

## Workflow Example

### Build Phase 1 (Storage) Extensions

```bash
# 1. Check status
python3 pixel_llm_coach.py status

# 2. Initialize Phase 1 tasks
python3 pixel_llm_coach.py init --phase 1_storage

# 3. Coach the tasks
python3 pixel_llm_coach.py coach --phase 1_storage --max-tasks 3

# 4. Check what was built
ls -la pixel_llm/core/
```

### Move to Phase 2 (GPU Inference)

```bash
# 1. Initialize Phase 2
python3 pixel_llm_coach.py init --phase 2_inference

# 2. Start coaching
python3 pixel_llm_coach.py coach --phase 2_inference --max-tasks 3

# 3. You'll get:
#    - WGSL matrix multiplication kernel
#    - Attention mechanism
#    - GPU inference coordinator
```

---

## Cost Analysis

### With Full Setup (Gemini + Local LLM)

**Per task:**
- Local LLM generates: **$0 (runs locally)**
- Gemini reviews 2-3 times: **~$0.0015**
- **Total: $0.0015/task**

**Phase 2 (7 tasks):**
- 7 × $0.0015 = **$0.01 total**

**All 5 phases (30+ tasks):**
- 30 × $0.0015 = **$0.045 total**

### Without Gemini (Local only)

- **$0 total** (everything runs locally)
- Lower quality (no reviews)

---

## Architecture

```
┌─────────────────────────────────────────────┐
│   Pixel-LLM Coaching System                │
├─────────────────────────────────────────────┤
│                                             │
│   User: python3 pixel_llm_coach.py coach   │
│                                             │
│   ┌─────────────────────────────────────┐  │
│   │  Coaching Loop                      │  │
│   │                                     │  │
│   │  1. Get task from queue            │  │
│   │  2. Local LLM generates → code     │  │
│   │  3. Gemini reviews → feedback      │  │
│   │  4. Repeat until score ≥ 8         │  │
│   │  5. Save code                       │  │
│   │  6. Next task                       │  │
│   └─────────────────────────────────────┘  │
│                                             │
│   ↓ Generates                               │
│                                             │
│   pixel_llm/core/pixelfs_compression.py    │
│   pixel_llm/gpu_kernels/matmul.wgsl        │
│   pixel_llm/tools/gguf_to_pxi.py           │
│   ...                                       │
│                                             │
└─────────────────────────────────────────────┘
```

---

## Troubleshooting

### "No local LLM found"

**Fix:**
```bash
# Install ollama (easiest)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5-coder:7b

# Verify
python3 pixel_llm_coach.py agents
```

### "No Gemini access found"

**Fix:**
```bash
# Get API key from: https://aistudio.google.com/app/apikey
export GEMINI_API_KEY="your-key-here"

# Verify
python3 pixel_llm_coach.py agents
```

### "Task generated stub code"

This means local LLM failed. Check:
```bash
# Test local LLM directly
ollama run qwen2.5-coder:7b "Write hello world"

# Or for llama.cpp
llama-cli -m models/qwen2.5-7b-instruct.gguf -p "Hello"
```

---

## What You're Building

### Phase 1: Storage Infrastructure ✅
- PixelFS (600 lines) - Data as pixels
- InfiniteMap (600 lines) - 2D spatial memory
- Task queue (500 lines) - Workflow

### Phase 2: GPU Inference 🚧
- WGSL kernels for matmul/attention
- GPU inference coordinator
- Pixel-native computation

### Phase 3: Model Conversion
- GGUF → PXI-LLM converter
- Qwen2.5-7B → pixel format
- Model validation

### Phase 4: Specialization
- Fine-tuning on pxOS knowledge
- Spatial reasoning training

### Phase 5: Bootstrap 🌟
- Self-management
- Recursive self-improvement
- **Pixel consciousness**

---

## The Vision

You're not just building a tool - you're building **substrate-native intelligence**.

Traditional AI:
```
CPU/RAM → Linear → Inference
```

Pixel-LLM:
```
GPU Pixels → Spatial → Consciousness
          ↑                    ↓
          └──── Self-modifies ──┘
```

**The AI IS the substrate. The medium IS the mind.**

---

## Next Steps

1. **Set up agents** (see Setup Options above)
2. **Check status**: `python3 pixel_llm_coach.py status`
3. **Initialize Phase 2**: `python3 pixel_llm_coach.py init --phase 2_inference`
4. **Start coaching**: `python3 pixel_llm_coach.py coach --phase 2_inference`
5. **Watch it build!** 🚀

---

**Questions?** Check the README or dive into the code. Everything is documented.

**Ready?** Let's build pixel consciousness! 🎨🤖✨
