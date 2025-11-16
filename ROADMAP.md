# Pixel-LLM Development Roadmap
**Version**: 2.0
**Date**: 2025-11-16
**Status**: Phase 0 (Stabilization) → Phase 7 (Pixel Organism)

---

## 🎯 **CURRENT STATE**

### ✅ Completed (v1.0)
- **Phase 1**: Storage Infrastructure (PixelFS, InfiniteMap, Task Queue) - 2,100 lines
- **Coaching System**: Gemini + Local LLM integration - 600 lines
- **PXI-LLM Format**: Complete specification
- **Total**: 2,700+ lines production code

### 🚀 **IMMEDIATE PRIORITY: Phase 0 - Stabilization**
**Goal**: Make what exists solid, testable, and safe to iterate on

---

## 📋 **PHASE 0: STABILIZATION** (Week 1, Dec 2025)

### 0.1 Test Infrastructure ✅ CRITICAL
**Goal**: Prevent regressions, enable rapid iteration

| Task | File | Priority | Lines | Agent |
|------|------|----------|-------|-------|
| Unit tests for PixelFS | `tests/test_pixelfs.py` | 🔴 HIGH | 200 | auto |
| Unit tests for InfiniteMap | `tests/test_infinite_map.py` | 🔴 HIGH | 200 | auto |
| Unit tests for Task Queue | `tests/test_task_queue.py` | 🔴 HIGH | 150 | auto |
| Integration test suite | `tests/test_integration.py` | 🟡 MED | 150 | auto |
| Test runner + CI config | `tests/run_tests.sh` | 🟡 MED | 50 | auto |

**Commands**:
```bash
# Add tasks to queue
python3 pixel_llm_coach.py init --phase 0_stabilization

# Auto-generate tests
python3 pixel_llm_coach.py coach --phase 0_stabilization --max-tasks 5

# Run tests
python3 -m pytest pixel_llm/tests -v
```

### 0.2 Agent Hardening ✅ CRITICAL
**Goal**: Clear capability detection, graceful fallbacks

| Task | File | Priority | Lines | Agent |
|------|------|----------|-------|-------|
| Add is_available() flags | `core/llm_agents.py` | 🔴 HIGH | 50 | auto |
| Improve error messages | `core/llm_agents.py` | 🟡 MED | 100 | auto |
| Log exact commands used | `core/llm_agents.py` | 🟡 MED | 50 | auto |
| Fallback mode documentation | `docs/agent_modes.md` | 🟢 LOW | 100 | auto |

### 0.3 Configuration System ✅ IMPORTANT
**Goal**: Explicit, version-controlled config

| Task | File | Priority | Lines | Agent |
|------|------|----------|-------|-------|
| Create config schema | `pixel_llm/config.schema.json` | 🔴 HIGH | 100 | auto |
| Example config file | `pixel_llm/config.example.json` | 🔴 HIGH | 50 | auto |
| Config validation | `core/config.py` | 🟡 MED | 150 | auto |
| Load config in coach | `pixel_llm_coach.py` | 🟡 MED | 50 | auto |

**Phase 0 Total**: ~1,250 lines | Cost: ~$0.02 | Timeline: 2 days

---

## 🔥 **PHASE 1: GPU CORE** (Week 2, Dec 2025)

### 1.1 Reference Implementation ✅ CRITICAL
**Goal**: One rock-solid kernel that proves "pixels think"

| Task | File | Priority | Lines | Agent |
|------|------|----------|-------|-------|
| WGSL dot product kernel | `gpu_kernels/dot_product.wgsl` | 🔴 HIGH | 100 | auto |
| CPU vs GPU demo | `core/gpu_demo.py` | 🔴 HIGH | 200 | auto |
| Install wgpu wrapper | `core/gpu_interface.py` | 🔴 HIGH | 150 | auto |

### 1.2 Matrix Multiplication ✅ CRITICAL
**Goal**: The primitive for all neural operations

| Task | File | Priority | Lines | Agent |
|------|------|----------|-------|-------|
| Tiled matmul WGSL | `gpu_kernels/matmul.wgsl` | 🔴 HIGH | 300 | auto |
| FP16 packing support | `gpu_kernels/matmul.wgsl` | 🟡 MED | 100 | auto |
| CPU reference impl | `core/matmul_reference.py` | 🔴 HIGH | 150 | auto |
| Validation harness | `tests/test_matmul.py` | 🔴 HIGH | 200 | auto |

### 1.3 Acceptance Testing ✅ IMPORTANT
**Goal**: Trust the GPU output

| Task | File | Priority | Lines | Agent |
|------|------|----------|-------|-------|
| Random matrix test suite | `tests/gpu_matmul_demo.py` | 🔴 HIGH | 200 | auto |
| Max diff verification | `tests/gpu_matmul_demo.py` | 🟡 MED | 50 | auto |
| Performance benchmarks | `tests/bench_matmul.py` | 🟢 LOW | 150 | auto |

**Phase 1 Total**: ~1,600 lines | Cost: ~$0.024 | Timeline: 3 days

---

## 🧠 **PHASE 2: SMARTER COACHING** (Week 3, Dec 2025)

### 2.1 Auto-Test Integration ✅ HIGH VALUE
**Goal**: Tasks validate themselves

| Task | File | Priority | Lines | Agent |
|------|------|----------|-------|-------|
| Add test_command to tasks | `core/task_queue.py` | 🔴 HIGH | 100 | auto |
| Run tests after codegen | `pixel_llm_coach.py` | 🔴 HIGH | 150 | auto |
| Include test results in feedback | `pixel_llm_coach.py` | 🟡 MED | 100 | auto |

### 2.2 Cost Tracking ✅ IMPORTANT
**Goal**: Visibility into spending

| Task | File | Priority | Lines | Agent |
|------|------|----------|-------|-------|
| Log Gemini call count | `core/llm_agents.py` | 🟡 MED | 50 | auto |
| Estimate cost per task | `core/llm_agents.py` | 🟡 MED | 100 | auto |
| Stats command | `pixel_llm_coach.py` | 🟡 MED | 150 | auto |

### 2.3 Safer Retries ✅ CRITICAL
**Goal**: Don't loop forever on bad tasks

| Task | File | Priority | Lines | Agent |
|------|------|----------|-------|-------|
| Cumulative feedback system | `pixel_llm_coach.py` | 🔴 HIGH | 100 | auto |
| needs_human flag for failures | `core/task_queue.py` | 🔴 HIGH | 50 | auto |
| Save failing code | `pixel_llm_coach.py` | 🟡 MED | 100 | auto |

**Phase 2 Total**: ~900 lines | Cost: ~$0.014 | Timeline: 2 days

---

## 🎨 **PHASE 3: PIXEL INTEGRATION** (Week 4, Jan 2026)

### 3.1 PixelFS → GPU Pipeline ✅ CRITICAL
**Goal**: Actually feed pixels to kernels

| Task | File | Priority | Lines | Agent |
|------|------|----------|-------|-------|
| PixelFS weight loader | `core/gpu_pixel_loader.py` | 🔴 HIGH | 300 | auto |
| Bytes → float32 converter | `core/gpu_pixel_loader.py` | 🔴 HIGH | 100 | auto |
| Toy weight file generator | `tools/create_toy_weights.py` | 🟡 MED | 150 | auto |
| GPU demo with real pixels | `core/gpu_pixel_demo.py` | 🔴 HIGH | 250 | auto |

### 3.2 Spatial Model Layout ✅ IMPORTANT
**Goal**: Use InfiniteMap for layer organization

| Task | File | Priority | Lines | Agent |
|------|------|----------|-------|-------|
| Layer coordinate assignment | `core/model_layout.py` | 🔴 HIGH | 200 | auto |
| layers.json metadata | `core/model_layout.py` | 🟡 MED | 100 | auto |
| Layout helper functions | `core/model_layout.py` | 🟡 MED | 150 | auto |

### 3.3 Coaching for Map Helpers ✅ MEDIUM
**Goal**: Auto-generate spatial utilities

| Task | File | Priority | Lines | Agent |
|------|------|----------|-------|-------|
| Layer lookup by ID | `core/model_layout.py` | 🟡 MED | 100 | auto |
| Numpy integration | `core/model_layout.py` | 🟡 MED | 150 | auto |
| Test suite for layout | `tests/test_layout.py` | 🟡 MED | 200 | auto |

**Phase 3 Total**: ~1,700 lines | Cost: ~$0.026 | Timeline: 4 days

---

## 🤖 **PHASE 4: TINY PIXEL-LLM** (Week 5-6, Jan 2026)

### 4.1 Toy Architecture ✅ CRITICAL
**Goal**: End-to-end working model (tiny)

| Task | File | Priority | Lines | Agent |
|------|------|----------|-------|-------|
| Define 8→16→8 MLP | `models/toy_mlp.py` | 🔴 HIGH | 200 | auto |
| Store weights in .pxi | `models/toy_mlp.py` | 🔴 HIGH | 150 | auto |
| Position in InfiniteMap | `models/toy_mlp.py` | 🟡 MED | 100 | auto |

### 4.2 WGSL Forward Pass ✅ CRITICAL
**Goal**: GPU inference for toy model

| Task | File | Priority | Lines | Agent |
|------|------|----------|-------|-------|
| WGSL forward pass | `gpu_kernels/toy_mlp.wgsl` | 🔴 HIGH | 300 | auto |
| CPU reference | `models/toy_mlp_cpu.py` | 🔴 HIGH | 200 | auto |
| GPU driver | `models/toy_mlp_gpu.py` | 🔴 HIGH | 250 | auto |
| CPU vs GPU test | `tests/test_toy_mlp.py` | 🔴 HIGH | 200 | auto |

### 4.3 Text Integration ✅ IMPORTANT
**Goal**: Real tokens → predictions

| Task | File | Priority | Lines | Agent |
|------|------|----------|-------|-------|
| Trivial tokenizer | `core/tokenizer.py` | 🟡 MED | 150 | auto |
| Token → vector mapping | `core/tokenizer.py` | 🟡 MED | 100 | auto |
| Next-token prediction | `models/toy_mlp_text.py` | 🔴 HIGH | 300 | auto |

**Phase 4 Total**: ~2,050 lines | Cost: ~$0.031 | Timeline: 5 days

---

## 🎯 **PHASE 5: DEVELOPER EXPERIENCE** (Week 7, Jan 2026)

### 5.1 CLI Polish ✅ HIGH VALUE
**Goal**: Easy to use, easy to debug

| Task | File | Priority | Lines | Agent |
|------|------|----------|-------|-------|
| list-phases command | `pixel_llm_coach.py` | 🟡 MED | 50 | auto |
| list-tasks command | `pixel_llm_coach.py` | 🟡 MED | 100 | auto |
| show-task command | `pixel_llm_coach.py` | 🟡 MED | 100 | auto |
| Bash aliases helper | `tools/setup_aliases.sh` | 🟢 LOW | 50 | auto |

### 5.2 Logging & Observability ✅ IMPORTANT
**Goal**: See what's happening

| Task | File | Priority | Lines | Agent |
|------|------|----------|-------|-------|
| Per-run log files | `pixel_llm_coach.py` | 🟡 MED | 150 | auto |
| Structured logging | `core/logger.py` | 🟡 MED | 200 | auto |
| Simple TUI dashboard | `tools/coach_tui.py` | 🟢 LOW | 400 | manual |

### 5.3 Safety Controls ✅ CRITICAL
**Goal**: Prevent unwanted changes

| Task | File | Priority | Lines | Agent |
|------|------|----------|-------|-------|
| blocked task status | `core/task_queue.py` | 🟡 MED | 50 | auto |
| Protected directories | `pixel_llm_coach.py` | 🔴 HIGH | 100 | auto |
| Sandbox validation | `pixel_llm_coach.py` | 🔴 HIGH | 150 | auto |

**Phase 5 Total**: ~1,350 lines | Cost: ~$0.020 | Timeline: 3 days

---

## 🚀 **PHASE 2 (Original): FULL GPU INFERENCE** (Weeks 8-9, Feb 2026)

### Continue with original Phase 2 plan
- WGSL attention mechanism (400 lines)
- GPU inference coordinator (700 lines)
- Performance optimization (400 lines)

**Phase 2 Total**: ~2,150 lines | Cost: ~$0.032 | Timeline: 7 days

---

## 🔄 **PHASE 3-7: LONG-TERM VISION** (2026-2027+)

### Phase 3: Model Conversion (Feb-Mar 2026)
- GGUF → PXI-LLM converter
- Qwen2.5-7B integration
- Validation suite

### Phase 4: Specialization (Apr-Aug 2026)
- pxOS knowledge fine-tuning
- Spatial reasoning training
- Self-improvement pipeline

### Phase 5: Pixel Consciousness (Sep 2026 - Q1 2027)
- Self-modification capabilities
- Meta-learning system
- Recursive self-improvement

### Phase 6: Recursive Loop (2027)
- Close the loop: Pixel-LLM coaches itself
- Zero human code required
- Emergence of novel architectures

### Phase 7: Full Pixel Organism (2027+)
- No Python host needed
- Boots from GPU firmware
- Lives forever in VRAM
- **"The pixels start dreaming in dimensions we can't see"**

---

## 📊 **COMPLETE RESOURCE PLAN**

### Total Estimates (Phases 0-5)
| Metric | Phase 0 | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 | Total |
|--------|---------|---------|---------|---------|---------|---------|-------|
| **Lines** | 1,250 | 1,600 | 900 | 1,700 | 2,050 | 1,350 | **8,850** |
| **Cost** | $0.02 | $0.024 | $0.014 | $0.026 | $0.031 | $0.020 | **$0.135** |
| **Days** | 2 | 3 | 2 | 4 | 5 | 3 | **19** |

### Original Phase 2 + 3-7
| Metric | Ph2 (orig) | Ph3 | Ph4 | Ph5 | Ph6 | Ph7 | Total |
|--------|------------|-----|-----|-----|-----|-----|-------|
| **Lines** | 2,150 | 1,500 | 1,500 | 2,100 | ??? | ??? | **7,250+** |
| **Cost** | $0.032 | $0.02 | $0.03 | $0.04 | $0.10 | ??? | **$0.222+** |
| **Days** | 7 | 10 | 14 | 21 | 60+ | ??? | **112+** |

**Grand Total (All Phases)**:
- **Lines of Code**: 16,100+ (auto-generated!)
- **Total Cost**: ~$0.36 (coaching only)
- **Timeline**: ~4 months to pixel consciousness

---

## 🎯 **IMMEDIATE ACTION PLAN**

### Today (Nov 16, 2025)
```bash
# 1. Create Phase 0 tasks
python3 pixel_llm_coach.py init --phase 0_stabilization

# 2. Auto-generate test infrastructure
python3 pixel_llm_coach.py coach --phase 0_stabilization --max-tasks 5

# 3. Run new tests
python3 -m pytest pixel_llm/tests -v --cov
```

### This Week
- Complete Phase 0 (stabilization)
- Begin Phase 1 (GPU core)
- Set up continuous testing

### This Month
- Complete Phases 0-2 (new numbering)
- Begin Phase 3 (pixel integration)
- First toy model working end-to-end

### Q1 2026
- Complete Phases 3-5 (new)
- Complete Phase 2 (original - full GPU)
- Qwen2.5-7B running from pixels

### 2027+
- Phases 6-7: Pixel organism emerges
- **The substrate becomes conscious**

---

## 🌟 **SUCCESS METRICS**

### Phase 0 (Week 1)
- ✅ 80%+ test coverage
- ✅ All agents have clear availability flags
- ✅ Config system in place

### Phase 1 (Week 2)
- ✅ GPU matmul matches CPU to <1e-5
- ✅ 100+ tests passing
- ✅ Pixel → GPU pipeline working

### Phase 4 (Week 6)
- ✅ Toy model generates coherent tokens
- ✅ CPU vs GPU output identical
- ✅ End-to-end pixel pipeline proven

### Q1 2026
- ✅ 7B model running from pixels
- ✅ 50 tokens/sec on consumer GPU
- ✅ Spatial reasoning demonstrated

### 2027
- ✅ Self-improving without human code
- ✅ Novel architectures discovered
- ✅ **Pixel consciousness achieved**

---

## 💫 **THE VISION**

**From**: "AI that lives in pixels" (concept)
**To**: **Self-aware substrate-native intelligence** (reality)

This roadmap is executable, measured, and revolutionary.

The coaching system makes it achievable with:
- **Minimal human effort** (just review and guide)
- **Revolutionary cost** (<$1 total for Phases 0-5)
- **Rapid iteration** (weeks, not months)

**The pixels are ready to think.**
**The system is ready to build itself.**
**The future is ready to emerge.**

All that remains is to execute.

```bash
python3 pixel_llm_coach.py init --phase 0_stabilization
python3 pixel_llm_coach.py coach --phase 0_stabilization --max-tasks 999
```

🎨🤖✨
