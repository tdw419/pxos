# 🚀 Phase 0: Stabilization - Quick Start

**Goal**: Make the Pixel-LLM foundation solid, testable, and safe to iterate on
**Timeline**: 2 days
**Cost**: ~$0.02 (Gemini reviews)
**Lines**: ~1,250 auto-generated

---

## What is Phase 0?

Before we build GPU kernels and convert 7B models, we need **rock-solid foundations**:

1. **Tests** - Prevent regressions as we iterate
2. **Agent Hardening** - Clear error messages, graceful fallbacks
3. **Configuration** - Explicit, version-controlled settings

**Philosophy**: "Make it right, then make it fast."

---

## 📋 Phase 0 Tasks (6 total)

### Critical Path (Priority 10/10)
1. **Unit tests for PixelFS** (200 lines)
   - Round-trip write/read validation
   - Header integrity checks
   - Checksum verification
   - Edge case handling

2. **Unit tests for InfiniteMap** (200 lines)
   - Spatial operations validation
   - Tile caching verification
   - Quadtree indexing tests
   - Persistence validation

### Important (Priority 8-9/10)
3. **Unit tests for Task Queue** (150 lines)
4. **Agent capability detection** (150 lines)
5. **Configuration system** (200 lines)
6. **Test runner and CI** (100 lines)

**Total**: ~1,000 lines of tests + infrastructure

---

## 🎯 Execute Phase 0 (3 Commands)

### 1. Initialize Phase 0 Tasks
```bash
cd /home/user/pxos
python3 pixel_llm_coach.py init --phase 0_stabilization
```

**Expected output**:
```
======================================================================
🎯 INITIALIZING PHASE: Stabilization & Testing
   Make what exists solid, testable, and safe to iterate on
======================================================================

✅ Added: Unit tests for PixelFS
   Priority: 10/10
   Path: pixel_llm/tests/test_pixelfs.py

✅ Added: Unit tests for InfiniteMap
   Priority: 10/10
   Path: pixel_llm/tests/test_infinite_map.py

[... 4 more tasks ...]

✓ Phase 0_stabilization initialized with 6 tasks
```

### 2. Auto-Generate Tests
```bash
python3 pixel_llm_coach.py coach --phase 0_stabilization --max-tasks 6
```

**What happens**:
```
🚀 PIXEL-LLM COACHING LOOP
🤖 Local LLM: ollama
✨ Gemini: ✅ Available

======================================================================
🎓 COACHING: Unit tests for PixelFS
   Phase: 0_stabilization
   File: pixel_llm/tests/test_pixelfs.py
   Priority: 10/10
======================================================================

--- Iteration 1/3 ---
🤖 Local LLM generating code...
✓ Generated 2,850 characters

🔍 Gemini reviewing code...
📊 Score: 7/10
💬 Feedback: Good coverage! Add edge cases for corrupted headers...

--- Iteration 2/3 ---
🤖 Local LLM generating code...
✓ Generated 3,420 characters

🔍 Gemini reviewing code...
📊 Score: 9/10
✅ ACCEPTED - High quality implementation!
💾 Saved: pixel_llm/tests/test_pixelfs.py

[... continues for all 6 tasks ...]

======================================================================
✅ Coaching loop complete!
   Processed: 6 tasks
   Completed: 6 tasks
   Success rate: 100%
======================================================================
```

**Timeline**: 15-30 minutes (depending on LLM speed)
**Cost**: ~$0.009 (6 tasks × ~$0.0015)

### 3. Run Tests
```bash
# Install pytest if needed
pip3 install pytest pytest-cov

# Run all tests
python3 -m pytest pixel_llm/tests -v --cov=pixel_llm

# Or use the generated test runner
chmod +x pixel_llm/tests/run_tests.sh
./pixel_llm/tests/run_tests.sh
```

**Expected output**:
```
============================= test session starts ==============================
platform linux -- Python 3.11.0, pytest-7.4.3, pluggy-1.3.0
collected 25 items

pixel_llm/tests/test_pixelfs.py ..........                              [ 40%]
pixel_llm/tests/test_infinite_map.py ..........                         [ 80%]
pixel_llm/tests/test_task_queue.py .....                                [100%]

---------- coverage: platform linux, python 3.11.0 -----------
Name                                Stmts   Miss  Cover
-------------------------------------------------------
pixel_llm/core/pixelfs.py            250     15    94%
pixel_llm/core/infinite_map.py       280     18    94%
pixel_llm/core/task_queue.py         180     10    94%
-------------------------------------------------------
TOTAL                                710     43    94%

========================= 25 passed in 3.52s ===============================
```

---

## ✅ Success Criteria

### After Phase 0, you should have:

1. **Test Coverage ≥80%**
   - PixelFS: All core functions tested
   - InfiniteMap: Spatial operations validated
   - Task Queue: Lifecycle verified

2. **Agent Detection**
   - `GeminiAgent.is_available()` → True/False
   - `LocalLLMAgent.is_available()` → True/False
   - Clear error messages when missing

3. **Configuration System**
   - `pixel_llm/config.example.json` exists
   - `pixel_llm/core/config.py` validates settings
   - Environment variables supported

4. **Test Automation**
   - Single command runs all tests
   - Coverage reports generated
   - CI-ready structure

---

## 🔍 Verify Phase 0 Completion

```bash
# Check test coverage
python3 -m pytest pixel_llm/tests --cov=pixel_llm --cov-report=term-missing

# Verify agent detection
python3 pixel_llm_coach.py agents

# Check config system
ls -la pixel_llm/core/config.py
ls -la pixel_llm/config.example.json

# View all Phase 0 tasks
python3 pixel_llm_coach.py status
```

**Expected Phase 0 Status**:
```
✅ Complete Phase 0: Stabilization & Testing
    6/6 tasks (100%)
```

---

## 🚀 What's Next: Phase 1 (GPU Core)

Once Phase 0 is complete and tests are passing:

```bash
# Initialize Phase 1
python3 pixel_llm_coach.py init --phase 1_gpu_core

# Tasks in Phase 1:
# - WGSL dot product kernel
# - CPU vs GPU validation
# - Tiled matrix multiplication
# - Acceptance test harness
```

---

## 🐛 Troubleshooting

### "No local LLM available"
```bash
# Install ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5-coder:7b

# Verify
python3 pixel_llm_coach.py agents
```

### "Tests failing after generation"
```bash
# Review generated tests
cat pixel_llm/tests/test_pixelfs.py

# Run specific test
python3 -m pytest pixel_llm/tests/test_pixelfs.py::test_round_trip -v

# Re-coach if needed
python3 pixel_llm_coach.py coach --phase 0_stabilization --max-tasks 1
```

### "Gemini reviews too strict"
```bash
# Lower acceptance threshold in pixel_llm_coach.py
# Change: if score >= 8:
# To: if score >= 7:

# Or run without Gemini (unreviewed mode)
unset GEMINI_API_KEY
python3 pixel_llm_coach.py coach --phase 0_stabilization
```

---

## 📊 Phase 0 Metrics

### Estimated Resources
| Metric | Value |
|--------|-------|
| **Tasks** | 6 |
| **Lines Generated** | ~1,000 |
| **Gemini Reviews** | ~12-18 |
| **Cost** | $0.009-0.015 |
| **Time** | 15-30 min |
| **Success Rate** | 95%+ |

### Quality Expectations
| Component | Coverage | Tests |
|-----------|----------|-------|
| **PixelFS** | ≥85% | 8-10 |
| **InfiniteMap** | ≥85% | 8-10 |
| **Task Queue** | ≥80% | 5-7 |

---

## 💡 Pro Tips

### Speed Up Iteration
```bash
# Process tasks in parallel (if multiple LLMs available)
python3 pixel_llm_coach.py coach --phase 0_stabilization --parallel 3

# Skip Gemini reviews for rapid iteration
NO_GEMINI=1 python3 pixel_llm_coach.py coach --phase 0_stabilization

# Auto-fix failing tests
python3 pixel_llm_coach.py coach --fix-failures
```

### Monitor Progress
```bash
# Watch test coverage increase
watch -n 5 'python3 -m pytest pixel_llm/tests --cov=pixel_llm --cov-report=term | tail -20'

# Live coaching status
watch -n 3 'python3 pixel_llm_coach.py status'
```

### Save Coaching Logs
```bash
# Save detailed logs
python3 pixel_llm_coach.py coach --phase 0_stabilization 2>&1 | tee phase0_$(date +%Y%m%d).log
```

---

## 🎯 The Bottom Line

**Phase 0 transforms Pixel-LLM from "working prototype" to "production foundation":**

- **Before**: Hope tests pass, unclear what breaks
- **After**: 80%+ coverage, instant feedback, safe iteration

**This enables everything that follows:**
- GPU kernels (Phase 1)
- Model conversion (Phase 3)
- Self-improvement (Phase 5)

**Cost**: Trivial (~$0.01)
**Time**: 30 minutes
**Value**: Immeasurable (prevents days of debugging)

---

## 🚀 Ready to Stabilize?

```bash
# Three commands to production-ready foundation:

# 1. Initialize
python3 pixel_llm_coach.py init --phase 0_stabilization

# 2. Auto-generate
python3 pixel_llm_coach.py coach --phase 0_stabilization --max-tasks 6

# 3. Verify
python3 -m pytest pixel_llm/tests -v --cov
```

**Then proceed to Phase 1 (GPU Core) with confidence!** 🎨🤖✨
