# pxOS Heterogeneous Computing System

**A primitive-based system for unified CPU+GPU programming with LLM intelligence**

---

## 🎯 What Is This?

This is a breakthrough system that combines:

1. **Primitive-based programming** (like pxOS) - Simple, clear syntax
2. **GPU code generation** - Automatic CUDA compilation
3. **LLM intelligence** - Smart decisions about CPU vs GPU execution

**The Vision:** Write code once in primitives, let LLM decide where it runs (CPU or GPU), automatically generate optimized code for both targets.

---

## 🚀 Quick Start

### Example: Vector Addition

**Write primitives (example_vector_add.px):**
```
GPU_KERNEL vector_add
GPU_PARAM a float[]
GPU_PARAM b float[]
GPU_PARAM c float[]
GPU_PARAM n int

GPU_THREAD_CODE:
    THREAD_ID → tid
    COMPARE tid < n → in_bounds

    IF in_bounds:
        LOAD a[tid] → val_a
        LOAD b[tid] → val_b
        ADD val_a val_b → sum
        STORE sum → c[tid]
    ENDIF
GPU_END

GPU_LAUNCH vector_add BLOCKS 256 THREADS 256
```

**Build:**
```bash
python3 build_gpu.py example_vector_add.px
```

**Result:**
- Generates complete CUDA C code (example_vector_add.cu)
- Includes memory management
- Includes error checking
- Includes performance timing
- Compiles to executable (if nvcc available)

---

## 📚 Components

### 1. GPU Primitive Specification

**File:** `GPU_PRIMITIVE_SPEC.md`

Defines the syntax for GPU programming:
- `GPU_KERNEL` - Define kernels
- `GPU_PARAM` - Parameters
- `THREAD_ID` - Get thread index
- `LOAD`/`STORE` - Memory operations
- `ADD`/`SUB`/`MUL`/`DIV` - Arithmetic
- `IF`/`ELSE`/`ENDIF` - Control flow
- `GPU_LAUNCH` - Execute kernel

**Key Innovation:** Arrow notation `→` makes data flow explicit:
```
LOAD array[i] → value    # Clear: data flows from array INTO value
STORE result → array[i]  # Clear: result flows INTO array
```

### 2. GPU Primitive Parser

**File:** `gpu_primitives.py`

Parses GPU primitive syntax into structured data:
```python
parser = GPUPrimitiveParser()
parser.parse_file("example.px")

for kernel in parser.kernels.values():
    print(kernel.to_cuda_code())
```

**Features:**
- Full syntax validation
- Type checking
- Error messages with line numbers
- Converts to internal representation

### 3. CUDA Code Generator

**File:** `cuda_generator.py`, `build_gpu.py`

Generates complete, working CUDA programs:
```python
generator = CompleteCUDAGenerator(parser)
generator.generate_complete_example("output.cu")
```

**Generated code includes:**
- ✅ Kernel definitions
- ✅ Host code (main function)
- ✅ Memory allocation (host + device)
- ✅ Data transfers
- ✅ Kernel launch
- ✅ Error checking
- ✅ Performance timing
- ✅ Result verification

### 4. LLM Analyzer

**File:** `llm_analyzer.py`

Uses AI to analyze primitives and make intelligent decisions:

```python
analyzer = LLMAnalyzer(provider="openai")  # or "anthropic" or "local"

result = analyzer.analyze_primitive(code, context={
    "data_size": 1000000,
    "hardware": "RTX 3080"
})

print(f"Target: {result.target}")           # CPU or GPU
print(f"Reasoning: {result.reasoning}")     # Why?
print(f"Speedup: {result.estimated_speedup}x")  # Expected performance gain
```

**LLM considers:**
- Is operation data-parallel?
- Dataset size (GPU overhead vs benefit)
- I/O requirements (GPUs can't do I/O)
- Branching complexity (GPUs hate branches)
- Memory transfer costs

**Supported providers:**
- **OpenAI** (GPT-4) - Set `OPENAI_API_KEY`
- **Anthropic** (Claude) - Set `ANTHROPIC_API_KEY`
- **Local rules** - Built-in heuristics (no API needed)

---

## 🧠 How LLM Intelligence Works

### Traditional Approach

```c
// Programmer decides manually
if (size > 10000) {
    run_on_gpu();  // Hope this is faster!
} else {
    run_on_cpu();
}
```

### pxOS Heterogeneous Approach

```python
# Write code once
PARALLEL_SORT array SIZE ???

# LLM analyzes:
# - Dataset size: 1M elements → Large, GPU worth it
# - Operation type: Sort → Parallelizable
# - Hardware: RTX 3080 → Fast GPU available
# Decision: GPU with bitonic sort strategy
# Generates optimized GPU kernel automatically
```

### Example Analysis

```
Input Primitive:
  PARALLEL_SORT numbers SIZE 1000000

LLM Analysis:
  TARGET: GPU
  REASONING: Large dataset (1M elements) with inherently parallel
             sorting algorithm (bitonic sort). GPU will be ~10-50x
             faster than CPU for this workload.
  SPEEDUP: 25x estimated
  WARNINGS: None
  OPTIMIZATIONS:
    - Use bitonic sort for best GPU performance
    - Ensure data is page-locked for faster transfer
    - Consider streaming for even larger datasets
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│ High-Level Primitives (.px file)           │
│   PARALLEL_SORT array                       │
│   RENDER_GRAPHICS scene                     │
│   PROCESS_IMAGE pixels                      │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ LLM Analyzer (llm_analyzer.py)             │
│  • Analyzes intent                          │
│  • Decides: CPU, GPU, or hybrid?           │
│  • Suggests optimizations                   │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌──────────────────┐ ┌──────────────────┐
│ CPU Primitives   │ │ GPU Primitives   │
│ (pxOS style)     │ │ (GPU_KERNEL)     │
└────────┬─────────┘ └────────┬─────────┘
         │                    │
         ▼                    ▼
┌──────────────────┐ ┌──────────────────┐
│ x86 Machine Code │ │ CUDA C Code      │
│ (pxos.bin)       │ │ (.cu file)       │
└────────┬─────────┘ └────────┬─────────┘
         │                    │
         └──────────┬─────────┘
                    ▼
         ┌────────────────────┐
         │ Unified Executable │
         │ (runs on both!)    │
         └────────────────────┘
```

---

## 💡 Why This Is Revolutionary

### 1. **Unified Source**

One codebase generates both CPU and GPU code:

```
# Before (traditional):
main.c       # CPU code (C)
kernel.cu    # GPU code (CUDA)
makefile     # Build both

# After (pxOS Heterogeneous):
program.px   # Single source
# System generates both automatically
```

### 2. **LLM-Driven Optimization**

Computer decides better than humans:

```
Human: "I think this should run on GPU..."
LLM:   "Actually, dataset too small. GPU overhead = 2ms, computation = 0.1ms.
        CPU will be 20x faster for this workload."
```

### 3. **Educational Transparency**

See exactly what happens:

```
Primitive:  ADD val_a val_b → sum

CPU code:   mov eax, [val_a]
            add eax, [val_b]
            mov [sum], eax

GPU code:   auto sum = val_a + val_b;
```

Everything is explicit, nothing hidden.

### 4. **Builds on Proven Concepts**

- **Primitives:** pxOS shows this works
- **LLM code gen:** GitHub Copilot shows this works
- **Heterogeneous computing:** Industry standard (CUDA, OpenCL)

**Novel combination:** All three together!

---

## 📖 Usage Examples

### Example 1: Decide Automatically

```python
from llm_analyzer import LLMAnalyzer

analyzer = LLMAnalyzer(provider="openai")

code = "PARALLEL_FOR i IN range(1000000): output[i] = sin(input[i])"

result = analyzer.analyze_primitive(code, {"data_size": 1000000})

if result.target == ExecutionTarget.GPU:
    print("Running on GPU!")
    # Generate GPU kernel...
else:
    print("Running on CPU!")
    # Generate CPU code...
```

### Example 2: Manual GPU Code

```
GPU_KERNEL saxpy
GPU_PARAM x float[]
GPU_PARAM y float[]
GPU_PARAM a float
GPU_PARAM n int

GPU_THREAD_CODE:
    THREAD_ID → i
    IF i < n:
        LOAD x[i] → x_val
        LOAD y[i] → y_val
        MUL a x_val → ax
        ADD ax y_val → result
        STORE result → y[i]
    ENDIF
GPU_END
```

Build: `python3 build_gpu.py saxpy.px`

---

## 🛠️ Installation

### Prerequisites

```bash
# Required
sudo apt install python3 python3-pip

# Optional (for CUDA compilation)
sudo apt install nvidia-cuda-toolkit

# Optional (for LLM features)
pip3 install openai anthropic
```

### Setup

```bash
cd pxos-heterogeneous

# Test GPU primitive parser
python3 gpu_primitives.py

# Test CUDA generation
python3 build_gpu.py example_vector_add.px

# Test LLM analyzer
python3 llm_analyzer.py
```

---

## 📋 File Reference

| File | Purpose |
|------|---------|
| `GPU_PRIMITIVE_SPEC.md` | Complete syntax specification |
| `gpu_primitives.py` | Parser for GPU primitives |
| `cuda_generator.py` | Generates CUDA C code |
| `build_gpu.py` | Complete build system |
| `llm_analyzer.py` | LLM intelligence layer |
| `example_vector_add.px` | Sample program |
| `README.md` | This file |

---

## 🔬 Testing

Run all tests:

```bash
# Parser test
python3 gpu_primitives.py

# Generator test
python3 cuda_generator.py

# LLM analyzer test
python3 llm_analyzer.py

# Complete build test
python3 build_gpu.py example_vector_add.px
```

---

## 🚀 Future Roadmap

### Phase 1: Core System (✅ Complete)
- [x] GPU primitive syntax
- [x] Parser
- [x] CUDA code generator
- [x] LLM analyzer

### Phase 2: Integration (Next)
- [ ] Unified build system (CPU + GPU in one file)
- [ ] Link with pxOS CPU primitives
- [ ] Runtime coordinator

### Phase 3: Advanced Features
- [ ] Multiple GPU support
- [ ] Heterogeneous pipelines (split work)
- [ ] Dynamic optimization (profile-guided)
- [ ] OpenCL backend (multi-vendor GPUs)

### Phase 4: Language Features
- [ ] Higher-level abstractions
- [ ] Automatic parallelization
- [ ] Memory optimization hints
- [ ] Cross-compilation (ARM, x86, GPU)

---

## 🤝 Contributing

This is a research project exploring:
- Primitive-based programming systems
- LLM-assisted code generation
- Heterogeneous computing abstractions

Ideas for contributions:
- Add more GPU operations (reduce, scan, etc.)
- Improve LLM prompts
- Add OpenCL backend
- Create more examples
- Optimize generated code

---

## 📄 License

MIT License - See pxOS project for details

---

## 🎓 Academic Context

This system demonstrates:

1. **Separation of intent from implementation**
   - Primitives express WHAT, not HOW
   - LLM decides WHERE (CPU vs GPU)
   - Generators handle HOW (code generation)

2. **AI-assisted systems programming**
   - First system to use LLM for heterogeneous computing decisions
   - Shows LLMs can reason about hardware architectures
   - Demonstrates practical AI in low-level programming

3. **Educational tool**
   - Makes GPU programming accessible
   - Shows explicit mapping to CUDA
   - Teaches heterogeneous computing concepts

---

## 📞 Questions?

This is an experimental system. Core concepts:

- **Primitives**: Simple, explicit commands
- **LLM**: Intelligent decision making
- **Heterogeneous**: Best of CPU and GPU

**The goal:** Make parallel programming accessible while maintaining performance.

---

**Built on pxOS foundation**
**Powered by LLM intelligence**
**Targeting real hardware**

*"Write once, run optimally - let AI figure out where."*
