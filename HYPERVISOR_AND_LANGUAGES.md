# Hypervisor & Language Architecture for pxOS

**Date:** November 2025
**Status:** Design + Initial Implementation

## The Core Question

> **Since pxOS runs on a pixel substrate where imperfect computing is allowed, what programming language should LLMs use to build the system? And how does that code actually execute?**

## The Answer: Layered Execution Model

```
┌─────────────────────────────────────────────────────────┐
│  Hardware (Real CPU/GPU)                                │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  Host OS (Linux/macOS/Windows)                          │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  CPython Runtime (bootstrap layer)                      │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  Pixel Hypervisor  ← ABSTRACTION BOUNDARY               │
│  ├─ Manages execution                                   │
│  ├─ Handles traps/syscalls                              │
│  ├─ Provides PixelBridge API                            │
│  └─ Enables imperfect computing                         │
└──────────────┬─────────────┬────────────────────────────┘
               │             │
       ┌───────▼──────┐  ┌──▼──────────────┐
       │  PXICPU      │  │  Guest Runtimes │
       │  (pixels)    │  │  (Python, etc.) │
       └──────────────┘  └─────────────────┘
```

## Language Choice: Use Python (For Now)

### Why Python?

1. **LLMs are excellent at Python**
   - Most training data includes Python
   - Natural for expressing algorithms
   - Easy to debug

2. **We have a Python → PXI compiler**
   - `python_to_pxi.py` already works
   - Can compile functions to pixel-native code
   - Gradual migration path

3. **Two execution modes:**
   - **Compiled**: Python → PXI → runs on PXICPU
   - **Guest**: Python stored in pixels, executed by hypervisor

4. **Future-proof:**
   - Can introduce "PXLang" (LLM-native language) later
   - Python becomes the meta-language/scaffolding
   - Smooth transition path

### What About a New Language?

Eventually, yes! A language **made by LLMs, for LLMs** that is:

- Structural/declarative (JSON-like or Lisp-like)
- Optimized for pixel operations
- Has LLM calls as first-class citizens
- Maps cleanly to PXI

But we don't need training data if:
- The language is simple and regular
- We provide clear spec + examples in prompts
- LLMs generate code from spec, not memory

**Roadmap:**
```
Phase 6:  Python → PXI (current)
Phase 7:  Python + Hypervisor (guest execution)
Phase 8:  PXLang spec + compiler
Phase 9:  LLMs writing PXLang exclusively
Phase 10: Python becomes bootstrap only
```

## The Hypervisor: How Code Actually Runs

### Two Execution Paths

#### Path 1: Compiled PXI Modules (Direct Execution)

```
LLM writes Python
       ↓
python_to_pxi.py (compiler)
       ↓
PXI module (.pxi.png)
       ↓
pack_file_to_boot_pixel.py
       ↓
Sub-boot pixel (file_id)
       ↓
PixelFS (/boot/*, /apps/*)
       ↓
boot_kernel.py loads via SYS_BLOB
       ↓
PXICPU executes pixel-native code
```

**Characteristics:**
- Fastest (no interpretation overhead)
- Statically compiled
- Limited by PXI opcode set
- Good for: kernels, drivers, core modules

#### Path 2: Guest Python (Hypervisor-Managed)

```
LLM writes Python
       ↓
Stored as sub-boot pixel
       ↓
PixelFS (/apps/my_module.py)
       ↓
Hypervisor loads via SYS_BLOB
       ↓
Hypervisor executes with PixelBridge API
       ↓
Guest reads/writes pixel memory
       ↓
Guest calls LLMs, loads files, samples noise
```

**Characteristics:**
- More flexible (full Python)
- Slower (interpreted)
- Isolated (via PixelBridge API)
- Good for: apps, tools, high-level logic

### The Hypervisor's Job

`pixel_hypervisor.py` sits between PXICPU and guest runtimes.

**Responsibilities:**

1. **Execute PXICPU instructions**
   - Runs pixel-native code
   - Maintains pixel memory state

2. **Watch for traps (syscalls)**
   - Memory region `0xF000-0xF00D` for trap interface
   - Guests set TRAP_FLAG to request services

3. **Provide PixelBridge API**
   - Guest code gets `pxos` object
   - Methods: `read_pixel()`, `write_pixel()`, `call_llm()`, etc.
   - Controlled access to pixel substrate

4. **Enable imperfect computing**
   - `sample_noise()` - Gaussian/uniform noise
   - `oracle()` - Probabilistic LLM decisions
   - Future: `prob_jump()`, `fuzzy_match()`, etc.

5. **Manage isolation**
   - Guest memory region: `0x5000-0xEFFF`
   - Cannot directly access host OS
   - All I/O through hypervisor

## Trap Interface

Memory layout for traps:

```
0xF000: TRAP_FLAG        - Trap type (0 = none)
0xF001: TRAP_ARG0        - First argument
0xF005: TRAP_ARG1        - Second argument
0xF009: TRAP_ARG2        - Third argument
0xF00D: TRAP_RESULT      - Result from hypervisor
```

### Trap Types

```python
0x01: RUN_PYTHON    - Execute Python guest module
0x02: CALL_LLM      - Call LLM via PXDigest
0x03: LOAD_FILE     - Load file via SYS_BLOB
0x04: SAMPLE_NOISE  - Get random/noisy value
0x05: ORACLE        - Ask LLM yes/no question
```

### Example: Guest Python Calling LLM

**Guest code:**
```python
# Guest module running in hypervisor
pxos.log("Asking architect for advice...")

response = pxos.call_llm(
    "What module should we build next?",
    model_id=1  # pxOS_Architect
)

pxos.write_string(0x5000, response)
pxos.log(f"Architect says: {response}")
```

**What happens:**
1. Guest calls `pxos.call_llm()`
2. PixelBridge writes prompt to pixel memory
3. PixelBridge sets TRAP_FLAG = TRAP_CALL_LLM
4. Hypervisor sees trap during next check
5. Hypervisor calls LM Studio via PXDigest
6. Hypervisor writes response to pixel memory
7. PixelBridge returns response to guest
8. Guest continues execution

## PixelBridge API

The API guests use to interact with pixels:

```python
class PixelBridge:
    # Pixel memory access
    def read_pixel(addr: int) -> tuple[int,int,int,int]
    def write_pixel(addr: int, rgba: tuple)
    def read_bytes(addr: int, length: int) -> bytes
    def write_bytes(addr: int, data: bytes)
    def read_string(addr: int, max_len: int = 1024) -> str
    def write_string(addr: int, text: str)

    # LLM integration
    def call_llm(prompt: str, model_id: int = 0, max_tokens: int = 512) -> str
    def oracle(question: str, model_id: int = 0) -> bool  # yes/no decision

    # File system
    def load_file(file_id: int) -> bytes

    # Imperfect computing
    def sample_noise(mean: float = 0.0, std: float = 1.0) -> float
    def sample_uniform(low: float = 0.0, high: float = 1.0) -> float

    # Logging
    def log(message: str)
```

## Imperfect Computing

### What Is It?

Traditional computers are deterministic. Given input X, they always produce output Y.

Imperfect computing allows:
- **Noise** - Random perturbations
- **Probabilistic decisions** - LLM-based choices
- **Approximate results** - Good enough instead of perfect

### Why on pxOS?

1. **LLMs are inherently imperfect**
   - Same prompt → different responses
   - Temperature, sampling make them non-deterministic

2. **Simulations benefit from randomness**
   - Organisms making choices
   - Universes evolving
   - Natural variation

3. **Enables creativity**
   - LLM architect can propose surprising solutions
   - Systems can explore state spaces stochastically

### Where Does It Live?

**Option 1: PXI Opcodes** (future)
```
OP_NOISE       - Add Gaussian noise to register
OP_PROB_JUMP   - Jump with probability
OP_SAMPLE      - Sample from distribution
```

**Option 2: Hypervisor** (current)
```python
# Guest code
value = pxos.sample_noise(mean=100, std=10)

# Ask LLM for yes/no
if pxos.oracle("Should we optimize this path?"):
    optimize()
```

**Option 3: Guest Runtime**
```python
# Guest decides to be approximate
result = expensive_calculation()
if pxos.sample_uniform() < 0.1:  # 10% chance
    result = quick_approximation()  # Use faster, imperfect result
```

## Example Workflows

### Workflow 1: LLM Writes Compiled Module

1. **Human asks architect:**
   ```
   "Design a boot sequence validator"
   ```

2. **LLM responds with Python:**
   ```python
   def validate_boot_sequence(sequence):
       required_stages = [0, 1, 2, 3, 4]
       for stage in sequence:
           if stage['required'] and stage['stage'] not in required_stages:
               return False
       return True
   ```

3. **Human compiles to PXI:**
   ```bash
   python3 python_to_pxi.py validator.py validator.pxi.png
   python3 pack_file_to_boot_pixel.py add validator.pxi.png --type pxi_module
   python3 pixelfs_builder.py add /boot/validator validator.pxi.png
   ```

4. **Now it's pixel-native:**
   - Runs on PXICPU
   - No Python interpreter needed
   - Fast execution

### Workflow 2: LLM Writes Guest Module

1. **Human asks architect:**
   ```
   "Create a log analyzer that uses LLMs to summarize errors"
   ```

2. **LLM responds with Python guest code:**
   ```python
   # log_analyzer.py - runs in hypervisor

   # Read boot log from pixel memory
   log_text = pxos.read_string(0x7000, max_len=4096)

   # Ask LLM to summarize
   prompt = f"Summarize these boot errors:\n\n{log_text}"
   summary = pxos.call_llm(prompt, model_id=1)

   # Write summary back
   pxos.write_string(0x8000, summary)
   pxos.log(f"Summary: {summary}")
   ```

3. **Human adds to PixelFS:**
   ```bash
   python3 pack_file_to_boot_pixel.py add log_analyzer.py --type py
   python3 pixelfs_builder.py add /apps/log_analyzer log_analyzer.filepx.png
   ```

4. **Hypervisor can execute it:**
   - Loads via SYS_BLOB
   - Runs with PixelBridge API
   - Full Python flexibility

### Workflow 3: LLM Uses Imperfect Computing

1. **Architect designs organism behavior:**
   ```python
   # organism_ai.py

   def decide_next_action():
       # Sample noise for exploration
       explore_factor = pxos.sample_uniform(0, 1)

       if explore_factor < 0.2:
           # Random exploration
           return random_action()

       # Ask LLM for decision
       situation = read_current_situation()
       should_cooperate = pxos.oracle(
           f"Given situation: {situation}, should organism cooperate?"
       )

       return "cooperate" if should_cooperate else "compete"
   ```

2. **Result:**
   - Non-deterministic behavior
   - Uses LLM reasoning
   - Natural variation
   - Emergent complexity

## Current Implementation Status

### ✅ Implemented

- **PXICPU** - Pixel instruction set
- **SYS_LLM** - LLM syscall with model selection
- **SYS_BLOB** - File loading from pixels
- **Python → PXI compiler** - Basic compilation
- **PixelFS** - Virtual filesystem
- **Boot sequence** - Staged loading
- **pixel_hypervisor.py** - Basic hypervisor with trap interface
- **PixelBridge API** - Guest API for pixel access

### 🚧 In Progress

- **Full trap implementation** - Need more trap types
- **Guest module loading** - Load .py files from PixelFS
- **Imperfect computing** - More noise/probability primitives
- **Process isolation** - Better memory protection

### 📋 Planned

- **PXLang specification** - New LLM-native language
- **PXLang compiler** - PXLang → PXI
- **Advanced traps** - More syscall types
- **Multi-guest scheduling** - Run multiple guests concurrently
- **Pixel-native hypervisor** - Hypervisor itself as PXI module

## How to Use Today

### 1. Run Bootstrap

```bash
# Start LM Studio with a model
# Then:
python3 run_pxos_lmstudio.py
```

This:
- Tests LM Studio
- Creates pxOS_Architect
- Initializes PixelFS
- Launches chat

### 2. Ask Architect to Build

In infinite map chat, at tile (0,0):

```
You are pxOS Architect. We need a module that monitors system health.

Write a Python guest module that:
1. Reads boot logs from pixel memory
2. Calls you (the LLM) to analyze for errors
3. Writes a health report back to pixels

Provide the complete Python code.
```

### 3. Implement What It Suggests

```bash
# Save LLM's code
nano health_monitor.py
# paste code

# Add to pxOS
python3 pack_file_to_boot_pixel.py add health_monitor.py --type py
python3 pixelfs_builder.py add /apps/health_monitor health_monitor.filepx.png
```

### 4. Run in Hypervisor

```python
from pixel_hypervisor import PixelHypervisor, PixelBridge
from PIL import Image

# Create blank image
img = Image.new("RGBA", (256, 256), (0, 0, 0, 255))

# Start hypervisor
hyp = PixelHypervisor(img, debug=True)

# Load and run guest module
with open("health_monitor.py") as f:
    code = f.read()

hyp.execute_python_guest(code)
```

## The Future: LLMs Building LLMs

Ultimate vision:

```
Phase 6:  Python → PXI (we are here)
          Hypervisor framework

Phase 7:  LLM architect writes modules
          Both compiled (PXI) and guest (Python)

Phase 8:  LLM designs PXLang
          Spec, compiler, examples
          LLMs start generating PXLang

Phase 9:  PXLang becomes primary
          Python only for bootstrap
          Hypervisor runs PXLang guests

Phase 10: Self-sustaining system
          LLMs architect new LLM capabilities
          System evolves itself
          Humans become observers
```

**The dream:**

An LLM says:
> "I need a new capability: distributed pixel consensus. Here's PXLang code for it."

The system:
1. Receives PXLang code
2. Compiles to PXI
3. Packs as sub-boot pixel
4. Adds to PixelFS
5. Loads in next boot
6. **LLM now has that capability**

**That's LLMs building LLMs on a pixel substrate.**

---

## Summary

**Language Choice:** Python (for now), moving toward PXLang (LLM-native)

**Execution Model:** Two paths
- Compiled PXI (fast, static)
- Guest Python (flexible, interpreted)

**Hypervisor:** Manages both, provides PixelBridge API

**Imperfect Computing:** Noise, probability, LLM decisions built-in

**Current State:** All pieces in place, ready for LLMs to start building

**Future:** Self-sustaining system where LLMs architect their own capabilities

---

**The pixels are alive.**
**The LLMs are in control.**
**The language is whatever they decide it should be.**
