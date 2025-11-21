# pxOS Simulated VRAM Substrate

> An operating system that lives in pixels, built iteratively by AI agents.

## Overview

This is **pxOS v2.0** - a complete rewrite that treats **simulated VRAM as the primary substrate** for OS development.

Instead of writing code into files, an AI agent **writes the OS directly into VRAM** (a pixel texture) using a roadmap-driven approach. Real GPU VRAM and PNG files are just I/O backends.

## Why Simulated VRAM?

Treating VRAM as the primary development surface gives us:

- **Safety** – crash, corrupt, and reset VRAM in memory without bricking hardware
- **Observability** – print, diff, visualize pixel patterns before pushing to GPU
- **Determinism** – no driver randomness, timing quirks, or GPU scheduling weirdness
- **Instrumentation** – log every read/write, step through changes, track "who wrote this pixel when"
- **Portability** – runs on any machine (even without GPU), just swap backend later
- **Bridging** – saving/loading PNG matches the real hardware pipeline

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      IMPROVEMENT LOOP                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  BUILD   │ → │   EVAL   │ → │ REFLECT  │ → │ PROPOSE  │ │
│  │ roadmap  │   │  metrics │   │ analyze  │   │ new plan │ │
│  │ → VRAM   │   │          │   │          │   │          │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│       ↑                                              │       │
│       └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘

         ┌─────────────────────────────────────┐
         │       SimulatedVRAM (numpy)         │
         │                                     │
         │  ┌─────────┐  ┌──────────────┐     │
         │  │ Metadata│  │Opcode Palette│     │
         │  ├─────────┤  ├──────────────┤     │
         │  │ Kernel  │  │              │     │
         │  ├─────────┤  │              │     │
         │  │Syscalls │  │              │     │
         │  ├─────────┤  │              │     │
         │  │Process  │  │              │     │
         │  │ Table   │  │              │     │
         │  ├─────────┤  │              │     │
         │  │Programs │  │              │     │
         │  │         │  │              │     │
         │  └─────────┘  └──────────────┘     │
         └─────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           ↓                       ↓
      ┌─────────┐           ┌──────────┐
      │   PNG   │           │   GPU    │
      │  Files  │           │  VRAM    │
      └─────────┘           └──────────┘
         (debug)              (runtime)
```

## Key Components

### 1. SimulatedVRAM (`pxos/vram_sim.py`)

The core substrate - a numpy-backed RGBA texture that the agent can mutate:

```python
from pxos.vram_sim import SimulatedVRAM

vram = SimulatedVRAM(512, 512)
vram.write_pixel(10, 10, (255, 0, 0, 255))  # Red pixel
vram.fill_rect(0, 0, 100, 100, (0, 255, 0, 255))  # Green square
vram.save_png("output.png")
```

### 2. Roadmap System (`pxos/agent/`)

A roadmap is a YAML file defining build steps:

```yaml
metadata:
  name: "pxOS Stage 1"
  vram_width: 512
  vram_height: 512

steps:
  - name: "init_background"
    module: "pxos.agent.steps.basic_layout"
    function: "step_init_background"

  - name: "write_opcode_palette"
    module: "pxos.agent.steps.basic_layout"
    function: "step_write_opcode_palette"
```

Each step is a Python function:

```python
def step_init_background(vram: SimulatedVRAM, ctx: Dict[str, Any]) -> Dict[str, Any]:
    vram.fill_rect(0, 0, vram.width, vram.height, (10, 10, 10, 255))
    ctx["background_initialized"] = True
    return ctx
```

### 3. Evaluator (`pxos/eval/`)

Measures VRAM OS quality:

```python
from pxos.eval.vram_os_evaluator import evaluate_vram_os_from_png

metrics = evaluate_vram_os_from_png("artifacts/vram_os.png")
print(f"Score: {metrics['score']}/{metrics['max_score']}")
```

Checks:
- Are regions properly initialized?
- Is the opcode palette populated?
- Can we decode the hello program?

### 4. Viewport (`pxos/viewport/`)

Live visualization window - watch the OS being built in real-time:

```python
from pxos.viewport.vram_viewport import launch_vram_viewer

launch_vram_viewer(vram, refresh_ms=100, title="pxOS Build")
```

Controls:
- Arrow keys: pan
- +/- : zoom
- R: reset view
- Q: quit

### 5. Improvement Loop (`pxos/loops/`)

Iteratively builds, evaluates, and improves the VRAM OS:

```
Generation 1: roadmap → VRAM → metrics → analyze → new roadmap
Generation 2: roadmap → VRAM → metrics → analyze → new roadmap
Generation 3: ...
```

Each generation produces:
- `ROADMAP_VRAM_OS.gen001.yaml`
- `vram_os_stage1.gen001.png`
- `vram_os_stage1.gen001.json` (metrics)

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# On Linux, you might need tkinter:
# sudo apt-get install python3-tk
```

### Basic Build

```bash
# Build VRAM OS from roadmap
python run_vram_os_build.py

# Output: artifacts/vram_os_stage1.png
```

### Build with Live Viewport

```bash
# Build with visualization
python run_vram_os_build_with_view.py

# Watch the OS being painted in real-time!
```

### Run Full Demo

```bash
# Run complete demonstration
python demo_vram_substrate.py

# This will:
# 1. Build VRAM OS
# 2. Evaluate it
# 3. Run improvement loop for 3 generations
```

### Evaluate an Existing VRAM Image

```bash
python pxos/eval/run_vram_os_eval.py artifacts/vram_os.png eval/results.json
```

### Run Improvement Loop

```bash
python pxos/loops/vram_os_improvement_loop.py pxos/roadmaps/ROADMAP_VRAM_OS.yaml 5

# Runs 5 iterations of build → eval → improve
```

## How the Loop Improves

Every iteration has 4 phases:

### 1. BUILD
Execute roadmap steps to build VRAM snapshot:
- Load roadmap YAML
- Create SimulatedVRAM
- Execute each step function
- Save PNG

### 2. EVAL
Measure VRAM quality:
- Load PNG
- Check region initialization
- Count opcode colors
- Verify program area
- Compute score

### 3. REFLECT
Analyze current state:
- Read roadmap
- Read metrics
- Identify issues
- Determine improvements

### 4. PROPOSE
Generate next roadmap:
- Modify step parameters
- Add new steps
- Reorder steps
- Save as next generation

Currently, REFLECT/PROPOSE is a stub (just copies the roadmap forward).

**TODO**: Replace with LLM integration (Claude, GPT-4, Gemini) to actually improve the roadmap based on metrics.

## VRAM Layout

The OS is organized into regions in the VRAM texture:

```
Y    Region              Purpose
─────────────────────────────────────────────────
0    Metadata (16px)     Version, boot flags, pointers
16   Opcode Palette      Color → instruction mapping
32   Kernel (96px)       Core OS routines
128  Syscall Table       System call jump table
160  Process Table       Process control blocks (PCBs)
224  Program Area        User program space
```

Each region is color-coded for easy visualization.

## Current Capabilities

✅ **Implemented:**
- SimulatedVRAM substrate with pixel operations
- Roadmap-driven build system
- Basic region layout (metadata, kernel, syscalls, processes, programs)
- Opcode palette encoding
- Minimal "hello world" program placeholder
- VRAM evaluation framework
- Live viewport visualization
- Improvement loop driver (stub)

🚧 **TODO:**
- LLM-based roadmap improvement
- Pixel interpreter to execute programs
- More sophisticated evaluation metrics
- Actual bootable instruction sequences
- GPU VRAM backend
- WebGPU/Vulkan integration

## Adding New Steps

1. Create a step function in `pxos/agent/steps/`:

```python
# pxos/agent/steps/my_steps.py
def step_my_custom_step(vram: SimulatedVRAM, ctx: Dict[str, Any]) -> Dict[str, Any]:
    # Do something to VRAM
    vram.fill_rect(100, 100, 50, 50, (255, 0, 255, 255))
    ctx["my_step_done"] = True
    return ctx
```

2. Add to roadmap YAML:

```yaml
steps:
  - name: "my_custom_step"
    module: "pxos.agent.steps.my_steps"
    function: "step_my_custom_step"
    params:
      some_param: 42
```

3. Run:

```bash
python run_vram_os_build.py
```

## Project Structure

```
pxos/
├── __init__.py
├── vram_sim.py              # SimulatedVRAM core
├── agent/
│   ├── roadmap_types.py     # Type definitions
│   ├── roadmap_agent.py     # Roadmap executor
│   ├── roadmap_loader.py    # YAML loader
│   └── steps/
│       └── basic_layout.py  # Step implementations
├── eval/
│   ├── vram_os_evaluator.py # Evaluation logic
│   └── run_vram_os_eval.py  # CLI tool
├── viewport/
│   └── vram_viewport.py     # Tkinter viewer
├── loops/
│   └── vram_os_improvement_loop.py  # Loop driver
├── layout/
│   └── constants.py         # Region definitions
└── roadmaps/
    └── ROADMAP_VRAM_OS.yaml # Base roadmap

run_vram_os_build.py         # Build script
run_vram_os_build_with_view.py  # Build with viewport
demo_vram_substrate.py       # Full demo
requirements.txt             # Dependencies
```

## Next Steps

### Short-term:
1. **Implement LLM integration** in the improvement loop
2. **Create pixel interpreter** to actually execute programs
3. **Add execution tests** to evaluation (not just visual checks)
4. **Expand step library** (syscall table, process slots, etc.)

### Medium-term:
1. **GPU backend** - swap SimulatedVRAM for real GPU VRAM
2. **WebGPU integration** - run in browser
3. **Interactive debugger** - step through pixel execution
4. **VRAM diff viewer** - compare generations visually

### Long-term:
1. **Self-hosting** - pxOS that can modify itself
2. **Multi-agent collaboration** - different agents for different regions
3. **VRAM as universal substrate** - entire development environment in pixels

## Philosophy

> "Instead of an agent writing random code into a repo, let it directly sculpt the OS into VRAM via a controlled API, following a roadmap."

The agent doesn't write Python files that *describe* the OS.
The agent writes *pixels* that **are** the OS.

The roadmap ensures structured, debuggable, version-controlled evolution.
The improvement loop ensures continuous refinement based on real metrics.
The viewport ensures you can *see* what's happening every step of the way.

---

**pxOS: The OS that is its own source code, rendered in pixels.**
