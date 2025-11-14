# pxOS GPU Terminal - Machine Code for LLMs

**Machine code for LLMs, built by LLMs, starting with the terminal.**

This is the foundational graphics layer for pxOS, implementing the "Frozen Shader Bus" architecture where LLMs work with structured data and stable machine code instead of complex GPU programming.

## Architecture Overview

```
┌───────────────────────────────────────────────────────────┐
│ LLM / Natural Language                                    │
│   "Draw a house with a red roof"                          │
├───────────────────────────────────────────────────────────┤
│ PXSCENE v0.1 (JSON)                    ← LLM-friendly    │
│   Structured scene description                            │
├───────────────────────────────────────────────────────────┤
│ pxscene_compile.py                     ← Compiler         │
│   JSON → Text conversion                                  │
├───────────────────────────────────────────────────────────┤
│ PXTERM v1 (Text Commands)              ← Machine code     │
│   CLEAR, PIXEL, RECT, HLINE, VLINE     ← 🔒 FROZEN       │
├───────────────────────────────────────────────────────────┤
│ pxos_llm_terminal.py                   ← Executor         │
│   Parse and execute PXTERM                                │
├───────────────────────────────────────────────────────────┤
│ pxos_gpu_terminal.py                   ← GPU abstraction  │
│   Layer management + compositing                          │
├───────────────────────────────────────────────────────────┤
│ CPU VRAM (numpy arrays)                ← Debuggable!      │
│   One buffer per layer                                    │
├───────────────────────────────────────────────────────────┤
│ Frozen Shader v0.1 (WGSL)              ← Never changes    │
│   textureSample(img, uv) → screen      ← 🔒 FROZEN       │
└───────────────────────────────────────────────────────────┘
```

## Key Concepts

### 1. Frozen Shader Bus

> **One frozen shader + evolving data/terminal on the CPU side.**
> We debug *data & protocol*, not shader code.

The WGSL shader is **frozen at v0.1** and never changes. It does exactly one thing: sample a texture and display it. All logic happens in Python.

### 2. PXTERM v1 - The Machine Code

**PXTERM** (Pixel Terminal Machine Code) is the stable, frozen instruction set:

```text
CLEAR 0 0 0
LAYER NEW ui 10
LAYER USE ui
RECT 100 100 200 150 255 0 0
SAVE output.png
```

- **Status**: 🔒 **FROZEN** - v1 will not change
- **Format**: Line-oriented text commands
- **Target**: LLMs and compilers can generate this directly

See: [`PXTERM_SPEC.md`](./PXTERM_SPEC.md)

### 3. PXSCENE v0.1 - The High-Level Language

**PXSCENE** is a JSON scene description language for LLMs:

```json
{
  "layers": [
    {
      "name": "background",
      "z": 0,
      "commands": [
        {"op": "CLEAR", "color": [0, 0, 0]},
        {"op": "RECT", "x": 100, "y": 100, "w": 200, "h": 150, "color": [255, 0, 0]}
      ]
    }
  ]
}
```

- **Status**: ✨ **Evolvable** - Can add features
- **Format**: Structured JSON
- **Target**: Easy for LLMs to generate correctly

See: [`PXSCENE_SPEC.md`](./PXSCENE_SPEC.md)

### 4. Layered Composition

Like Photoshop, everything is layers:

- Each layer has its own RGBA buffer
- Layers have z-index (back to front)
- Layers alpha-blend during composition
- All composition happens in Python (numpy)

```python
terminal.layer_new("background", z=0)
terminal.layer_new("ui", z=10)
terminal.layer_use("ui")
terminal.cmd_rect(100, 100, 200, 150, 255, 0, 0)
```

## LLM Quick Start 🤖

**This is the easiest way to use the system. Perfect for LLMs and humans alike.**

### 1. Use the One-Shot Runner

```bash
# Just run any PXSCENE JSON file:
python pxscene_run.py examples/scene1_basic.json

# That's it! Opens window with result and saves PNG
```

### 2. Have an LLM Generate Scenes

**Step 1**: Give your LLM the prompt from [`PROMPTS.md`](./PROMPTS.md):

```text
You are a graphics compiler assistant for pxOS...
[Full prompt in PROMPTS.md]
```

**Step 2**: Ask for a scene:

```text
Draw a sunset scene with orange sky and green ground
```

**Step 3**: Save the JSON output as `scene.json`

**Step 4**: Run it:

```bash
python pxscene_run.py scene.json
```

**See**: [`PROMPTS.md`](./PROMPTS.md) for complete LLM integration guide.

---

## Quick Start

### Installation

```bash
cd gpu_terminal
pip install -r requirements.txt
```

### Method 1: One-Shot Runner (Easiest)

```bash
python pxscene_run.py examples/scene1_basic.json
```

### Method 2: Write PXSCENE JSON (Recommended for LLMs)

Create `my_scene.json`:

```json
{
  "canvas": {"clear": [0, 0, 32]},
  "layers": [
    {
      "name": "main",
      "z": 0,
      "commands": [
        {"op": "RECT", "x": 100, "y": 100, "w": 200, "h": 150, "color": [255, 0, 0]}
      ]
    }
  ],
  "output": {"file": "my_scene.png"}
}
```

Compile and run:

```bash
python pxscene_compile.py my_scene.json my_scene.pxterm
python pxos_llm_terminal.py my_scene.pxterm
```

### Method 2: Write PXTERM Directly

Create `program.pxterm`:

```text
CLEAR 0 0 0
LAYER NEW main 10
LAYER USE main
RECT 100 100 200 150 255 0 0
SAVE output.png
```

Run:

```bash
python pxos_llm_terminal.py program.pxterm
```

### Method 3: Use Python API

```python
from pxos_gpu_terminal import PxOSTerminalGPU

terminal = PxOSTerminalGPU()
terminal.cmd_clear(0, 0, 64)
terminal.cmd_rect(100, 100, 200, 150, 255, 0, 0)
terminal.save_frame("output.png")
terminal.run()
```

## Examples

### Basic Scene

```bash
python pxscene_compile.py examples/scene1_basic.json
python pxos_llm_terminal.py examples/scene1_basic.pxterm
```

### UI Windows

```bash
python pxscene_compile.py examples/scene2_ui.json
python pxos_llm_terminal.py examples/scene2_ui.pxterm
```

### House Drawing

```bash
python pxscene_compile.py examples/scene3_house.json
python pxos_llm_terminal.py examples/scene3_house.pxterm
```

## File Structure

```
gpu_terminal/
├── shaders/
│   └── frozen_display.wgsl       # 🔒 FROZEN v0.1 shader
│
├── examples/
│   ├── scene1_basic.json         # Basic shapes
│   ├── scene2_ui.json            # UI windows
│   └── scene3_house.json         # Artistic scene
│
├── pxos_gpu_terminal.py          # GPU terminal (low-level)
├── pxos_llm_terminal.py          # PXTERM executor
├── pxscene_compile.py            # PXSCENE → PXTERM compiler
├── pxscene_run.py                # 🔥 One-shot runner (easiest!)
├── test_pipeline.py              # Pipeline test suite
│
├── PXTERM_SPEC.md                # Machine code spec (v1)
├── PXSCENE_SPEC.md               # Scene language spec (v0.1)
├── PROMPTS.md                    # 🤖 LLM integration guide
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## PXTERM v1 Commands

### Introspection

- `INFO` - Show canvas size and current layer
- `LAYERS` - List all layers
- `HELP` - Show help

### Layer Management

- `LAYER NEW name z` - Create layer with z-index
- `LAYER USE name` - Switch to layer
- `LAYER DELETE name` - Delete layer

### Drawing

- `CLEAR r g b [a]` - Fill layer with color
- `PIXEL x y r g b [a]` - Draw single pixel
- `RECT x y w h r g b [a]` - Draw rectangle
- `HLINE x y length r g b [a]` - Draw horizontal line
- `VLINE x y length r g b [a]` - Draw vertical line

### Utility

- `SAVE path` - Save frame to PNG
- `QUIT` - Exit

## Why This Works

### For Debugging

✅ **Shader is frozen** → Off the table as a bug source
✅ **VRAM is numpy** → Can print, save to PNG, validate
✅ **Logic is Python** → Step-through debugger works
✅ **Commands are text** → Just read them

### For LLMs

✅ **No GPU knowledge needed** → No WGSL, no WebGPU
✅ **Structured JSON** → Easy to generate correctly
✅ **Clear errors** → Validation at compile time
✅ **Stable target** → PXTERM v1 is frozen

### For Performance

✅ **Numpy operations** → Vectorized, fast
✅ **Layer composition** → One upload per frame
✅ **No Python loops** → RECT fills entire regions at once

## The Philosophy

> "The best GPU architecture is one where the GPU isn't the problem."

By freezing the shader at v0.1, we've created an unbreakable foundation. Everything above it can evolve:

- ✨ PXSCENE can add new operations
- ✨ New languages can compile to PXTERM
- ✨ pxVM can be added later
- 🔒 But PXTERM v1 and the shader stay frozen

## Compilation Pipeline

```bash
# Human writes natural language
"Draw a house with a red roof"

# LLM generates PXSCENE JSON
{
  "layers": [{
    "name": "house",
    "z": 10,
    "commands": [
      {"op": "RECT", "x": 250, "y": 250, "w": 300, "h": 200, "color": [222, 184, 135]},
      # ... roof, door, windows ...
    ]
  }]
}

# Compiler generates PXTERM
LAYER NEW house 10
LAYER USE house
RECT 250 250 300 200 222 184 135
# ...

# Terminal executes PXTERM
# GPU displays result
```

## Development Workflow

### For LLM Developers

1. Generate PXSCENE JSON
2. Validate structure
3. Compile to PXTERM
4. Execute and verify
5. Iterate

### For System Developers

1. Keep PXTERM v1 frozen
2. Add features to PXSCENE
3. Update compiler
4. Keep GPU terminal stable
5. Never touch the shader

## Future Directions

### Phase 2: Text Rendering

- Bitmap font loading
- TEXT operation in PXSCENE
- Terminal emulator layer

### Phase 3: pxVM Integration

- VM bytecode → PXTERM
- Stack-based execution
- Programs that draw

### Phase 4: Advanced Graphics

- LINE (Bresenham)
- CIRCLE (midpoint)
- BLIT (sprite/texture)
- GRADIENT fills

### Phase 5: Direct LLM Integration

- LLM API → PXSCENE generation
- Natural language → graphics
- Iterative refinement

## Specifications

- **PXTERM v1**: [`PXTERM_SPEC.md`](./PXTERM_SPEC.md) - 🔒 FROZEN
- **PXSCENE v0.1**: [`PXSCENE_SPEC.md`](./PXSCENE_SPEC.md) - ✨ EVOLVABLE

## Contributing

When adding features:

1. ✅ **DO** add operations to PXSCENE
2. ✅ **DO** update the compiler
3. ✅ **DO** add examples
4. ❌ **DON'T** change PXTERM v1
5. ❌ **DON'T** touch the frozen shader

## License

Part of the pxOS project.

## Version

- **GPU Terminal**: v0.1
- **PXTERM**: v1.0 (🔒 FROZEN)
- **PXSCENE**: v0.1 (✨ EVOLVABLE)
- **Frozen Shader**: v0.1 (🔒 FROZEN)

---

**Machine code for LLMs. Built by LLMs. Starting with the terminal.**
