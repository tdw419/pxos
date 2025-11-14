# pxOS GPU Terminal - Frozen Shader Architecture

**Version: 0.1**

## Overview

The GPU Terminal implements the "Frozen Shader Bus" pattern:

- **Frozen WGSL shader** that never changes (debugging bliss!)
- **CPU-side VRAM buffer** (numpy array) that the LLM manipulates
- **Data-only evolution** - change behavior by changing buffer contents, not shader code
- **Easy debugging** - all logic is in Python, shader is off the table as a bug source

## Architecture

```
┌─────────────────────────────────────────┐
│  LLM Terminal Commands (Python)         │
│  ↓                                       │
│  CPU VRAM Buffer (numpy)                │  ← All logic here (debuggable!)
│  ↓                                       │
│  Upload to GPU Texture                  │
│  ↓                                       │
│  Frozen Shader (frozen_display.wgsl)    │  ← Never changes!
│  ↓                                       │
│  Screen                                  │
└─────────────────────────────────────────┘
```

## Why This Works

**The Problem:** Shader debugging is hard. GPU state is opaque.

**The Solution:** Make the shader a dumb "display wire":
- Shader just samples a texture and displays it
- All drawing logic happens in Python (numpy)
- You can `print()` the VRAM buffer any time
- Shader binary stays frozen → no recompilation, no GPU-side bugs

## Files

```
gpu_terminal/
├── shaders/
│   └── frozen_display.wgsl    # v0.1 - NEVER CHANGES
├── pxos_gpu_terminal.py        # Main GPU terminal
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Installation

```bash
cd gpu_terminal
pip install -r requirements.txt
```

## Usage

### Run the demo

```bash
python pxos_gpu_terminal.py
```

This will open a window with some test graphics demonstrating the terminal commands.

### Use in code

```python
from pxos_gpu_terminal import PxOSTerminalGPU

# Create terminal
terminal = PxOSTerminalGPU()

# Execute commands
terminal.cmd_clear(0, 0, 0)              # Clear to black
terminal.cmd_pixel(100, 100, 255, 0, 0)  # Red pixel
terminal.cmd_rect(50, 50, 200, 100, 0, 255, 0)  # Green rectangle

# Start event loop
terminal.run()
```

## Terminal Commands (v0.1)

### CLEAR

```python
terminal.cmd_clear(r, g, b, a=255)
```

Clear entire screen to a solid color.

**Example:**
```python
terminal.cmd_clear(0, 0, 64)  # Dark blue background
```

### PIXEL

```python
terminal.cmd_pixel(x, y, r, g, b, a=255)
```

Set a single pixel. Origin (0,0) is top-left.

**Example:**
```python
terminal.cmd_pixel(400, 300, 255, 255, 255)  # White center pixel
```

### RECT

```python
terminal.cmd_rect(x, y, w, h, r, g, b, a=255)
```

Draw a filled rectangle.

**Example:**
```python
terminal.cmd_rect(100, 100, 200, 150, 255, 0, 0)  # Red rectangle
```

## Design Principles

### 1. Frozen Shader = Stable Foundation

The `frozen_display.wgsl` shader is **version 0.1** and should **never change**.

It's literally just:
- Vertex shader: pass through positions and UVs
- Fragment shader: sample texture and return color

That's it. No logic, no conditionals, no complexity.

### 2. CPU-Side Logic = Debuggable

All rendering logic lives in Python:
- Direct access to the `vram` numpy array
- Can print/inspect it any time
- Can validate it before upload
- Easy to unit test

### 3. Data Protocol Evolution

To add features, you change what you write into VRAM, not the shader:

- Want text rendering? Write glyph pixels into VRAM
- Want sprites? Blit sprite data into VRAM
- Want compute? Add a command buffer that a frozen compute shader reads

The shader stays frozen; the data protocol evolves.

## Debugging

### Check VRAM contents

```python
# Print a small region
print(terminal.vram[100:110, 100:110, 0])  # Red channel

# Validate before upload
assert terminal.vram.min() >= 0
assert terminal.vram.max() <= 255

# Save to file for inspection
from PIL import Image
Image.fromarray(terminal.vram).save("debug_vram.png")
```

### Shader is not the problem

If something looks wrong:
1. ✅ Check terminal command logic (Python)
2. ✅ Check VRAM buffer contents (numpy)
3. ✅ Check upload parameters (texture write)
4. ❌ ~~Check shader code~~ (it's frozen!)

## Next Steps

### Phase 1: Raster Terminal ✅ (Current)
- [x] Frozen display shader
- [x] CPU VRAM buffer
- [x] Basic commands (CLEAR, PIXEL, RECT)

### Phase 2: Text Terminal
- [ ] Bitmap font loading
- [ ] GLYPH command (draw character)
- [ ] TEXT command (draw string)
- [ ] Cursor state

### Phase 3: Advanced Graphics
- [ ] BLIT command (copy buffer region)
- [ ] LINE command
- [ ] CIRCLE command
- [ ] SPRITE command

### Phase 4: LLM Integration
- [ ] Terminal command parser
- [ ] LLM prompt interface
- [ ] pxVM integration
- [ ] Command history/replay

### Phase 5: Frozen Compute (optional)
- [ ] Second frozen compute shader
- [ ] Command buffer protocol
- [ ] Move some logic to GPU (while keeping shaders frozen)

## Philosophy

**"The best GPU architecture is one where the GPU isn't the problem."**

By freezing the shader, we've eliminated an entire class of bugs.
By keeping logic in Python, we've made everything inspectable and debuggable.
By using numpy for VRAM, we've made it easy to reason about and test.

This is the foundation for a stable, LLM-friendly GPU terminal.

## License

MIT (see LICENSE file in repository root)
