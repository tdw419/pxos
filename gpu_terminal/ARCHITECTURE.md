# Frozen Shader Architecture

## The Problem

Traditional GPU programming has several debugging challenges:

1. **Opaque GPU state** - Hard to inspect what's happening on the GPU
2. **Shader compilation** - Changes require recompilation and pipeline recreation
3. **GPU-side bugs** - Difficult to debug shader code
4. **LLM unfriendly** - WGSL/GLSL are harder for LLMs to reason about than Python

## The Solution: Frozen Shader Bus

Think of the GPU as a **wire**, not as a **processor**.

### Core Principle

> **One frozen shader + evolving data/terminal on the CPU side.**
> We debug *data & protocol*, not shader code.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: LLM/User Interface                                 │
│   - Natural language commands                               │
│   - Python API calls                                        │
│   - Terminal command parser                                 │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: Terminal Commands (Python)                         │
│   - cmd_clear(r, g, b, a)                                   │
│   - cmd_pixel(x, y, r, g, b, a)                             │
│   - cmd_rect(x, y, w, h, r, g, b, a)                        │
│   - [Future: cmd_text, cmd_blit, cmd_sprite]                │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: CPU VRAM Buffer                                    │
│   - numpy array: shape (600, 800, 4), dtype uint8          │
│   - Directly inspectable and debuggable                     │
│   - Can print, save to PNG, validate ranges                 │
│   - Upload to GPU each frame                                │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: Frozen Shader (WGSL)                               │
│   - Version 0.1 - NEVER CHANGES                             │
│   - Input: texture_2d<f32>                                  │
│   - Output: screen pixels                                   │
│   - Logic: sample(texture, uv)                              │
├─────────────────────────────────────────────────────────────┤
│ Layer 0: GPU Hardware                                       │
│   - WebGPU/wgpu-py                                          │
│   - Texture upload                                          │
│   - Render pipeline                                         │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Frame Rendering

```
CPU Side:                           GPU Side:
┌─────────────┐
│ Terminal    │
│ Commands    │
│ (Python)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ VRAM Buffer │
│ (numpy)     │
└──────┬──────┘
       │ write_texture()
       ▼
              ┌─────────────┐
              │ GPU Texture │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ Frozen      │
              │ Shader      │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ Screen      │
              └─────────────┘
```

## Why This Works

### 1. Stable Foundation

The frozen shader is **trivially simple**:

```wgsl
@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
    return textureSample(img, samp, in.uv);
}
```

There's literally nothing to go wrong here. It's off the table as a bug source.

### 2. Debuggable Logic

All rendering logic is in Python:

```python
def cmd_rect(self, x, y, w, h, r, g, b, a=255):
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(WIDTH, x + w)
    y2 = min(HEIGHT, y + h)
    self.vram[y1:y2, x1:x2] = [r, g, b, a]
```

You can:
- Print the VRAM before/after
- Save it to a PNG
- Validate ranges
- Unit test the logic
- Step through with a debugger

### 3. LLM-Friendly

The LLM only needs to:
1. Understand Python (which it's good at)
2. Manipulate numpy arrays (which it's good at)
3. Issue terminal commands (simple API)

It doesn't need to:
- Write WGSL shaders
- Understand GPU pipeline state
- Debug GPU-side issues

## Evolution Path

### Phase 1: Raster Terminal ✅

Current implementation:
- Frozen display shader
- CPU VRAM buffer
- Basic commands: CLEAR, PIXEL, RECT

### Phase 2: Text Terminal

Add text rendering without changing the shader:
- Load bitmap font into memory
- Add `cmd_glyph(x, y, char_code)` - blits character pixels into VRAM
- Add `cmd_text(x, y, string)` - draws string using glyphs
- All logic in Python, shader still frozen

### Phase 3: Advanced Graphics

More terminal commands, shader still frozen:
- `cmd_blit(src_x, src_y, dst_x, dst_y, w, h)` - copy VRAM region
- `cmd_line(x1, y1, x2, y2, r, g, b)` - draw line (Bresenham)
- `cmd_circle(x, y, radius, r, g, b)` - draw circle
- `cmd_sprite(x, y, sprite_id)` - blit sprite from atlas

### Phase 4: Frozen Compute (Optional)

If we need GPU acceleration while keeping stability:
- Add a **second frozen compute shader**
- Shader reads a "command buffer" and writes to VRAM texture
- Commands become data structures instead of CPU operations
- Shader is still frozen, only command data changes

Example:
```python
# Instead of:
for x in range(1000):
    terminal.cmd_pixel(x, y, r, g, b)

# We can upload a command buffer:
terminal.cmd_buffer.append({
    'type': 'LINE',
    'x1': 0, 'y1': y,
    'x2': 999, 'y2': y,
    'color': (r, g, b, a)
})
terminal.upload_commands()
# Frozen compute shader executes commands
```

## Design Principles

### 1. Minimize GPU Complexity

The GPU should be as dumb as possible. It's a display, not a computer.

### 2. Maximize CPU Visibility

All state should be inspectable from Python. No hidden GPU state.

### 3. Freeze Binaries, Evolve Data

Never change the shader code. Change what you feed into it.

### 4. Optimize Last

Get it working first. Make it debuggable second. Make it fast third.

## Debugging Strategy

When something looks wrong:

```
1. Is the terminal command correct?
   → Check Python logic

2. Is the VRAM buffer correct?
   → Print/save numpy array

3. Is the upload correct?
   → Check texture write parameters

4. Is the shader correct?
   → It's frozen, skip this step
```

Three places to look instead of dozens.

## Comparison to Traditional Approach

### Traditional GPU Programming

```
Python → WGSL Shader → GPU State → Screen
         ↑
         Changes frequently
         Hard to debug
         Opaque to LLM
```

### Frozen Shader Bus

```
Python → numpy VRAM → Frozen Shader → Screen
  ↑           ↑             ↑
  Easy     Visible      Never changes
```

## Future Extensions

### Multi-Buffer Support

Add more frozen shaders for different purposes:

- `frozen_display.wgsl` - main display (current)
- `frozen_compute_draw.wgsl` - execute draw commands
- `frozen_compute_blit.wgsl` - fast texture operations
- `frozen_post_process.wgsl` - screen effects

Each is frozen, only the data protocol evolves.

### Command Protocol

Define a binary command format:

```python
class CommandType(Enum):
    CLEAR = 0
    PIXEL = 1
    RECT = 2
    BLIT = 3
    TEXT = 4

# Commands are just structs:
cmd = struct.pack('Biiii', CommandType.RECT, x, y, w, h)
```

Frozen compute shader reads and executes these.

## Philosophy

**"The best code is code that doesn't change."**

By freezing the shader at v0.1, we've created a stable foundation.
Everything above it can evolve, but the GPU layer is solid.

This is the pxOS way:
- Simple, stable foundations
- Debuggable at every layer
- LLM-friendly design
- Evolution through data, not code

---

**Status:** v0.1 - Foundation complete
**Next:** Text rendering (Phase 2)
