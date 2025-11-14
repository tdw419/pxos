# PXSCENE Specification

## Overview

PXSCENE is a JSON-based scene description language for creating layered 2D graphics. It compiles to PXTERM instructions and renders via a GPU-accelerated terminal.

**Pipeline:**
```
PXSCENE (JSON) → pxscene_compile → PXTERM (text) → pxos_llm_terminal → GPU → PNG
```

## Architecture

### Version: v0.2

**New in v0.2:**
- Layout operators: `HSTACK`, `VSTACK`
- Text rendering: `TEXT` operation
- Multi-layer compositing with z-ordering

## PXSCENE Format

### Top-Level Structure

```json
{
  "canvas": {
    "width": 800,
    "height": 600,
    "clear": [0, 0, 0, 255]
  },
  "layers": [
    {
      "name": "background",
      "z": 0,
      "clear": [0, 0, 40, 255],
      "commands": [...]
    },
    {
      "name": "ui",
      "z": 10,
      "commands": [...]
    }
  ],
  "output": {
    "file": "output.png"
  }
}
```

### Canvas Object

- `width`, `height`: Canvas dimensions in pixels (default: 800×600)
- `clear`: Optional default clear color [r, g, b, a]

### Layer Object

- `name`: Layer identifier (string)
- `z`: Z-order for compositing (integer, lower = back)
- `clear`: Optional layer clear color [r, g, b, a]
- `commands`: Array of drawing commands

### Output Object

- `file`: Output filename (string, typically .png)

## Drawing Commands

All commands support `"color": [r, g, b]` or `"color": [r, g, b, a]` (0-255).

### RECT - Filled Rectangle

```json
{
  "op": "RECT",
  "x": 100,
  "y": 80,
  "w": 600,
  "h": 60,
  "color": [20, 20, 80, 255]
}
```

Compiles to: `RECT x y w h r g b a`

### PIXEL - Single Pixel

```json
{
  "op": "PIXEL",
  "x": 10,
  "y": 10,
  "color": [255, 0, 0, 255]
}
```

Compiles to: `PIXEL x y r g b a`

### HLINE - Horizontal Line

```json
{
  "op": "HLINE",
  "x": 0,
  "y": 100,
  "w": 800,
  "color": [128, 128, 128, 255]
}
```

Compiles to: `HLINE x y w r g b a`

### TEXT - Bitmap Text Rendering

```json
{
  "op": "TEXT",
  "x": 120,
  "y": 95,
  "color": [255, 255, 255, 255],
  "value": "HELLO WORLD"
}
```

**Semantics:**
- `x`, `y`: Top-left starting position (pixels)
- `color`: [r, g, b] or [r, g, b, a] (0-255)
- `value`: Text string (spaces allowed, `\n` for newlines)

Compiles to: `TEXT x y r g b [a] message...`

The terminal renders text using an 8×8 bitmap font. Characters are uppercase only (lowercase auto-converts). Unknown characters render as space.

## Layout Operators

Layout operators position children automatically. Children can be any drawing command, including nested layouts.

### HSTACK - Horizontal Stack

Arranges children left-to-right with optional spacing.

```json
{
  "op": "HSTACK",
  "x": 125,
  "y": 240,
  "spacing": 20,
  "children": [
    { "op": "RECT", "w": 150, "h": 120, "color": [255, 0, 0, 180] },
    { "op": "RECT", "w": 150, "h": 120, "color": [0, 255, 0, 180] },
    { "op": "RECT", "w": 150, "h": 120, "color": [0, 0, 255, 180] }
  ]
}
```

**Expansion:**
- First child: x=125, y=240
- Second child: x=125+150+20=295, y=240
- Third child: x=295+150+20=465, y=240

Children **must** have a `w` (width) field.

### VSTACK - Vertical Stack

Arranges children top-to-bottom with optional spacing.

```json
{
  "op": "VSTACK",
  "x": 120,
  "y": 200,
  "spacing": 10,
  "children": [
    { "op": "RECT", "w": 300, "h": 40, "color": [40, 40, 100, 255] },
    { "op": "RECT", "w": 300, "h": 40, "color": [40, 40, 100, 255] }
  ]
}
```

**Expansion:**
- First child: x=120, y=200
- Second child: x=120, y=200+40+10=250

Children **must** have an `h` (height) field.

### Nested Layouts

Layouts can be nested for complex structures:

```json
{
  "op": "HSTACK",
  "x": 50,
  "y": 50,
  "spacing": 20,
  "children": [
    {
      "op": "VSTACK",
      "spacing": 10,
      "children": [
        { "op": "RECT", "w": 100, "h": 50, "color": [255, 0, 0, 255] },
        { "op": "RECT", "w": 100, "h": 50, "color": [0, 255, 0, 255] }
      ]
    },
    {
      "op": "RECT",
      "w": 100,
      "h": 110,
      "color": [0, 0, 255, 255]
    }
  ]
}
```

This creates a 2×1 grid where the left cell contains a vertical stack.

## PXTERM Instruction Set

PXTERM is the low-level instruction format. PXSCENE compiles to this.

### Commands

```
CANVAS width height          # Initialize canvas
LAYER name z                 # Create layer with z-order
SELECT name                  # Switch to layer
CLEAR r g b [a]              # Clear layer to color
PIXEL x y r g b [a]          # Draw pixel
HLINE x y w r g b [a]        # Draw horizontal line
RECT x y w h r g b [a]       # Draw filled rectangle
TEXT x y r g b [a] message   # Draw text (rest of line)
DRAW [filename]              # Render and save
```

Comments start with `#`.

## Usage

### Quick Start

```bash
python pxscene_run.py scene.json
```

This compiles `scene.json` → `scene.pxterm` and renders to the specified output file.

### Step-by-Step

1. Compile: `python pxscene_compile.py scene.json scene.pxterm`
2. Execute: `python pxos_llm_terminal.py scene.pxterm output.png`

## Examples

### Basic Window with Title

```json
{
  "canvas": {
    "width": 800,
    "height": 600,
    "clear": [0, 0, 0, 255]
  },
  "layers": [
    {
      "name": "background",
      "z": 0,
      "commands": [
        { "op": "RECT", "x": 0, "y": 0, "w": 800, "h": 600, "color": [0, 0, 40, 255] }
      ]
    },
    {
      "name": "ui",
      "z": 10,
      "commands": [
        {
          "op": "RECT",
          "x": 100,
          "y": 80,
          "w": 600,
          "h": 60,
          "color": [20, 20, 80, 255]
        },
        {
          "op": "TEXT",
          "x": 120,
          "y": 95,
          "color": [255, 255, 255, 255],
          "value": "PXOS WINDOW TITLE"
        }
      ]
    }
  ],
  "output": {
    "file": "window.png"
  }
}
```

### Button Stack

```json
{
  "op": "VSTACK",
  "x": 120,
  "y": 200,
  "spacing": 20,
  "children": [
    {
      "op": "HSTACK",
      "spacing": 10,
      "children": [
        { "op": "RECT", "w": 300, "h": 40, "color": [40, 40, 100, 255] },
        { "op": "TEXT", "x": 10, "y": 10, "color": [255, 255, 255, 255], "value": "BUTTON A" }
      ]
    },
    {
      "op": "HSTACK",
      "spacing": 10,
      "children": [
        { "op": "RECT", "w": 300, "h": 40, "color": [40, 40, 100, 255] },
        { "op": "TEXT", "x": 10, "y": 10, "color": [255, 255, 255, 255], "value": "BUTTON B" }
      ]
    }
  ]
}
```

## LLM Integration

To enable LLMs to generate PXSCENE layouts, include this in your system prompt:

> You can create UI layouts using PXSCENE JSON format. Available operations:
>
> - **RECT**: `{"op": "RECT", "x": 0, "y": 0, "w": 100, "h": 100, "color": [r, g, b, a]}`
> - **TEXT**: `{"op": "TEXT", "x": 10, "y": 10, "color": [255, 255, 255, 255], "value": "Hello"}`
> - **HSTACK**: Horizontal layout with children: `{"op": "HSTACK", "x": 0, "y": 0, "spacing": 10, "children": [...]}`
> - **VSTACK**: Vertical layout with children: `{"op": "VSTACK", "x": 0, "y": 0, "spacing": 10, "children": [...]}`
>
> Layouts automatically position children. Save as `scene_name.json` and run with `python pxscene_run.py scene_name.json`.

## Technical Details

### GPU Rendering

- Uses WebGPU (wgpu-py) with frozen WGSL shader
- Layers composited via alpha blending (back-to-front by z-order)
- Single-pass fragment shader reads all layer buffers
- Output: RGBA8 texture → PNG via PIL

### Text Rendering

- 8×8 bitmap font stored as binary patterns (0b00111100 format)
- Currently includes: A-Z, 0-9, space
- Rendered directly into layer buffers (no font textures)
- Spacing between characters: 1px (configurable in renderer)

### Performance

- Compilation: ~1ms per 100 commands
- Rendering: ~10-50ms depending on layer count and resolution
- GPU compositing: constant time regardless of primitive count (shaders handle it)

## Extending PXSCENE

### Adding New Primitives

1. Define JSON schema in this spec
2. Add compilation logic in `pxscene_compile.py::compile_command()`
3. Add PXTERM command handler in `pxos_llm_terminal.py`
4. Implement primitive in `pxos_gpu_terminal.py`

### Adding Glyphs

Edit `pxos_text.py::GLYPHS` dictionary:

```python
"!": [
    0b00011000,
    0b00011000,
    0b00011000,
    0b00011000,
    0b00000000,
    0b00011000,
    0b00011000,
    0b00000000,
],
```

## Limitations

- Text is uppercase only (auto-converted)
- Font is fixed 8×8 (no scaling yet)
- No image/texture loading
- No animations (single-frame output)
- No input handling (render-only)

## Future Directions

- Variable-width fonts
- Image/sprite loading
- Animation timeline (keyframes)
- Input event system (mouse/keyboard)
- pxVM integration (bytecode → PXSCENE)
