# PXSCENE Specification

**Current Version: v0.2**

**PXSCENE** is a JSON-based scene description language designed for LLMs.

It provides a structured, human-readable format for describing graphics that compiles down to PXTERM v1 machine code.

## Philosophy

- **LLM-native**: JSON is easy for LLMs to generate and validate
- **Structured**: Explicit hierarchy (canvas → layers → commands)
- **Debuggable**: Human-readable, easy to inspect and modify
- **Compiled**: Compiles to PXTERM v1 (machine code layer)

## Architecture

```
┌─────────────────────────────────────────┐
│ LLM / High-Level Program                │
│   "Draw a red square on blue background"│
├─────────────────────────────────────────┤
│ PXSCENE v0.1 (JSON)                     │
│   Structured scene description          │
├─────────────────────────────────────────┤
│ pxscene_compile.py                      │
│   Compiler: JSON → PXTERM               │
├─────────────────────────────────────────┤
│ PXTERM v1 (Text Commands)               │
│   Machine code for pixel terminal       │
├─────────────────────────────────────────┤
│ pxos_llm_terminal.py                    │
│   Execute PXTERM instructions           │
├─────────────────────────────────────────┤
│ GPU (Frozen Shader)                     │
│   Display pixels                        │
└─────────────────────────────────────────┘
```

## JSON Structure

### Top-Level Schema

```json
{
  "canvas": {
    "width": 800,
    "height": 600,
    "clear": [r, g, b, a]
  },
  "layers": [
    {
      "name": "layer_name",
      "z": 0,
      "opacity": 255,
      "commands": [...]
    }
  ],
  "output": {
    "file": "output.png"
  }
}
```

### Canvas Object (Optional)

Defines global canvas properties.

```json
"canvas": {
  "width": 800,        // Canvas width (future feature)
  "height": 600,       // Canvas height (future feature)
  "clear": [0, 0, 0]   // Global background clear (r, g, b, [a])
}
```

### Layer Object

Represents a compositable layer.

```json
{
  "name": "background",  // Required: Layer identifier
  "z": 0,                // Required: Z-index for compositing order
  "opacity": 255,        // Optional: Layer opacity 0-255 (default: 255)
  "commands": [...]      // Required: Array of drawing commands
}
```

**Notes:**
- Layers are composited in z-index order (low to high)
- Higher z-index layers appear on top
- Opacity affects entire layer during compositing

### Command Objects

#### CLEAR

Fill layer with solid color.

```json
{
  "op": "CLEAR",
  "color": [r, g, b, a]
}
```

Example:
```json
{"op": "CLEAR", "color": [0, 0, 0, 255]}
```

#### PIXEL

Draw a single pixel.

```json
{
  "op": "PIXEL",
  "x": 100,
  "y": 100,
  "color": [r, g, b, a]
}
```

Example:
```json
{"op": "PIXEL", "x": 400, "y": 300, "color": [255, 0, 0]}
```

#### RECT

Draw a filled rectangle.

```json
{
  "op": "RECT",
  "x": 100,
  "y": 100,
  "w": 200,
  "h": 150,
  "color": [r, g, b, a]
}
```

Example:
```json
{"op": "RECT", "x": 100, "y": 100, "w": 200, "h": 150, "color": [255, 0, 0, 128]}
```

#### HLINE

Draw a horizontal line.

```json
{
  "op": "HLINE",
  "x": 0,
  "y": 100,
  "length": 800,
  "color": [r, g, b, a]
}
```

Example:
```json
{"op": "HLINE", "x": 0, "y": 300, "length": 800, "color": [255, 255, 255]}
```

#### VLINE

Draw a vertical line.

```json
{
  "op": "VLINE",
  "x": 400,
  "y": 0,
  "length": 600,
  "color": [r, g, b, a]
}
```

Example:
```json
{"op": "VLINE", "x": 400, "y": 0, "length": 600, "color": [0, 0, 255]}
```

#### COMMENT

Add a structured comment (not rendered).

```json
{
  "op": "COMMENT",
  "text": "This is a comment"
}
```

### Output Object (Optional)

Specifies output file for saving.

```json
"output": {
  "file": "scene.png"
}
```

Compiles to: `SAVE scene.png`

## Color Format

Colors are arrays of integers 0-255:

- `[r, g, b]` - RGB, alpha defaults to 255
- `[r, g, b, a]` - RGBA with explicit alpha

Examples:
- `[255, 0, 0]` - Opaque red
- `[0, 255, 0, 128]` - Semi-transparent green
- `[0, 0, 255, 255]` - Opaque blue

## Complete Example

```json
{
  "canvas": {
    "clear": [0, 0, 32]
  },
  "layers": [
    {
      "name": "background",
      "z": 0,
      "commands": [
        {
          "op": "RECT",
          "x": 0,
          "y": 0,
          "w": 800,
          "h": 600,
          "color": [0, 0, 50, 255]
        },
        {
          "op": "RECT",
          "x": 10,
          "y": 10,
          "w": 780,
          "h": 580,
          "color": [0, 0, 0, 255]
        }
      ]
    },
    {
      "name": "content",
      "z": 10,
      "commands": [
        {
          "op": "COMMENT",
          "text": "Draw a red box"
        },
        {
          "op": "RECT",
          "x": 100,
          "y": 100,
          "w": 200,
          "h": 150,
          "color": [255, 0, 0, 128]
        },
        {
          "op": "HLINE",
          "x": 0,
          "y": 300,
          "length": 800,
          "color": [255, 255, 255, 255]
        }
      ]
    }
  ],
  "output": {
    "file": "scene1.png"
  }
}
```

## Compilation

Compile PXSCENE JSON to PXTERM:

```bash
python pxscene_compile.py scene.json scene.pxterm
```

Execute PXTERM:

```bash
python pxos_llm_terminal.py scene.pxterm
```

## PXSCENE v0.2 - Layout Operations

Version 0.2 adds higher-level layout operators that exist **only in the JSON scene description**.
They are compiled down into PXTERM v1 instructions by `pxscene_compile.py`.

### HSTACK - Horizontal Stack

Stack children horizontally from left to right with automatic x-positioning.

**Schema:**
```json
{
  "op": "HSTACK",
  "x": 100,
  "y": 250,
  "spacing": 10,
  "children": [
    {"op": "RECT", "w": 150, "h": 100, "color": [255, 0, 0, 180]},
    {"op": "RECT", "w": 150, "h": 100, "color": [0, 255, 0, 180]},
    {"op": "RECT", "w": 150, "h": 100, "color": [0, 0, 255, 180]}
  ]
}
```

**Parameters:**
- `x`, `y` (optional, default 0): Top-left starting position of the stack
- `spacing` (optional, default 0): Gap in pixels between children
- `children` (required): Array of child commands to stack

**Behavior:**
- For `RECT` children: `w` and `h` are required
- Each child is placed at the current cursor position
- Cursor advances by `child_width + spacing` after each child
- Children's local `x`, `y` default to 0

**Example:**
```json
{
  "op": "HSTACK",
  "x": 125,
  "y": 240,
  "spacing": 40,
  "children": [
    {"op": "RECT", "w": 150, "h": 120, "color": [255, 0, 0, 180]},
    {"op": "RECT", "w": 150, "h": 120, "color": [0, 255, 0, 180]},
    {"op": "RECT", "w": 150, "h": 120, "color": [0, 0, 255, 180]}
  ]
}
```

Compiles to:
```
RECT 125 240 150 120 255 0 0 180
RECT 315 240 150 120 0 255 0 180   # 125 + 150 + 40
RECT 505 240 150 120 0 0 255 180   # 315 + 150 + 40
```

### VSTACK - Vertical Stack

Stack children vertically from top to bottom with automatic y-positioning.

**Schema:**
```json
{
  "op": "VSTACK",
  "x": 50,
  "y": 50,
  "spacing": 5,
  "children": [
    {"op": "RECT", "w": 200, "h": 40, "color": [60, 60, 60, 255]},
    {"op": "RECT", "w": 200, "h": 40, "color": [100, 100, 100, 255]},
    {"op": "RECT", "w": 200, "h": 40, "color": [140, 140, 140, 255]}
  ]
}
```

**Parameters:**
- `x`, `y` (optional, default 0): Top-left starting position
- `spacing` (optional, default 0): Vertical gap in pixels
- `children` (required): Array of child commands to stack

**Behavior:**
- For `RECT` children: `w` and `h` are required
- Each child is placed at `(x, current_y)`
- Cursor advances by `child_height + spacing` after each child
- Children's local `x`, `y` default to 0

**Example:**
```json
{
  "op": "VSTACK",
  "x": 50,
  "y": 50,
  "spacing": 10,
  "children": [
    {"op": "RECT", "w": 200, "h": 50, "color": [60, 60, 60, 255]},
    {"op": "RECT", "w": 200, "h": 50, "color": [80, 80, 80, 255]},
    {"op": "RECT", "w": 200, "h": 50, "color": [100, 100, 100, 255]}
  ]
}
```

Compiles to:
```
RECT 50 50 200 50 60 60 60 255
RECT 50 110 200 50 80 80 80 255    # 50 + 50 + 10
RECT 50 170 200 50 100 100 100 255 # 110 + 50 + 10
```

### Why Layout Operations?

Layout operations eliminate coordinate calculation for LLMs:

**Before (v0.1 - manual coordinates):**
```json
{"op": "RECT", "x": 125, "y": 240, "w": 150, "h": 120, "color": [255, 0, 0]},
{"op": "RECT", "x": 315, "y": 240, "w": 150, "h": 120, "color": [0, 255, 0]},
{"op": "RECT", "x": 505, "y": 240, "w": 150, "h": 120, "color": [0, 0, 255]}
```

**After (v0.2 - layout-driven):**
```json
{
  "op": "HSTACK",
  "x": 125,
  "y": 240,
  "spacing": 40,
  "children": [
    {"op": "RECT", "w": 150, "h": 120, "color": [255, 0, 0]},
    {"op": "RECT", "w": 150, "h": 120, "color": [0, 255, 0]},
    {"op": "RECT", "w": 150, "h": 120, "color": [0, 0, 255]}
  ]
}
```

LLMs describe **structure** instead of calculating **positions**.

### Compatibility

- ✅ All v0.1 operations still work
- ✅ Can mix v0.1 and v0.2 operations
- ✅ PXTERM v1 remains frozen (unchanged)
- ✅ Shader remains frozen (unchanged)
- ✅ Only the compiler adds layout engine

## Future Extensions

Potential future operations (v0.3+):

- `GRID` - 2D grid layout
- `CENTER` - Center children in a container
- `LINE` - Arbitrary angle lines (Bresenham)
- `CIRCLE` - Circle drawing
- `SPRITE` - Sprite/image blitting
- `TEXT` - Text rendering
- `GRADIENT` - Gradient fills
- `BLEND` - Custom blend modes
- `FILTER` - Image filters/effects

## Design Principles

1. **Keep JSON simple** - Easy for LLMs to generate correctly
2. **Explicit over implicit** - No magic defaults
3. **Validate early** - Catch errors at compile time
4. **Compile to stable target** - PXTERM v1 is frozen
5. **Extend carefully** - Add ops only when needed

## Version History

- **v0.2** (2025-11-14) - Layout operations
  - Added HSTACK (horizontal stack with automatic x-positioning)
  - Added VSTACK (vertical stack with automatic y-positioning)
  - Layout engine in compiler
  - Eliminates coordinate calculation for LLMs
  - Backward compatible with v0.1
  - PXTERM v1 and shader remain frozen

- **v0.1** (2025-11-14) - Initial specification
  - Basic operations: CLEAR, PIXEL, RECT, HLINE, VLINE
  - Layer-based composition
  - Canvas and output directives
  - Compiler validation
