# PXSCENE v0.1 Specification

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

## Future Extensions

Potential future operations (v0.2+):

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

- **v0.1** (2025-11-14) - Initial specification
  - Basic operations: CLEAR, PIXEL, RECT, HLINE, VLINE
  - Layer-based composition
  - Canvas and output directives
  - Compiler validation
