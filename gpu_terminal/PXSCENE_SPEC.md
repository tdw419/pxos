# PXSCENE Specification

**Current Version: v0.3**

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

## PXSCENE v0.3 - UI Widgets

Version 0.3 adds declarative UI widgets that compile down to PXTERM v1 primitives (RECT, HLINE, VLINE).

These widgets are **structural elements** that make it easier for LLMs to describe common UI patterns.

**Note:** Text rendering is not yet implemented. Widget labels are preserved in comments but not rendered visually.

### LABEL - Label Widget

A placeholder for text display (future: will render bitmap text).

**Schema:**
```json
{
  "op": "LABEL",
  "x": 100,
  "y": 100,
  "w": 150,
  "h": 20,
  "text": "Status: Ready",
  "color": [200, 200, 200, 255]
}
```

**Parameters:**
- `x`, `y` (optional, default 0): Position
- `w` (optional, default 100): Width in pixels
- `h` (optional, default 20): Height in pixels
- `text` (required): Label text (stored in comment for now)
- `color` (optional, default `[200, 200, 200, 255]`): Box color

**Compiles to:**
```
# LABEL: "Status: Ready" at (100, 100)
RECT 100 100 150 20 200 200 200 255
```

**Example:**
```json
{
  "op": "LABEL",
  "x": 50,
  "y": 400,
  "w": 200,
  "h": 25,
  "text": "System Ready",
  "color": [100, 200, 100, 255]
}
```

### BUTTON - Button Widget

A clickable button with border and background.

**Schema:**
```json
{
  "op": "BUTTON",
  "x": 100,
  "y": 100,
  "w": 120,
  "h": 40,
  "text": "OK",
  "bg_color": [80, 80, 80, 255],
  "border_color": [150, 150, 150, 255],
  "border_width": 2
}
```

**Parameters:**
- `x`, `y` (optional, default 0): Position
- `w` (optional, default 120): Width in pixels
- `h` (optional, default 40): Height in pixels
- `text` (required): Button text (stored in comment for now)
- `bg_color` (optional, default `[80, 80, 80, 255]`): Background color
- `border_color` (optional, default `[150, 150, 150, 255]`): Border color
- `border_width` (optional, default 2): Border thickness in pixels

**Compiles to:**
```
# BUTTON: "OK" at (100, 100)
HLINE 100 100 120 150 150 150       # Top border
HLINE 100 139 120 150 150 150       # Bottom border
VLINE 100 100 40 150 150 150        # Left border
VLINE 219 100 40 150 150 150        # Right border
RECT 102 102 116 36 80 80 80        # Background
```

**Example:**
```json
{
  "op": "BUTTON",
  "x": 200,
  "y": 300,
  "w": 150,
  "h": 50,
  "text": "Submit",
  "bg_color": [70, 130, 180, 255],
  "border_color": [100, 160, 210, 255],
  "border_width": 2
}
```

### WINDOW - Window Widget

A framed window with title bar and content area that can contain child elements.

**Schema:**
```json
{
  "op": "WINDOW",
  "x": 200,
  "y": 100,
  "w": 400,
  "h": 300,
  "title": "Settings",
  "title_bar_height": 30,
  "title_bar_color": [70, 130, 180, 255],
  "bg_color": [50, 50, 50, 255],
  "border_color": [100, 100, 100, 255],
  "children": [...]
}
```

**Parameters:**
- `x`, `y` (optional, default 0): Window position
- `w` (optional, default 400): Window width
- `h` (optional, default 300): Window height
- `title` (optional, default "Window"): Title bar text (comment only)
- `title_bar_height` (optional, default 30): Title bar height
- `title_bar_color` (optional, default `[70, 130, 180, 255]`): Title bar color
- `bg_color` (optional, default `[50, 50, 50, 255]`): Content background
- `border_color` (optional, default `[100, 100, 100, 255]`): Border color
- `children` (optional, default []): Child commands in content area

**Compiles to:**
```
# WINDOW: "Settings" at (200, 100) size=400x300
HLINE 200 100 400 100 100 100       # Top border
HLINE 200 399 400 100 100 100       # Bottom border
VLINE 200 100 300 100 100 100       # Left border
VLINE 599 100 300 100 100 100       # Right border
RECT 201 101 398 30 70 130 180      # Title bar
RECT 201 131 398 268 50 50 50       # Content background
# Window content (N children)
# ... child commands with relative positioning
```

**Example with VSTACK children:**
```json
{
  "op": "WINDOW",
  "x": 200,
  "y": 100,
  "w": 400,
  "h": 400,
  "title": "PXOS Control Panel",
  "children": [
    {
      "op": "VSTACK",
      "x": 0,
      "y": 0,
      "spacing": 15,
      "children": [
        {"op": "BUTTON", "w": 200, "h": 40, "text": "Settings"},
        {"op": "BUTTON", "w": 200, "h": 40, "text": "Display"},
        {"op": "BUTTON", "w": 200, "h": 40, "text": "Network"}
      ]
    }
  ]
}
```

**Child Positioning:**
- Children use **relative coordinates** within the window's content area
- Content area has 10px padding from window edges
- Child `x`, `y` are relative to content area top-left
- Layout operations (HSTACK/VSTACK) work inside windows

### Composing Widgets with Layouts

Widgets work seamlessly with v0.2 layout operations:

**Example: Vertical menu with buttons**
```json
{
  "op": "VSTACK",
  "x": 50,
  "y": 50,
  "spacing": 10,
  "children": [
    {"op": "BUTTON", "w": 200, "h": 40, "text": "New"},
    {"op": "BUTTON", "w": 200, "h": 40, "text": "Open"},
    {"op": "BUTTON", "w": 200, "h": 40, "text": "Save"},
    {"op": "LABEL", "w": 200, "h": 20, "text": "Ready", "color": [100, 200, 100]}
  ]
}
```

**Example: Horizontal toolbar**
```json
{
  "op": "HSTACK",
  "x": 10,
  "y": 10,
  "spacing": 5,
  "children": [
    {"op": "BUTTON", "w": 80, "h": 30, "text": "File"},
    {"op": "BUTTON", "w": 80, "h": 30, "text": "Edit"},
    {"op": "BUTTON", "w": 80, "h": 30, "text": "View"}
  ]
}
```

### Widget Compatibility

- ✅ All v0.1 operations still work
- ✅ All v0.2 layout operations still work
- ✅ Widgets can be used inside HSTACK/VSTACK
- ✅ Widgets compile down to PXTERM v1 primitives (RECT, HLINE, VLINE)
- ✅ PXTERM v1 remains frozen (unchanged)
- ✅ Shader remains frozen (unchanged)
- ✅ Text rendering deferred to future update

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

- **v0.3** (2025-11-14) - UI Widgets
  - Added LABEL widget (text placeholder)
  - Added BUTTON widget (border + background)
  - Added WINDOW widget (frame + title bar + children)
  - Widgets compile to PXTERM v1 primitives
  - HSTACK/VSTACK now handle widget children
  - Text rendering deferred to future update
  - Backward compatible with v0.1 and v0.2
  - PXTERM v1 and shader remain frozen

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
