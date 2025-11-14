# PXTERM v1 Specification

**PXTERM** (Pixel Terminal Machine Code) is a line-oriented, text-based instruction set for the pxOS GPU terminal.

It is the "machine code" layer - stable, frozen, and simple.

## Philosophy

- **Frozen ISA**: v1 is frozen and will not change
- **Text-based**: Human-readable, easy to generate and debug
- **Deterministic**: No ambiguity, clear semantics
- **LLM-friendly**: Simple syntax, explicit arguments
- **Composable**: Instructions combine to create complex graphics

## Status

🔒 **FROZEN** - PXTERM v1 is the stable machine code layer.

The shader underneath is frozen. The instructions are frozen. Only higher-level languages (like PXSCENE) will evolve.

## General Rules

1. **One instruction per line**
2. **Case-insensitive** commands (CLEAR = clear = Clear)
3. **Space-separated** arguments
4. **Comments** start with `#` and are ignored
5. **Empty lines** are ignored
6. **Numeric arguments** are integers unless specified
7. **Colors** are RGBA, each 0-255
8. **Alpha** is optional, defaults to 255
9. **Coordinates** are (x, y) with origin at top-left (0, 0)

## Architecture

```
┌────────────────────────────────────┐
│ PXTERM v1 (This Spec)              │
│   Text-based machine code          │
├────────────────────────────────────┤
│ pxos_llm_terminal.py               │
│   Parser and executor              │
├────────────────────────────────────┤
│ pxos_gpu_terminal.py               │
│   Layer manager + compositor       │
├────────────────────────────────────┤
│ CPU VRAM (numpy)                   │
│   Pixel buffers per layer          │
├────────────────────────────────────┤
│ Frozen Shader (WGSL)               │
│   Display texture on screen        │
└────────────────────────────────────┘
```

## Instruction Reference

### Introspection

#### INFO

Print terminal information.

**Syntax:**
```
INFO
```

**Output:**
- Canvas dimensions
- Current layer name
- Total number of layers

**Example:**
```
INFO
```

---

#### LAYERS

List all layers with their properties.

**Syntax:**
```
LAYERS
```

**Output:**
- Layer name
- Z-index
- Visibility
- Opacity

**Example:**
```
LAYERS
```

---

#### HELP

Show help text.

**Syntax:**
```
HELP
```

**Example:**
```
HELP
```

---

### Layer Management

#### LAYER NEW

Create a new layer.

**Syntax:**
```
LAYER NEW <name> <z_index>
```

**Arguments:**
- `name` (string): Layer identifier
- `z_index` (int): Z-order for compositing (lower = back, higher = front)

**Example:**
```
LAYER NEW background 0
LAYER NEW ui 10
LAYER NEW overlay 20
```

---

#### LAYER USE

Switch to a layer for drawing operations.

**Syntax:**
```
LAYER USE <name>
```

**Arguments:**
- `name` (string): Layer to activate

**Example:**
```
LAYER USE background
CLEAR 0 0 0
LAYER USE ui
RECT 100 100 200 150 255 0 0
```

---

#### LAYER DELETE

Delete a layer (cannot delete default layer).

**Syntax:**
```
LAYER DELETE <name>
```

**Arguments:**
- `name` (string): Layer to delete

**Example:**
```
LAYER DELETE temp_layer
```

---

### Drawing Commands

All drawing commands operate on the **current layer**.

#### CLEAR

Fill the current layer with a solid color.

**Syntax:**
```
CLEAR <r> <g> <b> [a]
```

**Arguments:**
- `r` (0-255): Red component
- `g` (0-255): Green component
- `b` (0-255): Blue component
- `a` (0-255): Alpha component (optional, default: 255)

**Examples:**
```
CLEAR 0 0 0              # Black, opaque
CLEAR 255 0 0 128        # Red, semi-transparent
CLEAR 135 206 235        # Sky blue
```

---

#### PIXEL

Draw a single pixel.

**Syntax:**
```
PIXEL <x> <y> <r> <g> <b> [a]
```

**Arguments:**
- `x` (int): X coordinate
- `y` (int): Y coordinate
- `r` (0-255): Red component
- `g` (0-255): Green component
- `b` (0-255): Blue component
- `a` (0-255): Alpha component (optional, default: 255)

**Examples:**
```
PIXEL 400 300 255 0 0          # Red pixel at center
PIXEL 10 10 0 255 0 128        # Semi-transparent green
```

---

#### RECT

Draw a filled rectangle.

**Syntax:**
```
RECT <x> <y> <w> <h> <r> <g> <b> [a]
```

**Arguments:**
- `x` (int): Top-left X coordinate
- `y` (int): Top-left Y coordinate
- `w` (int): Width in pixels
- `h` (int): Height in pixels
- `r` (0-255): Red component
- `g` (0-255): Green component
- `b` (0-255): Blue component
- `a` (0-255): Alpha component (optional, default: 255)

**Examples:**
```
RECT 100 100 200 150 255 0 0        # Red rectangle
RECT 0 0 800 600 0 0 0              # Fill screen black
RECT 50 50 100 100 0 255 0 128      # Semi-transparent green box
```

---

#### HLINE

Draw a horizontal line.

**Syntax:**
```
HLINE <x> <y> <length> <r> <g> <b> [a]
```

**Arguments:**
- `x` (int): Starting X coordinate
- `y` (int): Y coordinate (row)
- `length` (int): Length in pixels
- `r` (0-255): Red component
- `g` (0-255): Green component
- `b` (0-255): Blue component
- `a` (0-255): Alpha component (optional, default: 255)

**Examples:**
```
HLINE 0 300 800 255 255 255     # White line across screen
HLINE 100 200 200 255 0 0       # Red line, 200px long
```

---

#### VLINE

Draw a vertical line.

**Syntax:**
```
VLINE <x> <y> <length> <r> <g> <b> [a]
```

**Arguments:**
- `x` (int): X coordinate (column)
- `y` (int): Starting Y coordinate
- `length` (int): Length in pixels
- `r` (0-255): Red component
- `g` (0-255): Green component
- `b` (0-255): Blue component
- `a` (0-255): Alpha component (optional, default: 255)

**Examples:**
```
VLINE 400 0 600 0 0 255         # Blue line down center
VLINE 100 100 200 255 0 0       # Red line, 200px long
```

---

### Utility Commands

#### SAVE

Save the current composed frame to a PNG file.

**Syntax:**
```
SAVE <path>
```

**Arguments:**
- `path` (string): Output file path

**Examples:**
```
SAVE output.png
SAVE /tmp/frame_001.png
```

---

#### QUIT

Exit the terminal (interactive mode only).

**Syntax:**
```
QUIT
```

---

## Complete Example

```text
# PXTERM v1 Example Program
# Draw a layered scene with frame and content

INFO
LAYERS

# Create background layer
LAYER NEW background 0
LAYER USE background
CLEAR 0 0 50

# Draw border frame
RECT 10 10 780 580 0 0 0

# Create content layer
LAYER NEW content 10
LAYER USE content

# Draw a red box
RECT 100 100 200 150 255 0 0 128

# Draw a white crosshair
HLINE 0 300 800 255 255 255
VLINE 400 0 600 255 255 255

# Save result
SAVE example_output.png

INFO
LAYERS
```

## Coordinate System

```
(0,0) ──────────────→ X (799)
  │
  │
  │
  │
  ↓
  Y
(599)

Canvas: 800x600 (default)
Origin: Top-left
X: 0 to width-1
Y: 0 to height-1
```

## Error Handling

PXTERM commands that fail print an error but do not stop execution:

```
[CMD] PIXEL out of bounds: (1000, 1000)
[LAYER] Layer 'nonexistent' does not exist
```

## Execution Modes

### Script Mode

Run a PXTERM file:

```bash
python pxos_llm_terminal.py program.pxterm
```

### Interactive Mode (Limited)

Currently, interactive REPL is limited due to GUI event loop. Use script mode.

## Compilation Targets

PXTERM v1 can be generated from:

1. **PXSCENE v0.1** (JSON) - High-level scene description
2. **pxVM** (Future) - Virtual machine bytecode
3. **Direct LLM generation** - LLMs can emit PXTERM directly
4. **Hand-written** - Human-readable for debugging

## Design Principles

1. **Stability** - v1 is frozen, no breaking changes
2. **Simplicity** - Easy to parse, easy to generate
3. **Inspectability** - Plain text, human-readable
4. **Determinism** - Same input = same output
5. **Composability** - Complex graphics from simple ops

## Future (v2+)

Possible future extensions (would be separate version):

- `LINE x1 y1 x2 y2 r g b [a]` - Arbitrary lines
- `CIRCLE x y radius r g b [a]` - Circle drawing
- `TEXT x y "string" r g b [a]` - Text rendering
- `BLIT src_layer x y w h dst_x dst_y` - Layer blitting
- Variables and arithmetic
- Control flow (if/loop)

But v1 is **frozen** and will remain stable.

## Version History

- **v1.0** (2025-11-14) - Initial frozen specification
  - Introspection: INFO, LAYERS, HELP
  - Layer management: NEW, USE, DELETE
  - Drawing: CLEAR, PIXEL, RECT, HLINE, VLINE
  - Utility: SAVE, QUIT
  - Status: 🔒 FROZEN
