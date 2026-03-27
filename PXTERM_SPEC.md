# PXTERM v1 Specification

PXTERM is a line-oriented, text-based instruction set for the pxOS GPU terminal. It is designed to be a simple, stable machine code target for LLMs to generate graphics.

## General Rules

- **One instruction per line.**
- **Comments:** Lines starting with `#` are comments and are ignored.
- **Arguments:** Arguments are space-separated.
- **Numeric Arguments:** All numeric arguments are integers (e.g., coordinates, sizes, colors).
- **Color Format:** Colors are specified as `r g b [a]`, where each component is an integer from 0 to 255. The alpha component `a` is optional and defaults to 255 (fully opaque) if omitted.
- **Coordinate System:** The canvas origin `(0, 0)` is the top-left corner. The x-axis increases to the right, and the y-axis increases downwards.

## Instructions

### Introspection

#### INFO
Prints the canvas dimensions and the name of the currently active layer to standard output.

**Syntax:**
`INFO`

#### LAYERS
Lists all existing layers, including their name, z-index, and visibility status, to standard output.

**Syntax:**
`LAYERS`

### Layer Management

#### LAYER NEW
Creates a new layer.

**Syntax:**
`LAYER NEW <name> [z_index]`
- `<name>`: A string identifier for the new layer.
- `[z_index]`: (Optional) An integer specifying the layer's depth. Higher z-indices are rendered on top. Defaults to 0.

#### LAYER USE
Sets the active layer for all subsequent drawing commands.

**Syntax:**
`LAYER USE <name>`
- `<name>`: The name of the layer to make active.

### Drawing Primitives

All drawing commands operate on the currently active layer.

#### CLEAR
Fills the entire active layer with a solid color.

**Syntax:**
`CLEAR <r> <g> <b> [a]`

#### PIXEL
Sets the color of a single pixel.

**Syntax:**
`PIXEL <x> <y> <r> <g> <b> [a]`

#### HLINE
Draws a horizontal line.

**Syntax:**
`HLINE <x> <y> <length> <r> <g> <b> [a]`
- `<x>`, `<y>`: The starting coordinates of the line (leftmost point).
- `<length>`: The width of the line in pixels.

#### RECT
Draws a filled rectangle.

**Syntax:**
`RECT <x> <y> <w> <h> <r> <g> <b> [a]`
- `<x>`, `<y>`: The coordinates of the top-left corner.
- `<w>`, `<h>`: The width and height of the rectangle.

#### TEXT
Draws a single line of text using the built-in 8x8 bitmap font.

**Syntax:**
`TEXT <x> <y> <r> <g> <b> [a] <message...>`
- `<x>`, `<y>`: The coordinates of the top-left corner of the first character.
- `<message...>`: The string to be rendered. It must be the last argument(s) on the line.

## Example Script

The following script demonstrates creating layers and drawing shapes.

```text
# scene1.pxterm
# A simple test program for PXTERM v1.

# Print initial state
INFO
LAYERS

# Create and prepare a background layer
LAYER NEW background -1
LAYER USE background
CLEAR 0 0 50

# Create a scene layer on top
LAYER NEW scene 10
LAYER USE scene

# Draw a semi-transparent red box
RECT 100 100 200 150 255 0 0 128

# Draw some text inside the box
TEXT 110 120 255 255 255 Hello PXTERM

# Print final state
LAYERS
```
