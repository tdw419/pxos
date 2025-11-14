# PXSCENE v0.1 Specification

PXSCENE is a declarative JSON format for describing 2D scenes. It is designed to be a high-level, human-readable, and LLM-friendly language that compiles down to the lower-level PXTERM v1 instruction set.

## General Structure

A PXSCENE document is a JSON object with the following top-level keys:

- `canvas`: (Object) Defines the properties of the rendering surface.
- `layers`: (Array) A list of layer objects, each containing a set of drawing commands.
- `output`: (Object) Specifies the output file for the rendered scene.

## Canvas

The `canvas` object defines the overall rendering area.

- `width`, `height`: (Integer) The dimensions of the canvas in pixels.
- `clear`: (Array) An optional `[r, g, b, a]` color to fill the base layer.

## Layers

The `layers` array defines a stack of rendering surfaces. Layers are drawn in ascending order of their `z` index.

- `name`: (String) A unique identifier for the layer.
- `z`: (Integer) The z-index of the layer.
- `commands`: (Array) A list of drawing and layout commands.

## Commands

Commands are objects within a layer's `commands` array. Each command has an `op` field that specifies the operation to perform.

### Primitives

#### RECT
Draws a filled rectangle.

- `op`: "RECT"
- `x`, `y`, `w`, `h`: (Integer) The position and dimensions of the rectangle.
- `color`: (Array) An `[r, g, b, a]` color.

#### PIXEL
Draws a single pixel.

- `op`: "PIXEL"
- `x`, `y`: (Integer) The position of the pixel.
- `color`: (Array) An `[r, g, b, a]` color.

#### HLINE
Draws a horizontal line.

- `op`: "HLINE"
- `x`, `y`: (Integer) The starting position of the line.
- `length`: (Integer) The length of the line.
- `color`: (Array) An `[r, g, b, a]` color.

## PXSCENE v0.2 – Layout Operations

### HSTACK
Stacks children horizontally from left to right.

- `op`: "HSTACK"
- `x`, `y`: (Optional, Integer) The top-left starting position of the stack.
- `spacing`: (Optional, Integer) The gap in pixels between children.
- `children`: (Array) A list of child command objects.

### VSTACK
Stacks children vertically from top to bottom.

- `op`: "VSTACK"
- `x`, `y`: (Optional, Integer) The top-left starting position of the stack.
- `spacing`: (Optional, Integer) The vertical gap in pixels between children.
- `children`: (Array) A list of child command objects.

## PXSCENE v0.3 — Widgets

### LABEL
A simple text label.

- `op`: "LABEL"
- `x`, `y`: (Integer) The position of the label.
- `color`: (Array) The `[r, g, b, a]` color of the text.
- `value`: (String) The text to display.

### BUTTON
A rectangle with a text label.

- `op`: "BUTTON"
- `x`, `y`, `w`, `h`: (Integer) The position and dimensions of the button.
- `bg`: (Array) The `[r, g, b, a]` background color.
- `fg`: (Array) The `[r, g, b, a]` foreground (text) color.
- `label`: (String) The text to display on the button.
- `padding`: (Optional, Integer) The padding between the text and the edge of the button.

### WINDOW
A container with a title bar and content area for child elements.

- `op`: "WINDOW"
- `x`, `y`, `w`, `h`: (Integer) The position and dimensions of the window.
- `title`: (String) The text to display in the title bar.
- `frame_color`: (Array) The `[r, g, b, a]` color of the window's main body.
- `title_bar_color`: (Array) The `[r, g, b, a]` color of the title bar.
- `title_fg`: (Array) The `[r, g, b, a]` color of the title text.
- `children`: (Array) A list of child command objects to be placed in the window's content area.
