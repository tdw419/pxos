# LLM Usage Guide for pxOS GPU Terminal

This guide is for **LLMs** (or humans acting like LLMs) who want to draw graphics using the pxOS terminal.

## TL;DR

You have a **800x600 pixel canvas**. You can draw on it using three simple commands:

- `CLEAR r g b` - Fill screen with color
- `PIXEL x y r g b` - Draw one pixel
- `RECT x y w h r g b` - Draw a rectangle

All colors are 0-255. Origin (0,0) is top-left.

---

## The Mental Model

Think of the pxOS terminal as a **numpy array** that gets uploaded to the screen every frame.

```
VRAM[y][x] = [r, g, b, a]  # 800x600x4 array of uint8
```

When you issue commands like `PIXEL 100 100 255 0 0`, you're really just doing:

```python
VRAM[100][100] = [255, 0, 0, 255]  # Red pixel
```

The frozen shader takes that array and displays it. That's it.

## Why This Matters for LLMs

**You don't need to know WebGPU, WGSL, or shader programming.**

You only need to know:
1. How to write simple drawing commands
2. Basic geometry (rectangles, pixels)
3. RGB color values (0-255)

The GPU complexity is completely hidden. You're just manipulating a Python array.

---

## Command Reference

### CLEAR - Fill Screen

```
CLEAR r g b [a]
```

**What it does:** Sets every pixel to the same color.

**Examples:**
```
CLEAR 0 0 0           # Black
CLEAR 255 255 255     # White
CLEAR 0 0 255         # Blue
CLEAR 128 128 128 200 # Semi-transparent gray
```

**When to use:**
- Start of every drawing (clear the canvas)
- Creating backgrounds
- Resetting the screen

---

### PIXEL - Draw One Pixel

```
PIXEL x y r g b [a]
```

**What it does:** Sets one pixel at (x, y) to the specified color.

**Coordinates:**
- x: 0 (left) to 799 (right)
- y: 0 (top) to 599 (bottom)
- Origin (0, 0) is **top-left**

**Examples:**
```
PIXEL 0 0 255 0 0         # Red pixel at top-left corner
PIXEL 400 300 255 255 255 # White pixel at center
PIXEL 799 599 0 255 0     # Green pixel at bottom-right
```

**When to use:**
- Precise single-pixel drawing
- Plotting data points
- Creating pixel art
- Stars in a night sky
- Individual dots

---

### RECT - Draw Rectangle

```
RECT x y w h r g b [a]
```

**What it does:** Draws a filled rectangle.

**Parameters:**
- x, y: Top-left corner position
- w, h: Width and height in pixels
- r, g, b, a: Color (0-255)

**Examples:**
```
# Red 100x100 square at (50, 50)
RECT 50 50 100 100 255 0 0

# Green horizontal bar across screen
RECT 0 250 800 100 0 255 0

# Semi-transparent white box
RECT 200 200 400 200 255 255 255 128
```

**When to use:**
- Backgrounds
- UI elements
- Shapes (squares, bars, boxes)
- Filled areas
- Most drawing tasks

**Pro tip:** You can draw lines by making rectangles 1 pixel tall or wide:
```
RECT 0 300 800 1 255 255 255  # Horizontal line
RECT 400 0 1 600 255 255 255  # Vertical line
```

---

## Common Patterns

### Pattern 1: Clear and Draw

Always start with CLEAR:

```
CLEAR 0 0 0                    # Black background
RECT 100 100 200 150 255 0 0  # Red rectangle
```

### Pattern 2: Layering

Draw from back to front:

```
CLEAR 135 206 235              # Sky blue background
RECT 0 400 800 200 34 139 34  # Grass (bottom)
RECT 300 250 200 150 139 69 19 # House (on top of grass)
RECT 380 300 40 50 139 0 0    # Door (on top of house)
```

### Pattern 3: Gradients

Use loops of thin rectangles:

```
# Vertical gradient from black to white
# (pseudo-code, actual implementation would be multiple commands)
for y in range(600):
    gray = int(255 * y / 600)
    RECT 0 y 800 1 gray gray gray
```

### Pattern 4: Grids

```
# Draw a grid of lines
for x in range(0, 800, 50):
    RECT x 0 1 600 128 128 128  # Vertical lines

for y in range(0, 600, 50):
    RECT 0 y 800 1 128 128 128  # Horizontal lines
```

---

## Coordinate System

```
(0,0) ────────────────────> (799, 0)
  │                            X
  │
  │         Center
  │         (400, 300)
  │
  │
  ▼
  Y
(0, 599)                   (799, 599)
```

- **X increases** going right
- **Y increases** going down (not up!)
- Screen size: **800 x 600**

---

## Color Guide

### RGB Basics

Colors are specified as three numbers (r, g, b):
- **r** = red (0-255)
- **g** = green (0-255)
- **b** = blue (0-255)

### Common Colors

```
Black:       0   0   0
White:     255 255 255
Red:       255   0   0
Green:       0 255   0
Blue:        0   0 255
Yellow:    255 255   0
Cyan:        0 255 255
Magenta:   255   0 255
Gray:      128 128 128
```

### Transparency (Alpha)

Optional fourth value (a = alpha):
- 255 = fully opaque (default)
- 128 = half transparent
- 0 = fully transparent

```
RECT 100 100 200 150 255 0 0 128  # Semi-transparent red
```

**Note:** Alpha blending is NOT implemented yet. Alpha values are stored but don't currently affect rendering. For now, treat everything as opaque (255).

---

## Example LLM Sessions

### Session 1: User Request

**User:** "Draw a red square in the center of the screen"

**LLM Output:**
```
CLEAR 0 0 0
RECT 350 250 100 100 255 0 0
```

**Explanation:**
- Clear to black
- Center of screen is (400, 300)
- 100x100 square centered there is at (350, 250)

---

### Session 2: Complex Request

**User:** "Draw a simple house with a red roof and blue door"

**LLM Output:**
```
CLEAR 135 206 235
RECT 0 400 800 200 34 139 34
RECT 250 250 300 200 222 184 135
RECT 250 200 300 50 178 34 34
RECT 350 350 100 100 0 0 139
RECT 280 280 60 60 255 255 255
RECT 460 280 60 60 255 255 255
```

**Explanation:**
1. Sky blue background
2. Green grass (bottom)
3. Tan house body
4. Red roof (simplified rectangle)
5. Blue door
6. Two white windows

---

### Session 3: Data Visualization

**User:** "Plot these data points: (100, 200), (200, 150), (300, 180), (400, 100)"

**LLM Output:**
```
CLEAR 255 255 255
RECT 0 580 800 20 200 200 200
RECT 0 0 20 600 200 200 200
PIXEL 100 200 255 0 0
PIXEL 200 150 255 0 0
PIXEL 300 180 255 0 0
PIXEL 400 100 255 0 0
```

**Explanation:**
- White background
- Gray axes at bottom and left
- Red pixels at each data point

---

## How to Use This Terminal

### Option 1: Interactive Mode

Run:
```bash
python pxos_llm_terminal.py
```

Type commands at the `pxos>` prompt:
```
pxos> CLEAR 0 0 0
pxos> RECT 100 100 200 150 255 0 0
```

### Option 2: Script Mode

Create a text file (`drawing.txt`):
```
CLEAR 0 0 0
RECT 100 100 200 150 255 0 0
PIXEL 400 300 255 255 255
```

Run:
```bash
python pxos_llm_terminal.py drawing.txt
```

### Option 3: Python API

```python
from pxos_llm_terminal import batch_mode

batch_mode([
    "CLEAR 0 0 0",
    "RECT 100 100 200 150 255 0 0",
])
```

---

## Debugging

If something looks wrong:

1. **Check your coordinates**
   - Are they in bounds? (x: 0-799, y: 0-599)
   - Remember: (0,0) is **top-left**, not bottom-left

2. **Check your colors**
   - Are they 0-255?
   - Did you swap r/g/b?

3. **Check the order**
   - Did you CLEAR first?
   - Are you drawing back-to-front?

4. **Print the commands**
   - The terminal echoes each command
   - Look for error messages

**The shader is frozen → it's never the problem.**

If you see pixels on screen, the issue is in your command logic, not the GPU.

---

## Limitations (v0.1)

**Not yet implemented:**
- Alpha blending (transparency ignored)
- Text rendering (no fonts yet)
- Lines (use thin RECTs instead)
- Circles (use RECTs to approximate)
- Image loading
- Animation (you can draw frames, but no timer yet)

**Coming soon:**
- Bitmap text rendering
- LINE command
- CIRCLE command
- BLIT (copy regions)
- Timer/animation support

---

## Tips for LLMs

### 1. Always CLEAR first

```
CLEAR 0 0 0  # Good start
```

Without this, you'll have leftover pixels from previous drawings.

### 2. Use RECT for almost everything

PIXEL is slow if you need to fill large areas. Use RECT instead:

```
# Slow (1000 commands):
for x in range(1000):
    PIXEL x 300 255 0 0

# Fast (1 command):
RECT 0 300 1000 1 255 0 0
```

### 3. Think in layers

Draw from back (background) to front (foreground):

```
CLEAR ...       # Sky
RECT ...        # Ground
RECT ...        # Buildings
RECT ...        # Windows
PIXEL ...       # Stars
```

### 4. Use comments in scripts

```
# This is a comment
CLEAR 0 0 0

# Draw background
RECT 0 400 800 200 34 139 34
```

### 5. Validate before sending

Check that:
- All numbers are integers
- Colors are 0-255
- Coordinates are in bounds
- Commands are uppercase

---

## Philosophy

**The pxOS terminal is a "dumb canvas."**

It doesn't know what you're drawing. It doesn't have built-in shapes or fonts or sprites.

It just:
1. Stores pixels in a buffer
2. Shows them on screen

This makes it:
- **Simple** - only 3 commands to learn
- **Stable** - the shader never changes
- **Debuggable** - you can see exactly what you drew
- **LLM-friendly** - no GPU knowledge needed

You're not fighting a complex API. You're just setting pixels.

That's the power of the **Frozen Shader Bus**.

---

**Have fun drawing!**
