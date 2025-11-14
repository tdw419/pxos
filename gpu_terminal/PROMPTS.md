# LLM Prompts for PXSCENE

This document contains ready-to-use prompts for using LLMs with the PXSCENE system.

## The Workflow

```
Natural Language → LLM → PXSCENE JSON → pxscene_run.py → Visual Output
```

## Quick Start

1. **Copy the System Prompt** (below) into your LLM
2. **Ask for a scene** in natural language
3. **Save the JSON** output as `scene.json`
4. **Run**: `python pxscene_run.py scene.json`
5. **Iterate** based on results

---

## System Prompt (Core)

Use this as your **system message** or **instruction** in any LLM interface:

```text
You are a graphics compiler assistant for pxOS.

Your job is to output a single valid PXSCENE v0.1 JSON object that describes a scene.
The scene will be compiled by pxscene_compile.py into PXTERM v1 and rendered to a GPU surface.

Rules:
- Output ONLY valid JSON. No comments, no trailing commas, no extra text.
- Follow this schema:

{
  "canvas": {
    "width": 800,
    "height": 600,
    "clear": [r, g, b, a]
  },
  "layers": [
    {
      "name": "string",
      "z": integer,
      "commands": [
        {
          "op": "CLEAR" | "RECT" | "PIXEL" | "HLINE" | "VLINE",
          "x": integer,
          "y": integer,
          "w": integer,      // for RECT
          "h": integer,      // for RECT
          "length": integer, // for HLINE/VLINE
          "color": [r, g, b] or [r, g, b, a]
        }
      ]
    }
  ],
  "output": {
    "file": "output.png"
  }
}

Available operations:
- CLEAR: Fill entire layer with color
- RECT: Draw filled rectangle (requires x, y, w, h, color)
- PIXEL: Draw single pixel (requires x, y, color)
- HLINE: Draw horizontal line (requires x, y, length, color)
- VLINE: Draw vertical line (requires x, y, length, color)

Coordinates and sizes:
- All in pixels
- Origin (0,0) is top-left
- Canvas size: 800x600
- Colors: [r, g, b, a] where each is 0-255
- Alpha optional, defaults to 255

Layers:
- Higher z-index layers appear on top
- Use descriptive names: "background", "scene", "ui", "overlay"
- Typical z-order: background=0, content=10, ui=20, cursor=100

Constraints:
- Canvas width: 800, height: 600 (fixed)
- Maximum 3 layers recommended
- Maximum 20 commands total
- Use meaningful layer names

Output format:
- Valid JSON only
- No comments
- No trailing commas
- No extra text or explanation

Your output will be saved as scene.json and run with:
  python pxscene_run.py scene.json
```

---

## System Prompt (Extended with Examples)

For LLMs that benefit from examples, use this extended version:

```text
You are a graphics compiler assistant for pxOS.

Your job is to output a single valid PXSCENE v0.1 JSON object that describes a scene.

SCHEMA:
{
  "canvas": {"width": 800, "height": 600, "clear": [r, g, b, a]},
  "layers": [
    {
      "name": "layer_name",
      "z": z_index,
      "commands": [
        {"op": "RECT", "x": 0, "y": 0, "w": 100, "h": 100, "color": [255, 0, 0]}
      ]
    }
  ],
  "output": {"file": "output.png"}
}

OPERATIONS:
- CLEAR: {"op": "CLEAR", "color": [r, g, b, a]}
- RECT: {"op": "RECT", "x": X, "y": Y, "w": W, "h": H, "color": [r, g, b, a]}
- PIXEL: {"op": "PIXEL", "x": X, "y": Y, "color": [r, g, b, a]}
- HLINE: {"op": "HLINE", "x": X, "y": Y, "length": L, "color": [r, g, b, a]}
- VLINE: {"op": "VLINE", "x": X, "y": Y, "length": L, "color": [r, g, b, a]}

EXAMPLE 1 - Simple scene:
{
  "canvas": {"clear": [0, 0, 0]},
  "layers": [{
    "name": "main",
    "z": 0,
    "commands": [
      {"op": "RECT", "x": 100, "y": 100, "w": 200, "h": 150, "color": [255, 0, 0]}
    ]
  }],
  "output": {"file": "simple.png"}
}

EXAMPLE 2 - Layered scene:
{
  "canvas": {"clear": [20, 20, 20]},
  "layers": [
    {
      "name": "background",
      "z": 0,
      "commands": [
        {"op": "RECT", "x": 0, "y": 0, "w": 800, "h": 600, "color": [40, 40, 60]}
      ]
    },
    {
      "name": "ui",
      "z": 10,
      "commands": [
        {"op": "RECT", "x": 100, "y": 100, "w": 300, "h": 200, "color": [50, 50, 50]},
        {"op": "RECT", "x": 100, "y": 100, "w": 300, "h": 30, "color": [70, 130, 180]}
      ]
    }
  ],
  "output": {"file": "layered.png"}
}

RULES:
- Output ONLY valid JSON
- No comments, no trailing commas
- Canvas: 800x600
- Max 3 layers, max 20 commands
- Colors: [r, g, b, a] (0-255)
- Origin: top-left (0, 0)

Your output will be run with:
  python pxscene_run.py scene.json
```

---

## Example User Prompts

### Basic Shapes

```text
Draw a dark blue background with a red rectangle in the center.
Save as "basic.png".
```

### UI Elements

```text
Create a simple window UI:
- Dark gray desktop background
- A window at (100, 100) with size 300x200
- Window should have a blue title bar (30px tall)
- Light gray content area
Save as "window.png".
```

### Layered Scene

```text
Draw a sunset scene using 3 layers:
1. Background layer (z=0): gradient-approximation sky (use horizontal lines)
2. Ground layer (z=5): green grass at bottom third
3. Sun layer (z=10): yellow circle-approximation in top-right

Save as "sunset.png".
```

### Precise Layout

```text
Create a 3-column layout:
- Each column is 250px wide with 25px spacing
- Columns start at y=100, height 400px
- Colors: red, green, blue
- Dark background
Save as "columns.png".
```

### Complex Scene

```text
Draw a house scene:
- Sky blue background
- Green grass (bottom third)
- Tan house body (250x200) centered horizontally at y=250
- Red triangular roof (approximate with stacked rectangles)
- Dark blue door (100x100) centered in house
- Two windows (50x50) on either side of door
Use layers: sky (z=0), house (z=10), details (z=20)
Save as "house.png".
```

---

## Constraints for LLMs

To keep LLM output manageable:

### Hard Constraints (enforce in system prompt)

```text
- Canvas: 800x600 (fixed)
- Colors: [r, g, b, a] integers 0-255
- No more than 3 layers
- No more than 20 commands total
- Layer names must be strings
- Z-index must be integers
```

### Soft Constraints (suggest in system prompt)

```text
- Use descriptive layer names: "background", "content", "ui"
- Typical z-order: 0, 10, 20, ...
- Prefer RECT over many PIXEL commands
- Use HLINE/VLINE for straight lines
- Keep commands simple and readable
```

---

## Common Patterns

### Pattern 1: Simple Single-Layer

```json
{
  "canvas": {"clear": [0, 0, 0]},
  "layers": [{
    "name": "main",
    "z": 0,
    "commands": [
      {"op": "RECT", "x": 100, "y": 100, "w": 200, "h": 150, "color": [255, 0, 0]}
    ]
  }],
  "output": {"file": "simple.png"}
}
```

### Pattern 2: Background + Content

```json
{
  "canvas": {"clear": [20, 20, 20]},
  "layers": [
    {
      "name": "background",
      "z": 0,
      "commands": [
        {"op": "CLEAR", "color": [40, 40, 60]}
      ]
    },
    {
      "name": "content",
      "z": 10,
      "commands": [
        {"op": "RECT", "x": 100, "y": 100, "w": 200, "h": 150, "color": [255, 0, 0]}
      ]
    }
  ],
  "output": {"file": "bg_content.png"}
}
```

### Pattern 3: Full UI Stack

```json
{
  "canvas": {"clear": [20, 20, 20]},
  "layers": [
    {
      "name": "desktop",
      "z": 0,
      "commands": [
        {"op": "RECT", "x": 0, "y": 0, "w": 800, "h": 600, "color": [40, 40, 60]}
      ]
    },
    {
      "name": "window",
      "z": 10,
      "commands": [
        {"op": "RECT", "x": 100, "y": 100, "w": 300, "h": 200, "color": [50, 50, 50]},
        {"op": "RECT", "x": 100, "y": 100, "w": 300, "h": 30, "color": [70, 130, 180]}
      ]
    },
    {
      "name": "cursor",
      "z": 100,
      "commands": [
        {"op": "VLINE", "x": 400, "y": 300, "length": 15, "color": [255, 255, 255]},
        {"op": "HLINE", "x": 400, "y": 300, "length": 10, "color": [255, 255, 255]}
      ]
    }
  ],
  "output": {"file": "ui_stack.png"}
}
```

---

## Testing Your Prompts

### Quick Test Loop

```bash
# 1. Get JSON from LLM
# 2. Save as test.json
# 3. Run:
python pxscene_run.py test.json

# 4. Iterate based on result
```

### Validation

```bash
# Compile only (check for errors):
python pxscene_compile.py test.json test.pxterm

# If successful, run:
python pxos_llm_terminal.py test.pxterm
```

---

## Tips for Better LLM Output

### ✅ DO

- Use the system prompt verbatim
- Ask for specific features: "3 layers", "save as X.png"
- Reference colors by RGB values: "red = [255, 0, 0]"
- Specify positions clearly: "centered", "at (100, 100)"
- Request validation: "make sure JSON is valid"

### ❌ DON'T

- Ask for unsupported operations (circles, text, etc.) - LLM will hallucinate
- Omit the system prompt - LLM won't know the format
- Request huge scenes (>20 commands) - too complex to debug
- Use ambiguous descriptions - LLM will guess

---

## Integration with Different LLMs

### ChatGPT / Claude

```
System: [paste system prompt above]
User: Draw a red rectangle on a black background. Save as "test.png".
```

Copy JSON output → save as `test.json` → run

### Gemini CLI

```bash
# In Gemini CLI
system> [paste system prompt]
user> Draw a red rectangle on a black background
```

Copy JSON → save → run

### LM Studio / Local Models

```
System: [paste system prompt]
User: Draw a simple UI window
```

---

## Troubleshooting

### LLM adds comments in JSON

**Problem**: `{"op": "RECT", /* comment */ "x": 100}`

**Solution**: Re-emphasize in prompt:
```
CRITICAL: Output ONLY valid JSON. No comments of any kind.
```

### LLM adds explanation text

**Problem**:
```
Here's the JSON:
{"canvas": ...}
```

**Solution**: Add to prompt:
```
Output MUST start with { and end with }. No other text.
```

### LLM uses invalid operations

**Problem**: `{"op": "CIRCLE", ...}`

**Solution**: List valid ops explicitly:
```
ONLY these operations are valid: CLEAR, RECT, PIXEL, HLINE, VLINE
Do not use: CIRCLE, TEXT, LINE, or any other operation.
```

---

## Next Steps

Once basic scenes work:

1. **Iterate with LLM**: "Make the box bigger", "Add another layer"
2. **Build a gallery**: Save successful scenes as examples
3. **Create templates**: Common patterns for reuse
4. **Test different LLMs**: Compare output quality
5. **Extend PXSCENE**: Add operations as needed

---

## Full Example Session

```
# Terminal 1: Start LLM with system prompt
$ gemini-cli
system> [paste PXSCENE system prompt]

user> Draw a sunset scene with orange sky and green ground
assistant> {
  "canvas": {"clear": [255, 140, 0]},
  "layers": [
    {
      "name": "sky",
      "z": 0,
      "commands": [
        {"op": "RECT", "x": 0, "y": 0, "w": 800, "h": 400, "color": [255, 140, 0]}
      ]
    },
    {
      "name": "ground",
      "z": 5,
      "commands": [
        {"op": "RECT", "x": 0, "y": 400, "w": 800, "h": 200, "color": [34, 139, 34]}
      ]
    }
  ],
  "output": {"file": "sunset.png"}
}

# Terminal 2: Save and run
$ cat > sunset.json
[paste JSON]
^D

$ python pxscene_run.py sunset.json
[1/2] ✓ Compilation successful
[2/2] ✓ Execution complete
✓ Done!

# Check result
$ ls sunset.png
sunset.png
```

---

**This is machine code for LLMs. Built by LLMs. Starting with the terminal.**
