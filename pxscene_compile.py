#!/usr/bin/env python3
"""
PXSCENE Compiler - Converts PXSCENE JSON to PXTERM instructions
Supports: RECT, PIXEL, HLINE, TEXT, HSTACK, VSTACK

IMPERFECT COMPUTING MODE:
- Bad/missing values use safe defaults
- Unknown ops generate warning comments, not errors
- Never crashes, always produces valid PXTERM output
"""
from __future__ import annotations
import json
import sys
from typing import List, Dict, Any, Tuple


# Imperfect computing helpers - never crash, always return something valid
def safe_int(d: Dict, key: str, default: int = 0, clamp_min: int = None, clamp_max: int = None) -> int:
    """Safely extract integer, with optional clamping. Never raises."""
    try:
        val = int(d.get(key, default))
        if clamp_min is not None:
            val = max(clamp_min, val)
        if clamp_max is not None:
            val = min(clamp_max, val)
        return val
    except (ValueError, TypeError, KeyError):
        return default


def safe_color(d: Dict, key: str = "color", default: Tuple[int, int, int, int] = (255, 255, 255, 255)) -> Tuple[int, int, int, int]:
    """Safely extract RGBA color. Returns default on any error."""
    try:
        color = d.get(key, list(default))
        if not isinstance(color, (list, tuple)):
            return default

        # Extract r, g, b, optional a
        r = int(color[0]) if len(color) > 0 else default[0]
        g = int(color[1]) if len(color) > 1 else default[1]
        b = int(color[2]) if len(color) > 2 else default[2]
        a = int(color[3]) if len(color) > 3 else default[3]

        # Clamp to valid range
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        a = max(0, min(255, a))

        return (r, g, b, a)
    except (ValueError, TypeError, IndexError):
        return default


def safe_str(d: Dict, key: str, default: str = "") -> str:
    """Safely extract string. Never raises."""
    try:
        val = d.get(key, default)
        return str(val) if val is not None else default
    except Exception:
        return default


def compile_command(cmd: Dict[str, Any], base_x: int = 0, base_y: int = 0) -> List[str]:
    """
    Compile a single PXSCENE command to PXTERM instructions.
    Layout commands (HSTACK, VSTACK) expand into their children.
    base_x, base_y are used for relative positioning within layouts.
    """
    lines = []
    op = safe_str(cmd, "op", "").upper()

    if op == "RECT":
        # RECT with absolute positioning (imperfect: clamps negative sizes to 0)
        x = base_x + safe_int(cmd, "x", 0)
        y = base_y + safe_int(cmd, "y", 0)
        w = safe_int(cmd, "w", 0, clamp_min=0)
        h = safe_int(cmd, "h", 0, clamp_min=0)
        r, g, b, a = safe_color(cmd)
        lines.append(f"RECT {x} {y} {w} {h} {r} {g} {b} {a}")

    elif op == "PIXEL":
        x = base_x + safe_int(cmd, "x", 0)
        y = base_y + safe_int(cmd, "y", 0)
        r, g, b, a = safe_color(cmd)
        lines.append(f"PIXEL {x} {y} {r} {g} {b} {a}")

    elif op == "HLINE":
        x = base_x + safe_int(cmd, "x", 0)
        y = base_y + safe_int(cmd, "y", 0)
        w = safe_int(cmd, "w", 0, clamp_min=0)
        r, g, b, a = safe_color(cmd)
        lines.append(f"HLINE {x} {y} {w} {r} {g} {b} {a}")

    elif op == "TEXT":
        # TEXT command with position and color
        x = base_x + safe_int(cmd, "x", 0)
        y = base_y + safe_int(cmd, "y", 0)
        r, g, b, a = safe_color(cmd)
        text = safe_str(cmd, "value", "")
        if a != 255:
            line = f"TEXT {x} {y} {r} {g} {b} {a} {text}"
        else:
            line = f"TEXT {x} {y} {r} {g} {b} {text}"
        lines.append(line)

    elif op == "HSTACK":
        # Horizontal stack: arrange children left to right
        x = base_x + safe_int(cmd, "x", 0)
        y = base_y + safe_int(cmd, "y", 0)
        spacing = safe_int(cmd, "spacing", 0)
        children = cmd.get("children", [])

        cursor_x = x
        for child in children:
            # Compile child with current cursor position (imperfect: skip bad children)
            try:
                child_lines = compile_command(child, cursor_x, y)
                lines.extend(child_lines)

                # Move cursor right by child width + spacing
                child_width = safe_int(child, "w", 0)
                cursor_x += child_width + spacing
            except Exception as e:
                lines.append(f"# WARNING: HSTACK child failed: {e}")

    elif op == "VSTACK":
        # Vertical stack: arrange children top to bottom
        x = base_x + safe_int(cmd, "x", 0)
        y = base_y + safe_int(cmd, "y", 0)
        spacing = safe_int(cmd, "spacing", 0)
        children = cmd.get("children", [])

        cursor_y = y
        for child in children:
            # Compile child with current cursor position (imperfect: skip bad children)
            try:
                child_lines = compile_command(child, x, cursor_y)
                lines.extend(child_lines)

                # Move cursor down by child height + spacing
                child_height = safe_int(child, "h", 0)
                cursor_y += child_height + spacing
            except Exception as e:
                lines.append(f"# WARNING: VSTACK child failed: {e}")

    elif op == "":
        # Empty op - silently skip (LLM might generate empty commands)
        pass

    else:
        # Unknown op - generate warning comment, continue compilation (imperfect)
        lines.append(f"# WARNING: Unknown operation '{op}' - skipped")
        print(f"Warning: Unknown operation '{op}'", file=sys.stderr)

    return lines


def compile_pxscene(scene: Dict[str, Any]) -> List[str]:
    """Compile entire PXSCENE to PXTERM instructions (imperfect: never crashes)"""
    lines = []

    # Canvas setup (imperfect: use safe defaults)
    canvas = scene.get("canvas", {})
    width = safe_int(canvas, "width", 800, clamp_min=1, clamp_max=4096)
    height = safe_int(canvas, "height", 600, clamp_min=1, clamp_max=4096)
    lines.append(f"CANVAS {width} {height}")

    # Process layers (imperfect: skip broken layers)
    layers = scene.get("layers", [])
    for layer in layers:
        try:
            name = safe_str(layer, "name", "layer")
            z = safe_int(layer, "z", 0)
            lines.append(f"LAYER {name} {z}")
            lines.append(f"SELECT {name}")

            # Clear layer if specified (imperfect: safe color extraction)
            if "clear" in layer:
                r, g, b, a = safe_color(layer, "clear", (0, 0, 0, 0))
                lines.append(f"CLEAR {r} {g} {b} {a}")

            # Compile commands (imperfect: skip broken commands)
            commands = layer.get("commands", [])
            for cmd in commands:
                try:
                    cmd_lines = compile_command(cmd)
                    lines.extend(cmd_lines)
                except Exception as e:
                    lines.append(f"# WARNING: Command failed: {e}")
        except Exception as e:
            lines.append(f"# WARNING: Layer '{layer.get('name', '?')}' failed: {e}")

    # Output (imperfect: safe filename)
    output = scene.get("output", {})
    output_file = safe_str(output, "file", "output.png")
    lines.append(f"DRAW {output_file}")

    return lines


def main():
    if len(sys.argv) < 2:
        print("Usage: python pxscene_compile.py <input.json> [output.pxterm]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace(".json", ".pxterm")

    # Load scene
    with open(input_file, 'r') as f:
        scene = json.load(f)

    # Compile
    instructions = compile_pxscene(scene)

    # Write output
    with open(output_file, 'w') as f:
        f.write("\n".join(instructions) + "\n")

    print(f"Compiled {input_file} -> {output_file}")
    print(f"Generated {len(instructions)} instructions")


if __name__ == "__main__":
    main()
