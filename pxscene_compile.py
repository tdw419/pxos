#!/usr/bin/env python3
"""
PXSCENE Compiler - Converts PXSCENE JSON to PXTERM instructions
Supports: RECT, PIXEL, HLINE, TEXT, HSTACK, VSTACK
"""
from __future__ import annotations
import json
import sys
from typing import List, Dict, Any


def compile_command(cmd: Dict[str, Any], base_x: int = 0, base_y: int = 0) -> List[str]:
    """
    Compile a single PXSCENE command to PXTERM instructions.
    Layout commands (HSTACK, VSTACK) expand into their children.
    base_x, base_y are used for relative positioning within layouts.
    """
    lines = []
    op = cmd.get("op", "").upper()

    if op == "RECT":
        # RECT with absolute positioning
        x = base_x + int(cmd.get("x", 0))
        y = base_y + int(cmd.get("y", 0))
        w = int(cmd["w"])
        h = int(cmd["h"])
        r, g, b, *rest = cmd["color"]
        a = rest[0] if rest else 255
        lines.append(f"RECT {x} {y} {w} {h} {r} {g} {b} {a}")

    elif op == "PIXEL":
        x = base_x + int(cmd.get("x", 0))
        y = base_y + int(cmd.get("y", 0))
        r, g, b, *rest = cmd["color"]
        a = rest[0] if rest else 255
        lines.append(f"PIXEL {x} {y} {r} {g} {b} {a}")

    elif op == "HLINE":
        x = base_x + int(cmd.get("x", 0))
        y = base_y + int(cmd.get("y", 0))
        w = int(cmd["w"])
        r, g, b, *rest = cmd["color"]
        a = rest[0] if rest else 255
        lines.append(f"HLINE {x} {y} {w} {r} {g} {b} {a}")

    elif op == "TEXT":
        # TEXT command with position and color
        x = base_x + int(cmd.get("x", 0))
        y = base_y + int(cmd.get("y", 0))
        r, g, b, *rest = cmd["color"]
        a = rest[0] if rest else 255
        text = cmd.get("value", "")
        if a != 255:
            line = f"TEXT {x} {y} {r} {g} {b} {a} {text}"
        else:
            line = f"TEXT {x} {y} {r} {g} {b} {text}"
        lines.append(line)

    elif op == "HSTACK":
        # Horizontal stack: arrange children left to right
        x = base_x + int(cmd.get("x", 0))
        y = base_y + int(cmd.get("y", 0))
        spacing = int(cmd.get("spacing", 0))
        children = cmd.get("children", [])

        cursor_x = x
        for child in children:
            # Compile child with current cursor position
            child_lines = compile_command(child, cursor_x, y)
            lines.extend(child_lines)

            # Move cursor right by child width + spacing
            child_width = int(child.get("w", 0))
            cursor_x += child_width + spacing

    elif op == "VSTACK":
        # Vertical stack: arrange children top to bottom
        x = base_x + int(cmd.get("x", 0))
        y = base_y + int(cmd.get("y", 0))
        spacing = int(cmd.get("spacing", 0))
        children = cmd.get("children", [])

        cursor_y = y
        for child in children:
            # Compile child with current cursor position
            child_lines = compile_command(child, x, cursor_y)
            lines.extend(child_lines)

            # Move cursor down by child height + spacing
            child_height = int(child.get("h", 0))
            cursor_y += child_height + spacing

    else:
        print(f"Warning: Unknown operation '{op}'", file=sys.stderr)

    return lines


def compile_pxscene(scene: Dict[str, Any]) -> List[str]:
    """Compile entire PXSCENE to PXTERM instructions"""
    lines = []

    # Canvas setup
    canvas = scene.get("canvas", {})
    width = canvas.get("width", 800)
    height = canvas.get("height", 600)
    lines.append(f"CANVAS {width} {height}")

    # Process layers
    layers = scene.get("layers", [])
    for layer in layers:
        name = layer.get("name", "layer")
        z = layer.get("z", 0)
        lines.append(f"LAYER {name} {z}")
        lines.append(f"SELECT {name}")

        # Clear layer if specified
        if "clear" in layer:
            r, g, b, *rest = layer["clear"]
            a = rest[0] if rest else 255
            lines.append(f"CLEAR {r} {g} {b} {a}")

        # Compile commands
        commands = layer.get("commands", [])
        for cmd in commands:
            cmd_lines = compile_command(cmd)
            lines.extend(cmd_lines)

    # Output
    output = scene.get("output", {})
    output_file = output.get("file", "output.png")
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
