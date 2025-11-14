#!/usr/bin/env python3
"""
pxscene_compile.py - Compile a PXSCENE JSON file to PXTERM v1 assembly.
Usage:
  python pxscene_compile.py <scene.json> <out.pxterm>
"""

import json
import sys

def emit_rect(lines, x, y, w, h, color):
    r, g, b, *rest = color
    a = rest[0] if rest else 255
    if a != 255:
        lines.append(f"RECT {x} {y} {w} {h} {r} {g} {b} {a}")
    else:
        lines.append(f"RECT {x} {y} {w} {h} {r} {g} {b}")

def emit_pixel(lines, x, y, color):
    r, g, b, *rest = color
    a = rest[0] if rest else 255
    if a != 255:
        lines.append(f"PIXEL {x} {y} {r} {g} {b} {a}")
    else:
        lines.append(f"PIXEL {x} {y} {r} {g} {b}")

def emit_hline(lines, x, y, length, color):
    r, g, b, *rest = color
    a = rest[0] if rest else 255
    if a != 255:
        lines.append(f"HLINE {x} {y} {length} {r} {g} {b} {a}")
    else:
        lines.append(f"HLINE {x} {y} {length} {r} {g} {b}")

def compile_command(lines, base_x, base_y, cmd):
    """
    Lower a single PXSCENE command (including HSTACK/VSTACK) into PXTERM lines.
    base_x, base_y are offsets applied to the command's own x,y (for nested layouts).
    """
    op = cmd["op"].upper()

    if op == "RECT":
        x = base_x + int(cmd.get("x", 0))
        y = base_y + int(cmd.get("y", 0))
        w = int(cmd["w"])
        h = int(cmd["h"])
        emit_rect(lines, x, y, w, h, cmd["color"])

    elif op == "PIXEL":
        x = base_x + int(cmd.get("x", 0))
        y = base_y + int(cmd.get("y", 0))
        emit_pixel(lines, x, y, cmd["color"])

    elif op == "HLINE":
        x = base_x + int(cmd.get("x", 0))
        y = base_y + int(cmd.get("y", 0))
        length = int(cmd["length"])
        emit_hline(lines, x, y, length, cmd["color"])

    elif op == "HSTACK":
        x0 = base_x + int(cmd.get("x", 0))
        y0 = base_y + int(cmd.get("y", 0))
        spacing = int(cmd.get("spacing", 0))
        cursor_x = x0

        for child in cmd.get("children", []):
            child_op = child["op"].upper()
            child_local = dict(child)
            child_local.setdefault("x", 0)
            child_local.setdefault("y", 0)

            if child_op == "RECT":
                w = int(child_local["w"])
                compile_command(lines, cursor_x, y0, child_local)
                cursor_x += w + spacing
            else:
                compile_command(lines, cursor_x, y0, child_local)

    elif op == "VSTACK":
        x0 = base_x + int(cmd.get("x", 0))
        y0 = base_y + int(cmd.get("y", 0))
        spacing = int(cmd.get("spacing", 0))
        cursor_y = y0

        for child in cmd.get("children", []):
            child_op = child["op"].upper()
            child_local = dict(child)
            child_local.setdefault("x", 0)
            child_local.setdefault("y", 0)

            if child_op == "RECT":
                h = int(child_local["h"])
                compile_command(lines, x0, cursor_y, child_local)
                cursor_y += h + spacing
            else:
                compile_command(lines, x0, cursor_y, child_local)

    elif op == "LABEL":
        x = base_x + int(cmd.get("x", 0))
        y = base_y + int(cmd.get("y", 0))
        r, g, b, *rest = cmd["color"]
        a = rest[0] if rest else 255
        text = cmd.get("value", "")
        if a != 255:
            lines.append(f"TEXT {x} {y} {r} {g} {b} {a} {text}")
        else:
            lines.append(f"TEXT {x} {y} {r} {g} {b} {text}")

    elif op == "BUTTON":
        x = base_x + int(cmd.get("x", 0))
        y = base_y + int(cmd.get("y", 0))
        w = int(cmd["w"])
        h = int(cmd["h"])

        bg = cmd.get("bg", [60, 60, 120, 255])
        fg = cmd.get("fg", [255, 255, 255, 255])
        label = cmd.get("label", "")
        padding = int(cmd.get("padding", 4))

        emit_rect(lines, x, y, w, h, bg)

        r, g, b, *rest = fg
        a = rest[0] if rest else 255
        text_x = x + padding
        text_y = y + padding

        if a != 255:
            lines.append(f"TEXT {text_x} {text_y} {r} {g} {b} {a} {label}")
        else:
            lines.append(f"TEXT {text_x} {text_y} {r} {g} {b} {label}")

    elif op == "WINDOW":
        x = base_x + int(cmd.get("x", 0))
        y = base_y + int(cmd.get("y", 0))
        w = int(cmd["w"])
        h = int(cmd["h"])

        frame_color = cmd.get("frame_color", [20, 20, 60, 255])
        title_bar_color = cmd.get("title_bar_color", [30, 30, 90, 255])
        title_fg = cmd.get("title_fg", [255, 255, 255, 255])
        title = cmd.get("title", "")
        title_bar_height = int(cmd.get("title_bar_height", 40))
        padding = int(cmd.get("padding", 8))

        emit_rect(lines, x, y, w, h, frame_color)
        emit_rect(lines, x, y, w, title_bar_height, title_bar_color)

        tr, tg, tb, *rest = title_fg
        ta = rest[0] if rest else 255
        text_x = x + padding
        text_y = y + (title_bar_height // 2 - 4)

        if ta != 255:
            lines.append(f"TEXT {text_x} {text_y} {tr} {tg} {tb} {ta} {title}")
        else:
            lines.append(f"TEXT {text_x} {text_y} {tr} {tg} {tb} {title}")

        content_x = x + padding
        content_y = y + title_bar_height + padding

        for child in cmd.get("children", []):
            compile_command(lines, content_x, content_y, child)

    else:
        lines.append(f"# WARNING: unknown op {op}, skipping")

def compile_scene(scene_data):
    lines = []

    canvas = scene_data.get("canvas", {})
    width = canvas.get("width", 800)
    height = canvas.get("height", 600)
    clear_color = canvas.get("clear")

    if clear_color:
        lines.append(f"CLEAR {clear_color[0]} {clear_color[1]} {clear_color[2]} {clear_color[3] if len(clear_color) > 3 else 255}")

    layers = scene_data.get("layers", [])
    for layer in layers:
        name = layer["name"]
        z = int(layer.get("z", 0))

        lines.append(f"# Layer: {name}")
        lines.append(f"LAYER NEW {name} {z}")
        lines.append(f"LAYER USE {name}")

        for cmd in layer.get("commands", []):
            compile_command(lines, 0, 0, cmd)

        lines.append("")

    output = scene_data.get("output", {})
    if "file" in output:
        lines.append(f"SAVE {output['file']}")

    return "\n".join(lines)

def main():
    if len(sys.argv) != 3:
        print("Usage: pxscene_compile.py <scene.json> <out.pxterm>")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    with open(in_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

    pxterm_code = compile_scene(scene_data)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(pxterm_code)

    print(f"Compiled {in_path} to {out_path}")

if __name__ == "__main__":
    main()
