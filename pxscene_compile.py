#!/usr/bin/env python3
"""
pxscene_compile.py - Compile a PXSCENE JSON file to PXTERM v1 assembly.
Usage:
  python pxscene_compile.py <scene.json> <out.pxterm>
"""

import json
import sys

def safe_int(d, key, default=0):
    try:
        return int(d.get(key, default))
    except (ValueError, TypeError):
        return default

def safe_color(raw, default=(255, 0, 255, 255)):
    try:
        r, g, b, *rest = raw
        a = rest[0] if rest else 255
        return (int(r), int(g), int(b), int(a))
    except (ValueError, TypeError):
        return default

def emit_rect(lines, x, y, w, h, color):
    r, g, b, a = color
    if a != 255:
        lines.append(f"RECT {x} {y} {w} {h} {r} {g} {b} {a}")
    else:
        lines.append(f"RECT {x} {y} {w} {h} {r} {g} {b}")

def emit_pixel(lines, x, y, color):
    r, g, b, a = color
    if a != 255:
        lines.append(f"PIXEL {x} {y} {r} {g} {b} {a}")
    else:
        lines.append(f"PIXEL {x} {y} {r} {g} {b}")

def emit_hline(lines, x, y, length, color):
    r, g, b, a = color
    if a != 255:
        lines.append(f"HLINE {x} {y} {length} {r} {g} {b} {a}")
    else:
        lines.append(f"HLINE {x} {y} {length} {r} {g} {b}")

def compile_command(lines, base_x, base_y, cmd):
    op = cmd.get("op", "UNKNOWN").upper()

    try:
        if op == "RECT":
            x = base_x + safe_int(cmd, "x")
            y = base_y + safe_int(cmd, "y")
            w = max(0, safe_int(cmd, "w"))
            h = max(0, safe_int(cmd, "h"))
            color = safe_color(cmd.get("color"))
            emit_rect(lines, x, y, w, h, color)

        elif op == "PIXEL":
            x = base_x + safe_int(cmd, "x")
            y = base_y + safe_int(cmd, "y")
            color = safe_color(cmd.get("color"))
            emit_pixel(lines, x, y, color)

        elif op == "HLINE":
            x = base_x + safe_int(cmd, "x")
            y = base_y + safe_int(cmd, "y")
            length = max(0, safe_int(cmd, "length"))
            color = safe_color(cmd.get("color"))
            emit_hline(lines, x, y, length, color)

        elif op in ("CONSOLE", "PRINT"):
            lines.append(f"PRINT {cmd.get('value', '')}")

        elif op == "HSTACK":
            x0 = base_x + safe_int(cmd, "x")
            y0 = base_y + safe_int(cmd, "y")
            spacing = safe_int(cmd, "spacing")
            cursor_x = x0

            for child in cmd.get("children", []):
                compile_command(lines, cursor_x, y0, child)
                if child.get("op", "").upper() == "RECT":
                    cursor_x += safe_int(child, "w") + spacing

        elif op == "VSTACK":
            x0 = base_x + safe_int(cmd, "x")
            y0 = base_y + safe_int(cmd, "y")
            spacing = safe_int(cmd, "spacing")
            cursor_y = y0

            for child in cmd.get("children", []):
                compile_command(lines, x0, cursor_y, child)
                if child.get("op", "").upper() == "RECT":
                    cursor_y += safe_int(child, "h") + spacing

        elif op == "LABEL" or op == "TEXT":
            x = base_x + safe_int(cmd, "x")
            y = base_y + safe_int(cmd, "y")
            color = safe_color(cmd.get("color"))
            r, g, b, a = color
            text = cmd.get("value", "")
            if a != 255:
                lines.append(f"TEXT {x} {y} {r} {g} {b} {a} {text}")
            else:
                lines.append(f"TEXT {x} {y} {r} {g} {b} {text}")

        elif op == "BUTTON":
            x = base_x + safe_int(cmd, "x")
            y = base_y + safe_int(cmd, "y")
            w = max(0, safe_int(cmd, "w"))
            h = max(0, safe_int(cmd, "h"))

            bg = safe_color(cmd.get("bg"), default=(60,60,120,255))
            fg = safe_color(cmd.get("fg"), default=(255,255,255,255))
            label = cmd.get("label", "")
            padding = safe_int(cmd, "padding", 4)

            emit_rect(lines, x, y, w, h, bg)

            r, g, b, a = fg
            text_x = x + padding
            text_y = y + padding

            if a != 255:
                lines.append(f"TEXT {text_x} {text_y} {r} {g} {b} {a} {label}")
            else:
                lines.append(f"TEXT {text_x} {text_y} {r} {g} {b} {label}")

        elif op == "WINDOW":
            x = base_x + safe_int(cmd, "x")
            y = base_y + safe_int(cmd, "y")
            w = max(0, safe_int(cmd, "w"))
            h = max(0, safe_int(cmd, "h"))

            frame_color = safe_color(cmd.get("frame_color"), default=(20,20,60,255))
            title_bar_color = safe_color(cmd.get("title_bar_color"), default=(30,30,90,255))
            title_fg = safe_color(cmd.get("title_fg"), default=(255,255,255,255))
            title = cmd.get("title", "")
            title_bar_height = safe_int(cmd, "title_bar_height", 40)
            padding = safe_int(cmd, "padding", 8)

            emit_rect(lines, x, y, w, h, frame_color)
            emit_rect(lines, x, y, w, title_bar_height, title_bar_color)

            tr, tg, tb, ta = title_fg
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

    except Exception as e:
        lines.append(f"# ERROR: Failed to compile command {cmd}: {e}")

def compile_scene(scene_data):
    lines = []

    canvas = scene_data.get("canvas", {})
    clear_color = canvas.get("clear")
    if clear_color:
        r,g,b, *a = safe_color(clear_color)
        lines.append(f"CLEAR {r} {g} {b} {a[0] if a else 255}")

    layers = scene_data.get("layers", [])
    for layer in layers:
        name = layer.get("name", "unnamed_layer")
        z = safe_int(layer, "z")

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
    if len(sys.argv) < 3:
        print("Usage: pxscene_compile.py <scene.json> <out.pxterm> [--strict]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    imperfect_mode = "--strict" not in sys.argv

    try:
        with open(in_path, "r", encoding="utf-8") as f:
            scene_data = json.load(f)
    except Exception as e:
        if imperfect_mode:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"PRINT [error] Failed to load scene: {e}\nSAVE error.png")
            print(f"Failed to load scene {in_path}, wrote error to {out_path}")
            return
        else:
            raise

    pxterm_code = compile_scene(scene_data)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(pxterm_code)

    print(f"Compiled {in_path} to {out_path}")

if __name__ == "__main__":
    main()
