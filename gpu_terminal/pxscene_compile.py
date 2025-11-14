#!/usr/bin/env python3
"""
pxscene_compile.py - Compile PXSCENE JSON into PXTERM v1 script.

This is the "compiler" layer that bridges LLM-friendly structured data
and the PXTERM machine code.

Architecture:
  LLM → PXSCENE JSON → pxscene_compile → PXTERM v1 → pxos_llm_terminal → GPU

Supports:
  - PXSCENE v0.1: Basic operations (CLEAR, PIXEL, RECT, HLINE, VLINE)
  - PXSCENE v0.2: Layout operations (HSTACK, VSTACK)

Usage:
  python pxscene_compile.py scene.json scene.pxterm
  python pxos_llm_terminal.py scene.pxterm
"""

import sys
import json
from typing import List, Dict, Any
from pathlib import Path


# ============================================================================
# PXTERM Emission Helpers
# ============================================================================

def emit_rect(lines: List[str], x: int, y: int, w: int, h: int, color: List[int]):
    """Emit a RECT command to PXTERM"""
    r, g, b, *rest = color
    a = rest[0] if rest else 255
    if a != 255:
        lines.append(f"RECT {x} {y} {w} {h} {r} {g} {b} {a}")
    else:
        lines.append(f"RECT {x} {y} {w} {h} {r} {g} {b}")


def emit_pixel(lines: List[str], x: int, y: int, color: List[int]):
    """Emit a PIXEL command to PXTERM"""
    r, g, b, *rest = color
    a = rest[0] if rest else 255
    if a != 255:
        lines.append(f"PIXEL {x} {y} {r} {g} {b} {a}")
    else:
        lines.append(f"PIXEL {x} {y} {r} {g} {b}")


def emit_hline(lines: List[str], x: int, y: int, length: int, color: List[int]):
    """Emit an HLINE command to PXTERM"""
    r, g, b, *rest = color
    a = rest[0] if rest else 255
    if a != 255:
        lines.append(f"HLINE {x} {y} {length} {r} {g} {b} {a}")
    else:
        lines.append(f"HLINE {x} {y} {length} {r} {g} {b}")


def emit_vline(lines: List[str], x: int, y: int, length: int, color: List[int]):
    """Emit a VLINE command to PXTERM"""
    r, g, b, *rest = color
    a = rest[0] if rest else 255
    if a != 255:
        lines.append(f"VLINE {x} {y} {length} {r} {g} {b} {a}")
    else:
        lines.append(f"VLINE {x} {y} {length} {r} {g} {b}")


# ============================================================================
# Layout Engine - Compile Commands Recursively
# ============================================================================

def compile_command(lines: List[str], base_x: int, base_y: int, cmd: Dict[str, Any]):
    """
    Lower a single PXSCENE command (including HSTACK/VSTACK) into PXTERM lines.

    This is the layout engine that compiles high-level layout operations
    into low-level PXTERM drawing commands.

    Args:
        lines: Output list to append PXTERM commands to
        base_x, base_y: Base offset to apply to command coordinates
        cmd: PXSCENE command dictionary

    Supports:
        - v0.1 ops: CLEAR, PIXEL, RECT, HLINE, VLINE, COMMENT
        - v0.2 ops: HSTACK, VSTACK (layout operators)
    """
    op = cmd.get("op", "").upper()

    # ===== v0.1 Basic Operations =====

    if op == "CLEAR":
        r, g, b, *rest = cmd["color"]
        a = rest[0] if rest else 255
        lines.append(f"CLEAR {int(r)} {int(g)} {int(b)} {int(a)}")

    elif op == "RECT":
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

    elif op == "VLINE":
        x = base_x + int(cmd.get("x", 0))
        y = base_y + int(cmd.get("y", 0))
        length = int(cmd["length"])
        emit_vline(lines, x, y, length, cmd["color"])

    elif op == "COMMENT":
        # Allow structured comments in JSON
        comment = cmd.get("text", "")
        lines.append(f"# {comment}")

    # ===== v0.2 Layout Operations =====

    elif op == "HSTACK":
        # Horizontal stack of children - automatic x-positioning
        x0 = base_x + int(cmd.get("x", 0))
        y0 = base_y + int(cmd.get("y", 0))
        spacing = int(cmd.get("spacing", 0))
        cursor_x = x0

        lines.append(f"# HSTACK at ({x0}, {y0}) spacing={spacing}")

        for child in cmd.get("children", []):
            child_op = child.get("op", "").upper()
            child_local = dict(child)  # shallow copy
            # Default local offsets to 0 so compile_command adds base
            child_local.setdefault("x", 0)
            child_local.setdefault("y", 0)

            if child_op == "RECT":
                w = int(child_local["w"])
                h = int(child_local["h"])
                compile_command(lines, cursor_x, y0, child_local)
                cursor_x += w + spacing
            else:
                # For non-RECT, place at current cursor
                compile_command(lines, cursor_x, y0, child_local)
                # Note: Could add width calculation for other ops if needed

    elif op == "VSTACK":
        # Vertical stack of children - automatic y-positioning
        x0 = base_x + int(cmd.get("x", 0))
        y0 = base_y + int(cmd.get("y", 0))
        spacing = int(cmd.get("spacing", 0))
        cursor_y = y0

        lines.append(f"# VSTACK at ({x0}, {y0}) spacing={spacing}")

        for child in cmd.get("children", []):
            child_op = child.get("op", "").upper()
            child_local = dict(child)
            child_local.setdefault("x", 0)
            child_local.setdefault("y", 0)

            if child_op == "RECT":
                w = int(child_local["w"])
                h = int(child_local["h"])
                compile_command(lines, x0, cursor_y, child_local)
                cursor_y += h + spacing
            else:
                # For non-RECT, place at current cursor
                compile_command(lines, x0, cursor_y, child_local)

    else:
        lines.append(f"# WARNING: Unknown operation '{op}', skipping")


def compile_scene(scene: Dict[str, Any]) -> List[str]:
    """
    Compile a PXSCENE JSON structure into PXTERM v1 commands.

    Supports both v0.1 (basic) and v0.2 (layout) operations.

    Args:
        scene: PXSCENE JSON structure

    Returns:
        List of PXTERM command lines
    """
    lines: List[str] = []

    # Header
    lines.append("# PXTERM v1 program")
    lines.append("# Generated from PXSCENE (v0.1/v0.2)")
    lines.append("#")
    lines.append("# Architecture: LLM → PXSCENE → PXTERM → GPU")
    lines.append("")

    # Introspection
    lines.append("INFO")
    lines.append("LAYERS")
    lines.append("")

    # Canvas setup (optional global clear)
    canvas = scene.get("canvas", {})
    if "clear" in canvas:
        r, g, b, *rest = canvas["clear"]
        a = rest[0] if rest else 255
        lines.append(f"# Canvas clear")
        lines.append(f"CLEAR {int(r)} {int(g)} {int(b)} {int(a)}")
        lines.append("")

    # Process layers
    layers = scene.get("layers", [])
    for layer_idx, layer in enumerate(layers):
        name = layer.get("name", f"layer_{layer_idx}")
        z = int(layer.get("z", 0))
        opacity = int(layer.get("opacity", 255))

        lines.append(f"# Layer: {name} (z={z}, opacity={opacity})")
        lines.append(f"LAYER NEW {name} {z}")
        lines.append(f"LAYER USE {name}")
        lines.append("")

        # Process commands for this layer using layout engine
        commands = layer.get("commands", [])
        for cmd in commands:
            compile_command(lines, 0, 0, cmd)

        lines.append("")  # Blank line after each layer

    # Output directives
    output = scene.get("output", {})
    if "file" in output:
        lines.append(f"# Save output")
        lines.append(f"SAVE {output['file']}")
        lines.append("")

    # Final status
    lines.append("# End of program")
    lines.append("INFO")
    lines.append("LAYERS")

    return lines


def validate_scene(scene: Dict[str, Any]) -> bool:
    """
    Validate PXSCENE structure (v0.1/v0.2).

    Returns True if valid, prints errors and returns False otherwise.
    """
    errors = []

    # Check top-level structure
    if not isinstance(scene, dict):
        errors.append("Scene must be a JSON object")
        for err in errors:
            print(f"[VALIDATE ERROR] {err}", file=sys.stderr)
        return False

    # Validate layers
    layers = scene.get("layers", [])
    if not isinstance(layers, list):
        errors.append("'layers' must be an array")
    else:
        for idx, layer in enumerate(layers):
            if not isinstance(layer, dict):
                errors.append(f"Layer {idx} must be an object")
                continue

            if "name" not in layer:
                errors.append(f"Layer {idx} missing 'name' field")

            commands = layer.get("commands", [])
            if not isinstance(commands, list):
                errors.append(f"Layer {idx} 'commands' must be an array")
                continue

            for cmd_idx, cmd in enumerate(commands):
                if not isinstance(cmd, dict):
                    errors.append(f"Layer {idx}, command {cmd_idx} must be an object")
                    continue

                if "op" not in cmd:
                    errors.append(f"Layer {idx}, command {cmd_idx} missing 'op' field")

    # Print errors
    if errors:
        print(f"[VALIDATE] Found {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return False

    return True


def main():
    """Main entry point"""

    if len(sys.argv) not in (2, 3):
        print("pxscene_compile.py - PXSCENE → PXTERM v1 Compiler")
        print()
        print("Supports:")
        print("  - PXSCENE v0.1: Basic operations")
        print("  - PXSCENE v0.2: Layout operations (HSTACK, VSTACK)")
        print()
        print("Usage:")
        print("  python pxscene_compile.py <scene.json> <output.pxterm>")
        print("  python pxscene_compile.py <scene.json>  (outputs to <scene>.pxterm)")
        print()
        print("Examples:")
        print("  python pxscene_compile.py scene1.json scene1.pxterm")
        print("  python pxscene_compile.py scene1.json")
        print()
        print("Then run:")
        print("  python pxos_llm_terminal.py scene1.pxterm")
        sys.exit(1)

    # Parse arguments
    scene_path = Path(sys.argv[1])

    if len(sys.argv) == 3:
        out_path = Path(sys.argv[2])
    else:
        # Auto-generate output name: scene.json → scene.pxterm
        out_path = scene_path.with_suffix(".pxterm")

    # Load scene
    print(f"[COMPILE] Loading scene: {scene_path}")
    try:
        with open(scene_path, "r", encoding="utf-8") as f:
            scene = json.load(f)
    except FileNotFoundError:
        print(f"[COMPILE ERROR] File not found: {scene_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[COMPILE ERROR] Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate
    print(f"[COMPILE] Validating PXSCENE structure...")
    if not validate_scene(scene):
        sys.exit(1)

    # Compile
    print(f"[COMPILE] Compiling to PXTERM v1 (with layout engine)...")
    lines = compile_scene(scene)

    # Write output
    print(f"[COMPILE] Writing to: {out_path}")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
    except IOError as e:
        print(f"[COMPILE ERROR] Failed to write: {e}", file=sys.stderr)
        sys.exit(1)

    # Success
    print(f"[COMPILE] ✓ Success!")
    print(f"[COMPILE] Generated {len(lines)} lines of PXTERM v1 code")
    print()
    print("To run:")
    print(f"  python pxos_llm_terminal.py {out_path}")


if __name__ == "__main__":
    main()
