#!/usr/bin/env python3
"""
pxscene_compile.py - Compile PXSCENE v0.1 JSON into PXTERM v1 script.

This is the "compiler" layer that bridges LLM-friendly structured data
and the PXTERM machine code.

Architecture:
  LLM → PXSCENE JSON → pxscene_compile → PXTERM v1 → pxos_llm_terminal → GPU

Usage:
  python pxscene_compile.py scene.json scene.pxterm
  python pxos_llm_terminal.py scene.pxterm
"""

import sys
import json
from typing import List, Dict, Any
from pathlib import Path


def compile_scene(scene: Dict[str, Any]) -> List[str]:
    """
    Compile a PXSCENE JSON structure into PXTERM v1 commands.

    Args:
        scene: PXSCENE v0.1 JSON structure

    Returns:
        List of PXTERM command lines
    """
    lines: List[str] = []

    # Header
    lines.append("# PXTERM v1 program")
    lines.append("# Generated from PXSCENE v0.1")
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

        # Process commands for this layer
        commands = layer.get("commands", [])
        for cmd_idx, cmd in enumerate(commands):
            op = cmd.get("op", "").upper()

            if op == "CLEAR":
                r, g, b, *rest = cmd["color"]
                a = rest[0] if rest else 255
                lines.append(f"CLEAR {int(r)} {int(g)} {int(b)} {int(a)}")

            elif op == "PIXEL":
                x = int(cmd["x"])
                y = int(cmd["y"])
                r, g, b, *rest = cmd["color"]
                a = rest[0] if rest else 255
                lines.append(f"PIXEL {x} {y} {int(r)} {int(g)} {int(b)} {int(a)}")

            elif op == "RECT":
                x = int(cmd["x"])
                y = int(cmd["y"])
                w = int(cmd["w"])
                h = int(cmd["h"])
                r, g, b, *rest = cmd["color"]
                a = rest[0] if rest else 255
                lines.append(f"RECT {x} {y} {w} {h} {int(r)} {int(g)} {int(b)} {int(a)}")

            elif op == "HLINE":
                x = int(cmd["x"])
                y = int(cmd["y"])
                length = int(cmd["length"])
                r, g, b, *rest = cmd["color"]
                a = rest[0] if rest else 255
                lines.append(f"HLINE {x} {y} {length} {int(r)} {int(g)} {int(b)} {int(a)}")

            elif op == "VLINE":
                x = int(cmd["x"])
                y = int(cmd["y"])
                length = int(cmd["length"])
                r, g, b, *rest = cmd["color"]
                a = rest[0] if rest else 255
                lines.append(f"VLINE {x} {y} {length} {int(r)} {int(g)} {int(b)} {int(a)}")

            elif op == "COMMENT":
                # Allow structured comments in JSON
                comment = cmd.get("text", "")
                lines.append(f"# {comment}")

            else:
                lines.append(f"# WARNING: Unknown operation '{op}', skipping")

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
    Validate PXSCENE v0.1 structure.

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
        print("pxscene_compile.py - PXSCENE v0.1 → PXTERM v1 Compiler")
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
    print(f"[COMPILE] Validating PXSCENE v0.1 structure...")
    if not validate_scene(scene):
        sys.exit(1)

    # Compile
    print(f"[COMPILE] Compiling to PXTERM v1...")
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
