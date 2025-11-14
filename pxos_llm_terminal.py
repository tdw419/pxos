#!/usr/bin/env python3
"""
PxOS LLM Terminal - PXTERM command interpreter
Parses and executes PXTERM instruction format

IMPERFECT COMPUTING MODE:
- Bad commands log warnings and continue (never crash)
- Missing args use defaults when possible
- Unknown commands are logged and skipped
- Drawing errors are caught and logged
"""
from __future__ import annotations
import sys
from typing import Optional
from pxos_gpu_terminal import PxOSTerminalGPU


def run_pxterm_file(filename: str, output: Optional[str] = None, imperfect: bool = True):
    """Execute a .pxterm file (imperfect mode: never crashes)"""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[ERROR] Could not read file '{filename}': {e}")
        if not imperfect:
            raise
        return

    # Parse canvas dimensions (imperfect: use defaults if missing/bad)
    try:
        first_line = lines[0].strip().split()
        if first_line[0] != "CANVAS":
            print("[WARNING] First line should be 'CANVAS width height', using defaults")
            width, height = 800, 600
        else:
            width = int(first_line[1]) if len(first_line) > 1 else 800
            height = int(first_line[2]) if len(first_line) > 2 else 600
            width = max(1, min(4096, width))  # clamp to reasonable range
            height = max(1, min(4096, height))
    except Exception as e:
        print(f"[WARNING] Error parsing CANVAS: {e}, using defaults 800x600")
        width, height = 800, 600

    # Create terminal (imperfect: always try, even if canvas setup failed)
    try:
        term = PxOSTerminalGPU(width=width, height=height)
        print(f"Created {width}x{height} canvas")
    except Exception as e:
        print(f"[ERROR] Could not create terminal: {e}")
        if not imperfect:
            raise
        return

    # Process commands
    for line_num, line in enumerate(lines[1:], start=2):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(maxsplit=1)
        if not parts:
            continue

        cmd = parts[0]
        args = parts[1].split() if len(parts) > 1 else []

        try:
            if cmd == "LAYER":
                # LAYER name z
                if len(args) < 2:
                    print(f"Line {line_num}: Usage: LAYER name z")
                    continue
                name = args[0]
                z = int(args[1])
                term.add_layer(name, z)
                term.set_layer(name)
                print(f"Created layer '{name}' at z={z}")

            elif cmd == "SELECT":
                # SELECT layer_name
                if len(args) < 1:
                    print(f"Line {line_num}: Usage: SELECT layer_name")
                    continue
                term.set_layer(args[0])
                print(f"Selected layer '{args[0]}'")

            elif cmd == "CLEAR":
                # CLEAR r g b [a]
                if len(args) < 3:
                    print(f"Line {line_num}: Usage: CLEAR r g b [a]")
                    continue
                r, g, b = int(args[0]), int(args[1]), int(args[2])
                a = int(args[3]) if len(args) > 3 else 255
                term.clear_layer(r, g, b, a)
                print(f"Cleared layer to ({r},{g},{b},{a})")

            elif cmd == "PIXEL":
                # PIXEL x y r g b [a]
                if len(args) < 5:
                    print(f"Line {line_num}: Usage: PIXEL x y r g b [a]")
                    continue
                x = int(args[0])
                y = int(args[1])
                r = int(args[2])
                g = int(args[3])
                b = int(args[4])
                a = int(args[5]) if len(args) > 5 else 255
                term.pixel(x, y, r, g, b, a)

            elif cmd == "HLINE":
                # HLINE x y w r g b [a]
                if len(args) < 6:
                    print(f"Line {line_num}: Usage: HLINE x y w r g b [a]")
                    continue
                x = int(args[0])
                y = int(args[1])
                w = int(args[2])
                r = int(args[3])
                g = int(args[4])
                b = int(args[5])
                a = int(args[6]) if len(args) > 6 else 255
                term.hline(x, y, w, r, g, b, a)

            elif cmd == "RECT":
                # RECT x y w h r g b [a]
                if len(args) < 7:
                    print(f"Line {line_num}: Usage: RECT x y w h r g b [a]")
                    continue
                x = int(args[0])
                y = int(args[1])
                w = int(args[2])
                h = int(args[3])
                r = int(args[4])
                g = int(args[5])
                b = int(args[6])
                a = int(args[7]) if len(args) > 7 else 255
                term.rect(x, y, w, h, r, g, b, a)
                print(f"RECT at ({x},{y}) size {w}x{h}")

            elif cmd == "TEXT":
                # TEXT x y r g b [a] message...
                if len(args) < 5:
                    print(f"Line {line_num}: Usage: TEXT x y r g b [a] message...")
                    continue

                x = int(args[0])
                y = int(args[1])
                r = int(args[2])
                g = int(args[3])
                b = int(args[4])

                # Check if next arg is alpha or start of message
                # If we have at least 7 args and arg[5] is numeric, it's alpha
                if len(args) >= 7:
                    try:
                        a = int(args[5])
                        msg_parts = args[6:]
                    except ValueError:
                        a = 255
                        msg_parts = args[5:]
                else:
                    a = 255
                    msg_parts = args[5:] if len(args) > 5 else []

                message = " ".join(msg_parts)
                term.text(x, y, message, r, g, b, a)
                print(f'TEXT "{message}" at ({x},{y}) color ({r},{g},{b},{a})')

            elif cmd == "PRINT":
                # PRINT message... - log to console (stdout for now)
                message = " ".join(args) if args else ""
                print(f"[PXTERM] {message}")

            elif cmd == "DRAW":
                # DRAW [output.png]
                output_file = args[0] if args else output
                if output_file:
                    term.save_frame(output_file)
                else:
                    frame = term.draw_frame()
                    print(f"Rendered frame: {frame.shape}")

            else:
                # Unknown command - imperfect: log and continue, don't crash
                print(f"Line {line_num}: [WARNING] Unknown command '{cmd}' - skipped")

        except Exception as e:
            # Imperfect computing: catch ALL exceptions, log, and continue
            print(f"Line {line_num}: [ERROR] '{line.strip()}' -> {type(e).__name__}: {e}")
            continue

    # Final draw if no explicit DRAW command (imperfect: catch errors)
    if output:
        try:
            term.save_frame(output)
        except Exception as e:
            print(f"[ERROR] Could not save final frame to '{output}': {e}")
            if not imperfect:
                raise


def main():
    if len(sys.argv) < 2:
        print("Usage: python pxos_llm_terminal.py <file.pxterm> [output.png]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    run_pxterm_file(input_file, output_file)


if __name__ == "__main__":
    main()
