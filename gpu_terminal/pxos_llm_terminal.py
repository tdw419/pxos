"""
pxOS LLM Terminal - Text Command Protocol

This provides a stable text protocol (PXTERM v1) on top of the GPU terminal.
The LLM (or user) can issue simple text commands to draw graphics.

Commands:
- INFO                     - Show canvas info
- LAYERS                   - List all layers
- LAYER NEW name z         - Create new layer
- LAYER USE name           - Switch to layer
- CLEAR r g b [a]          - Clear current layer
- PIXEL x y r g b [a]      - Draw a pixel
- RECT x y w h r g b [a]   - Draw a rectangle
- HLINE x y length r g b [a] - Draw horizontal line
- VLINE x y length r g b [a] - Draw vertical line
- SAVE path                - Save frame to PNG
- HELP                     - Show help
- QUIT                     - Exit

All coordinates are integers, colors are 0-255.
"""

import sys
from pathlib import Path
from pxos_gpu_terminal import PxOSTerminalGPU


class PxOSLLMTerminal:
    """
    LLM-facing terminal interface.

    Provides text-based commands (PXTERM v1) that map to GPU terminal API.
    """

    def __init__(self):
        self.gpu_terminal = PxOSTerminalGPU()
        self.running = True

    def process_command(self, line: str) -> bool:
        """
        Process a single PXTERM command.

        Returns True if the frame should be redrawn.
        """
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            return False

        parts = line.split()
        if not parts:
            return False

        cmd = parts[0].upper()
        args = parts[1:]

        try:
            # ========== Introspection commands ==========

            if cmd == "INFO":
                self.gpu_terminal.info()
                return False

            elif cmd == "LAYERS":
                self.gpu_terminal.layer_list()
                return False

            elif cmd == "HELP":
                self.print_help()
                return False

            elif cmd == "QUIT":
                self.running = False
                return False

            # ========== Layer management ==========

            elif cmd == "LAYER":
                if len(args) < 2:
                    print("[LLM] ERROR: LAYER requires subcommand (NEW/USE/DELETE)")
                    return False

                subcmd = args[0].upper()

                if subcmd == "NEW":
                    if len(args) < 3:
                        print("[LLM] ERROR: LAYER NEW requires name and z_index")
                        return False
                    name = args[1]
                    z_index = int(args[2])
                    self.gpu_terminal.layer_new(name, z_index)
                    return False

                elif subcmd == "USE":
                    if len(args) < 2:
                        print("[LLM] ERROR: LAYER USE requires name")
                        return False
                    name = args[1]
                    self.gpu_terminal.layer_use(name)
                    return False

                elif subcmd == "DELETE":
                    if len(args) < 2:
                        print("[LLM] ERROR: LAYER DELETE requires name")
                        return False
                    name = args[1]
                    self.gpu_terminal.layer_delete(name)
                    return True

                else:
                    print(f"[LLM] ERROR: Unknown LAYER subcommand: {subcmd}")
                    return False

            # ========== Drawing commands ==========

            elif cmd == "CLEAR":
                if len(args) not in (3, 4):
                    print(f"[LLM] ERROR: CLEAR requires 3 or 4 args (r g b [a]), got {len(args)}")
                    return False
                rgba = list(map(int, args))
                if len(rgba) == 3:
                    rgba.append(255)
                self.gpu_terminal.cmd_clear(*rgba)
                return True

            elif cmd == "PIXEL":
                if len(args) not in (5, 6):
                    print(f"[LLM] ERROR: PIXEL requires 5 or 6 args (x y r g b [a]), got {len(args)}")
                    return False
                xy = list(map(int, args[:2]))
                rgba = list(map(int, args[2:]))
                if len(rgba) == 3:
                    rgba.append(255)
                self.gpu_terminal.cmd_pixel(*xy, *rgba)
                return True

            elif cmd == "RECT":
                if len(args) not in (7, 8):
                    print(f"[LLM] ERROR: RECT requires 7 or 8 args (x y w h r g b [a]), got {len(args)}")
                    return False
                xywh = list(map(int, args[:4]))
                rgba = list(map(int, args[4:]))
                if len(rgba) == 3:
                    rgba.append(255)
                self.gpu_terminal.cmd_rect(*xywh, *rgba)
                return True

            elif cmd == "HLINE":
                if len(args) not in (6, 7):
                    print(f"[LLM] ERROR: HLINE requires 6 or 7 args (x y length r g b [a]), got {len(args)}")
                    return False
                xyl = list(map(int, args[:3]))
                rgba = list(map(int, args[3:]))
                if len(rgba) == 3:
                    rgba.append(255)
                self.gpu_terminal.cmd_hline(*xyl, *rgba)
                return True

            elif cmd == "VLINE":
                if len(args) not in (6, 7):
                    print(f"[LLM] ERROR: VLINE requires 6 or 7 args (x y length r g b [a]), got {len(args)}")
                    return False
                xyl = list(map(int, args[:3]))
                rgba = list(map(int, args[3:]))
                if len(rgba) == 3:
                    rgba.append(255)
                self.gpu_terminal.cmd_vline(*xyl, *rgba)
                return True

            # ========== Utility commands ==========

            elif cmd == "SAVE":
                if len(args) != 1:
                    print(f"[LLM] ERROR: SAVE requires path argument")
                    return False
                self.gpu_terminal.save_frame(args[0])
                return False

            else:
                print(f"[LLM] ERROR: Unknown command: {cmd}")
                print("[LLM] Type HELP for available commands")
                return False

        except ValueError as e:
            print(f"[LLM] ERROR: Invalid arguments: {e}")
            return False
        except Exception as e:
            print(f"[LLM] ERROR: {e}")
            return False

    def run_script(self, script_path: Path):
        """Execute a PXTERM script file"""
        print(f"[LLM] Executing script: {script_path}")

        try:
            with open(script_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                should_redraw = self.process_command(line)

            # Final redraw to show results
            print(f"[LLM] Script completed ({len(lines)} lines)")
            print("[LLM] Window is open. Close to exit.")

        except FileNotFoundError:
            print(f"[LLM] ERROR: File not found: {script_path}")
            sys.exit(1)
        except Exception as e:
            print(f"[LLM] ERROR: {e}")
            sys.exit(1)

    def run_interactive(self):
        """Run interactive REPL mode"""
        print("=" * 60)
        print("pxOS LLM Terminal - PXTERM v1")
        print("=" * 60)
        print("Type HELP for available commands, QUIT to exit")
        print()

        while self.running:
            try:
                line = input("pxos> ").strip()
                should_redraw = self.process_command(line)

            except EOFError:
                print("\n[LLM] EOF received, exiting...")
                break
            except KeyboardInterrupt:
                print("\n[LLM] Interrupted, exiting...")
                break

        print("[LLM] Goodbye!")

    def print_help(self):
        """Print help text"""
        help_text = """
PXTERM v1 Commands:

Introspection:
  INFO                    - Show canvas size and current layer
  LAYERS                  - List all layers with z-index
  HELP                    - Show this help

Layer Management:
  LAYER NEW name z        - Create new layer with z-index
  LAYER USE name          - Switch to layer for drawing
  LAYER DELETE name       - Delete layer

Drawing (operates on current layer):
  CLEAR r g b [a]         - Fill layer with color
  PIXEL x y r g b [a]     - Draw single pixel
  RECT x y w h r g b [a]  - Draw filled rectangle
  HLINE x y len r g b [a] - Draw horizontal line
  VLINE x y len r g b [a] - Draw vertical line

Utility:
  SAVE path               - Save frame to PNG file
  QUIT                    - Exit terminal

Notes:
  - All numeric arguments are integers
  - Colors: r, g, b, a are 0-255
  - Alpha (a) is optional, defaults to 255
  - Lines starting with # are comments
  - Canvas size: 800x600 (default)

Examples:
  CLEAR 0 0 0                    # Black background
  LAYER NEW ui 10                # Create UI layer at z=10
  LAYER USE ui                   # Switch to UI layer
  RECT 100 100 200 150 255 0 0   # Red rectangle
  SAVE output.png                # Save to file
"""
        print(help_text)


def main():
    """Main entry point"""

    # Check if script mode or interactive mode
    if len(sys.argv) > 1:
        # Script mode: python pxos_llm_terminal.py script.pxterm
        script_path = Path(sys.argv[1])
        terminal = PxOSLLMTerminal()
        terminal.run_script(script_path)

        # Keep window open
        terminal.gpu_terminal.run()

    else:
        # Interactive mode
        terminal = PxOSLLMTerminal()

        # Start the GPU event loop
        # Note: In interactive mode, we can't use input() while the GPU runs
        # So we just run the GPU with a demo scene
        print("[LLM] Interactive mode: Starting with demo scene")
        print("[LLM] For full REPL, run a script instead")
        print()

        # Draw demo
        terminal.gpu_terminal.cmd_clear(0, 0, 32)
        terminal.gpu_terminal.layer_new("demo", 10)
        terminal.gpu_terminal.layer_use("demo")
        terminal.gpu_terminal.cmd_rect(100, 100, 200, 150, 255, 0, 0, 200)
        terminal.gpu_terminal.cmd_hline(0, 300, 800, 255, 255, 255)

        terminal.gpu_terminal.run()


if __name__ == "__main__":
    main()
