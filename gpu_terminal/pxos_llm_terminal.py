"""
pxOS LLM Terminal - Text Command Interface

This provides a stable text protocol on top of the GPU terminal.
The LLM (or user) can issue simple text commands to draw graphics.

Commands:
- CLEAR r g b [a]      - Clear screen to color
- PIXEL x y r g b [a]  - Draw a pixel
- RECT x y w h r g b [a] - Draw a rectangle
- QUIT                 - Exit

All coordinates are integers, colors are 0-255.
"""

import threading
import sys
from pxos_gpu_terminal import PxOSTerminalGPU, WIDTH, HEIGHT


class PxOSLLMTerminal:
    """
    LLM-facing terminal with text command protocol.

    Runs the GPU terminal in the main thread and accepts
    commands from stdin in a background thread.
    """

    def __init__(self):
        self.gpu_terminal = PxOSTerminalGPU()
        self.running = True
        self.command_thread = None

    def parse_command(self, line: str) -> bool:
        """
        Parse and execute a command line.

        Returns:
            True if should continue, False if should quit
        """
        line = line.strip()
        if not line or line.startswith('#'):
            return True

        parts = line.split()
        cmd = parts[0].upper()
        args = parts[1:]

        try:
            if cmd == "QUIT" or cmd == "EXIT":
                print("[LLM] Quitting...")
                return False

            elif cmd == "CLEAR":
                if len(args) not in (3, 4):
                    print(f"[LLM] ERROR: CLEAR requires 3 or 4 args (r g b [a]), got {len(args)}")
                    return True
                rgba = list(map(int, args))
                if len(rgba) == 3:
                    rgba.append(255)
                self.gpu_terminal.cmd_clear(*rgba)

            elif cmd == "PIXEL":
                if len(args) not in (5, 6):
                    print(f"[LLM] ERROR: PIXEL requires 5 or 6 args (x y r g b [a]), got {len(args)}")
                    return True
                coords = list(map(int, args[:2]))
                rgba = list(map(int, args[2:]))
                if len(rgba) == 3:
                    rgba.append(255)
                self.gpu_terminal.cmd_pixel(*coords, *rgba)

            elif cmd == "RECT":
                if len(args) not in (7, 8):
                    print(f"[LLM] ERROR: RECT requires 7 or 8 args (x y w h r g b [a]), got {len(args)}")
                    return True
                xywh = list(map(int, args[:4]))
                rgba = list(map(int, args[4:]))
                if len(rgba) == 3:
                    rgba.append(255)
                self.gpu_terminal.cmd_rect(*xywh, *rgba)

            elif cmd == "HLINE":
                if len(args) not in (6, 7):
                    print(f"[LLM] ERROR: HLINE requires 6 or 7 args (x y length r g b [a]), got {len(args)}")
                    return True
                xyl = list(map(int, args[:3]))
                rgba = list(map(int, args[3:]))
                if len(rgba) == 3:
                    rgba.append(255)
                self.gpu_terminal.cmd_hline(*xyl, *rgba)

            elif cmd == "VLINE":
                if len(args) not in (6, 7):
                    print(f"[LLM] ERROR: VLINE requires 6 or 7 args (x y length r g b [a]), got {len(args)}")
                    return True
                xyl = list(map(int, args[:3]))
                rgba = list(map(int, args[3:]))
                if len(rgba) == 3:
                    rgba.append(255)
                self.gpu_terminal.cmd_vline(*xyl, *rgba)

            elif cmd == "HELP":
                self.print_help()

            elif cmd == "INFO":
                self.print_info()

            else:
                print(f"[LLM] ERROR: Unknown command '{cmd}'. Type HELP for command list.")

        except ValueError as e:
            print(f"[LLM] ERROR: Invalid number format: {e}")
        except Exception as e:
            print(f"[LLM] ERROR: {e}")

        return True

    def print_help(self):
        """Print help message"""
        print()
        print("=" * 60)
        print("pxOS LLM Terminal - Command Reference")
        print("=" * 60)
        print()
        print("CLEAR r g b [a]")
        print("  Clear entire screen to color (0-255 per channel)")
        print("  Example: CLEAR 0 0 0")
        print()
        print("PIXEL x y r g b [a]")
        print("  Draw a single pixel")
        print("  Example: PIXEL 400 300 255 255 255")
        print()
        print("RECT x y w h r g b [a]")
        print("  Draw a filled rectangle")
        print("  Example: RECT 100 100 200 150 255 0 0")
        print()
        print("HLINE x y length r g b [a]")
        print("  Draw a horizontal line")
        print("  Example: HLINE 0 100 800 255 0 0")
        print()
        print("VLINE x y length r g b [a]")
        print("  Draw a vertical line")
        print("  Example: VLINE 400 0 600 0 255 0")
        print()
        print("INFO")
        print("  Show terminal information")
        print()
        print("HELP")
        print("  Show this help message")
        print()
        print("QUIT or EXIT")
        print("  Exit the terminal")
        print()
        print("Notes:")
        print("  - Origin (0, 0) is top-left")
        print("  - Alpha channel is optional (defaults to 255)")
        print("  - Lines starting with # are comments")
        print("=" * 60)
        print()

    def print_info(self):
        """Print terminal information"""
        print()
        print("=" * 60)
        print("pxOS LLM Terminal - Information")
        print("=" * 60)
        print(f"Display Size: {WIDTH}x{HEIGHT}")
        print(f"VRAM Shape: {self.gpu_terminal.vram.shape}")
        print(f"VRAM Dtype: {self.gpu_terminal.vram.dtype}")
        print(f"Frozen Shader: frozen_display.wgsl v0.1")
        print("=" * 60)
        print()

    def command_loop(self):
        """Background thread: read and execute commands from stdin"""
        print()
        print("=" * 60)
        print("pxOS LLM Terminal v0.1")
        print("=" * 60)
        print("Type HELP for command list, QUIT to exit")
        print()

        while self.running:
            try:
                line = input("pxos> ")
                if not self.parse_command(line):
                    self.running = False
                    # Note: We can't easily stop the wgpu event loop from here,
                    # so user will need to close the window manually
                    print("[LLM] Please close the window to exit.")
                    break
            except EOFError:
                print("\n[LLM] EOF received, exiting...")
                self.running = False
                print("[LLM] Please close the window to exit.")
                break
            except KeyboardInterrupt:
                print("\n[LLM] Interrupted, exiting...")
                self.running = False
                print("[LLM] Please close the window to exit.")
                break

    def run(self):
        """Start both the command loop and GUI"""
        # Start command loop in background thread
        self.command_thread = threading.Thread(target=self.command_loop, daemon=True)
        self.command_thread.start()

        # Run GUI in main thread (blocking)
        self.gpu_terminal.run()

        # When GUI exits, signal command thread to stop
        self.running = False


def batch_mode(commands: list[str]):
    """
    Run a list of commands in batch mode (no interactive loop).

    Args:
        commands: List of command strings to execute

    Example:
        batch_mode([
            "CLEAR 0 0 0",
            "RECT 100 100 200 150 255 0 0",
        ])
    """
    terminal = PxOSLLMTerminal()

    print("[LLM] Batch mode: executing commands...")
    for cmd in commands:
        print(f"pxos> {cmd}")
        terminal.parse_command(cmd)

    print("[LLM] Batch complete. Close window to exit.")
    terminal.gpu_terminal.run()


def script_mode(script_path: str):
    """
    Execute commands from a script file.

    Args:
        script_path: Path to file containing commands (one per line)

    Example script file:
        # Draw a red square
        CLEAR 0 0 0
        RECT 100 100 200 200 255 0 0
    """
    with open(script_path, 'r') as f:
        commands = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    batch_mode(commands)


def main():
    """Main entry point: interactive mode"""
    terminal = PxOSLLMTerminal()

    # Optional: start with a demo scene
    print("[LLM] Loading demo scene...")
    terminal.gpu_terminal.cmd_clear(16, 16, 32)  # Dark background
    terminal.gpu_terminal.cmd_rect(50, 50, 100, 100, 255, 0, 0)  # Red square
    terminal.gpu_terminal.cmd_pixel(400, 300, 255, 255, 255)  # Center pixel

    # Start interactive mode
    terminal.run()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Script mode: python pxos_llm_terminal.py script.txt
        script_mode(sys.argv[1])
    else:
        # Interactive mode
        main()
