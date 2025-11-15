from pxos_gpu_terminal import PxOSTerminalGPU, run
import shlex
import threading
import sys

def process_command(term, line):
    """Processes a single command line, returns False on QUIT."""
    try:
        parts = shlex.split(line)
        if not parts:
            return True
        cmd = parts[0].upper()
        args = parts[1:]

        if cmd == "QUIT":
            return False

        if cmd == "INFO":
            print(f"Canvas: {term.width}x{term.height}, current layer: '{term.active_layer_name}'")
        elif cmd == "LAYERS":
            print("Layers:")
            for name, layer in term.layers.items():
                print(f"  - {name}: z_index={layer.z_index}, visible={layer.visible}")
        elif cmd == "LAYER":
            if not args:
                term.print_line("[pxterm warn] LAYER command requires sub-command (NEW, USE).")
            else:
                sub_cmd = args[0].upper()
                if sub_cmd == "NEW" and len(args) >= 2:
                    name = args[1]
                    z_index = int(args[2]) if len(args) > 2 else 0
                    term.new_layer(name, z_index)
                elif sub_cmd == "USE" and len(args) == 2:
                    name = args[1]
                    term.use_layer(name)
                else:
                    term.print_line(f"[pxterm warn] Invalid LAYER command: {line}")
        elif cmd == "SAVE":
            if len(args) == 1:
                path = args[0]
                term.save_frame(path)
            else:
                term.print_line("[pxterm warn] SAVE command expects a single argument: <filepath>")
        elif cmd == "TEXT":
            if len(args) < 6:
                term.print_line("[pxterm warn] Usage: TEXT x y r g b [a] message...")
            else:
                x, y, r, g, b = [int(arg) for arg in args[:5]]
                a = 255
                message_start_index = 5
                if len(args) >= 7:
                    try:
                        a = int(args[5])
                        message_start_index = 6
                    except ValueError:
                        pass
                message = " ".join(args[message_start_index:])
                term.text(x, y, message, r, g, b, a)
        elif cmd == "PRINT":
            message = " ".join(args)
            term.print_line(message)
        else:
            int_args = [int(arg) for arg in args]
            if cmd == "CLEAR":
                if len(int_args) in (3, 4):
                    term.clear(*int_args)
                else:
                    term.print_line("[pxterm warn] CLEAR expects 3 or 4 arguments: r g b [a]")
            elif cmd == "PIXEL":
                if len(int_args) in (5, 6):
                    term.pixel(*int_args)
                else:
                    term.print_line("[pxterm warn] PIXEL expects 5 or 6 arguments: x y r g b [a]")
            elif cmd == "HLINE":
                if len(int_args) in (5, 6):
                    term.hline(*int_args)
                else:
                    term.print_line("[pxterm warn] HLINE expects 5 or 6 arguments: x y length r g b [a]")
            elif cmd == "RECT":
                if len(int_args) in (6, 7):
                    term.rect(*int_args)
                else:
                    term.print_line("[pxterm warn] RECT expects 6 or 7 arguments: x y w h r g b [a]")
            else:
                term.print_line(f"[pxterm warn] Unknown command: {line}")

        if cmd not in ["INFO", "LAYERS", "PRINT", "LAYER", "SAVE"]:
             term.draw_frame()

    except Exception as e:
        if term.imperfect:
            term.print_line(f"[pxterm error] {line} -> {e}")
        else:
            raise
    return True

def run_script(term, lines):
    """Executes a list of PXTERM instructions non-interactively."""
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not process_command(term, line):
            break
    term.draw_frame()

def repl(term):
    """Runs the interactive command loop."""
    print("LLM Terminal. For commands, see PXTERM_SPEC.md")
    while True:
        try:
            line = input("pxos> ").strip()
            if not process_command(term, line):
                break
        except Exception as e:
            print(f"An error occurred: {e}")

def main():
    is_script_mode = len(sys.argv) > 1
    imperfect_mode = "--strict" not in sys.argv
    term = PxOSTerminalGPU(offscreen=is_script_mode, imperfect=imperfect_mode)

    term.new_layer("background", z_index=-1)
    term.use_layer("background")
    term.clear(20, 20, 30)
    term.use_layer("default")
    term.draw_frame()

    if is_script_mode:
        script_path = sys.argv[1]
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            run_script(term, lines)
        except FileNotFoundError:
            term.print_line(f"Error: Script file not found at {script_path}")

        print("Script finished.")
    else:
        # Interactive REPL mode
        command_thread = threading.Thread(target=repl, args=(term,))
        command_thread.daemon = True
        command_thread.start()

        def animation_frame():
            term.canvas.request_draw(animation_frame)
        animation_frame()
        run()

if __name__ == "__main__":
    main()
