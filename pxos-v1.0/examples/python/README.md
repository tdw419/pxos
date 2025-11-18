# pxOS Python Examples

These examples demonstrate writing pxOS programs in Python using the `pxos` module API.

## How It Works

1. **Write Python** using the `pxos` module
2. **Compile** with `pxpyc.py` to generate primitives
3. **Build** with `build_pxos.py` to create bootable binary

The Python code runs on your **development machine** and generates pxOS primitives. The compiled output runs on **pxOS** (in QEMU or real hardware).

## Examples

### hello_simple.py

The simplest possible program - clear screen and print message.

```bash
cd ../../tools
python3 pxpyc.py ../examples/python/hello_simple.py -o ../pxos_commands.txt
cd ..
python3 build_pxos.py
qemu-system-i386 -fda pxos.bin
```

### hello_multiline.py

Demonstrates printing multiple lines of text in sequence.

### hello_colors.py

Shows how to use VGA color attributes with `make_color()`.

### bootloader_demo.py

Simulates a bootloader sequence with status messages.

## Quick Start

```bash
# Install path (from pxos-v1.0 directory)
cd tools

# Compile Python to primitives
python3 pxpyc.py ../examples/python/hello_simple.py -o ../pxos_commands.txt

# Build pxOS
cd ..
python3 build_pxos.py

# Run in QEMU
qemu-system-i386 -fda pxos.bin
```

## One-Command Build

The compiler can do everything in one step:

```bash
cd tools
python3 pxpyc.py ../examples/python/hello_simple.py --run
```

This will:
1. Compile Python → primitives
2. Build pxOS binary
3. Launch QEMU

## Available API Functions

See `tools/pxos/__init__.py` for the complete API documentation.

### Display

- `clear_screen(color=0x07)` - Clear screen with color
- `print_text(text, row=None, col=None, attr=0x07)` - Print string
- `print_char(char, row=None, col=None, attr=0x07)` - Print character
- `move_cursor(row, col)` - Move cursor

### Colors

- `make_color(fg, bg=0)` - Create color attribute
- Constants: `BLACK`, `BLUE`, `GREEN`, `CYAN`, `RED`, `MAGENTA`, `BROWN`, `LIGHT_GRAY`, `WHITE`, etc.

### Keyboard

- `read_key()` - Wait for keypress
- `check_key()` - Non-blocking key check

### Control Flow

- `loop_forever()` - Infinite loop (halt system)
- `delay(ms)` - Delay (approximate)

## Limitations (v0.1)

Current compiler supports:
- ✅ Basic function calls
- ✅ String literals
- ✅ Integer literals
- ✅ Function definitions (main only)
- ✅ Simple imports

Not yet supported:
- ❌ Variables
- ❌ Loops (for/while)
- ❌ Conditionals (if/else)
- ❌ Expressions
- ❌ Multiple functions
- ❌ Classes

These will be added in future versions as the compiler matures.

## For LLM Developers

When writing pxOS code, prefer Python over raw primitives:

**Good** (Python):
```python
from pxos import clear_screen, print_text

def main():
    clear_screen()
    print_text("Hello!")

if __name__ == "__main__":
    main()
```

**Avoid** (Raw primitives):
```
WRITE 0x7C00 0xB8
WRITE 0x7C01 0x00
WRITE 0x7C02 0xB8
...
```

The Python API is:
- More readable
- Less error-prone
- Self-documenting
- Easier to maintain

## Next Steps

See the parent [README.md](../../README.md) for:
- Building from primitives
- Running in QEMU
- Creating your own programs
