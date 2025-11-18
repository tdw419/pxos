# pxOS Quick Start Guide

Welcome to pxOS! This guide will get you started with Python development for pxOS.

---

## What You've Got

pxOS now supports **two ways** to write code:

1. **Python (recommended)** - Write Python, compile to primitives
2. **Primitives (advanced)** - Write raw WRITE/DEFINE commands

---

## Prerequisites

### Required

- Python 3.7+
- QEMU (for testing): `qemu-system-i386`

### Optional (for Pixel Cartridges)

```bash
pip install pillow numpy
```

---

## Hello World in 30 Seconds

### 1. Write Python Code

Create `my_program.py`:

```python
from pxos import clear_screen, print_text, loop_forever

def main():
    clear_screen()
    print_text("Hello from Python!")
    loop_forever()

if __name__ == "__main__":
    main()
```

### 2. Compile & Run

```bash
cd tools
python3 pxpyc.py my_program.py --run
```

That's it! QEMU will launch with your OS.

---

## Step-by-Step Workflow

### Option A: One Command

```bash
cd pxos-v1.0/tools
python3 pxpyc.py ../examples/python/hello_simple.py --run
```

This:
1. Compiles Python → primitives
2. Builds pxos.bin
3. Launches QEMU

### Option B: Manual Steps

```bash
# 1. Compile Python to primitives
cd pxos-v1.0/tools
python3 pxpyc.py ../examples/python/hello_simple.py -o ../pxos_commands.txt

# 2. Build binary
cd ..
python3 build_pxos.py

# 3. Test in QEMU
qemu-system-i386 -fda pxos.bin
```

---

## Example Programs

All in `examples/python/`:

| File | Description |
|------|-------------|
| `hello_simple.py` | Basic hello world |
| `hello_multiline.py` | Multiple text lines |
| `hello_colors.py` | Color attributes |
| `bootloader_demo.py` | Boot sequence simulation |

Try them:

```bash
cd tools
python3 pxpyc.py ../examples/python/hello_multiline.py --run
```

---

## Python API Overview

### Display Functions

```python
clear_screen(color=0x07)                    # Clear screen
print_text("Hello", row=10, col=20)         # Print string
print_char('X', row=5, col=5)               # Print character
move_cursor(row, col)                       # Move cursor
```

### Colors

```python
from pxos import make_color, WHITE, BLUE, RED

# Create color attribute
attr = make_color(WHITE, BLUE)  # White on blue
print_text("Title", attr=attr)

# Available colors:
# BLACK, BLUE, GREEN, CYAN, RED, MAGENTA, BROWN,
# LIGHT_GRAY, DARK_GRAY, LIGHT_BLUE, LIGHT_GREEN,
# LIGHT_CYAN, LIGHT_RED, LIGHT_MAGENTA, YELLOW, WHITE
```

### Keyboard

```python
key = read_key()        # Wait for key (blocking)
key = check_key()       # Check for key (non-blocking)
```

### Control Flow

```python
loop_forever()          # Infinite loop (halt system)
delay(500)              # Delay ~500ms
```

### Memory

```python
value = peek(0xB8000)   # Read byte from memory
poke(0xB8000, 0x41)     # Write byte to memory
```

---

## Creating Your Own Programs

### Template

```python
#!/usr/bin/env python3
"""
My pxOS Program
"""

from pxos import clear_screen, print_text, loop_forever
from pxos import make_color, WHITE, BLUE

def main():
    # Clear screen with blue background
    clear_screen(make_color(WHITE, BLUE))

    # Print title
    print_text("My Program v1.0")

    # Your code here...

    # Halt
    loop_forever()

if __name__ == "__main__":
    main()
```

### Compile & Test

```bash
cd tools
python3 pxpyc.py my_program.py --run
```

---

## Working with Primitives (Advanced)

If you prefer raw primitives:

### Create `my_primitives.txt`

```
COMMENT My pxOS Program
DEFINE boot_start 0x7C00

WRITE 0x7C00 0xB4    COMMENT MOV AH, 0x0E
WRITE 0x7C01 0x0E
WRITE 0x7C02 0xB0    COMMENT MOV AL, 'H'
WRITE 0x7C03 0x48
WRITE 0x7C04 0xCD    COMMENT INT 0x10
WRITE 0x7C05 0x10
WRITE 0x7C06 0xEB    COMMENT JMP $
WRITE 0x7C07 0xFE
```

### Build

```bash
python3 build_pxos.py
qemu-system-i386 -fda pxos.bin
```

See `docs/primitives.md` for full reference.

---

## Pixel Cartridges (Porting Foreign Binaries)

Convert any binary to pixel format for archival and porting.

### Create Cartridge

```bash
cd tools
python3 make_pxcart.py some_program.bin \
  --isa x86_32 \
  --abi elf_linux \
  --entry 0x400000 \
  --license MIT \
  --author "Your Name" \
  -o program.pxcart.png
```

**Note**: Requires `pip install pillow numpy`

### Read Cartridge

```bash
# Show info
python3 read_pxcart.py program.pxcart.png --info

# Extract binary
python3 read_pxcart.py program.pxcart.png --extract restored.bin

# Verify integrity
python3 read_pxcart.py program.pxcart.png --verify
```

See `docs/pxcartridge_v0.md` for format specification.

---

## Troubleshooting

### "Command not found: qemu-system-i386"

Install QEMU:

```bash
# Ubuntu/Debian
sudo apt install qemu-system-x86

# macOS
brew install qemu

# Arch
sudo pacman -S qemu
```

### "This tool requires PIL (Pillow) and numpy"

For pixel cartridge tools:

```bash
pip install pillow numpy
```

### Compilation Errors

Check your Python syntax:

```bash
python3 -m py_compile your_program.py
```

Make sure you're importing from `pxos`:

```python
from pxos import clear_screen, print_text
```

### Binary Too Large

Boot sector limit: 512 bytes (minus 2 for signature = 510 bytes usable)

Current Python compiler v0.1 generates simple code, but if you hit limits:
- Reduce string lengths
- Simplify logic
- Use primitives for ultra-compact code

---

## Next Steps

1. **Try all examples**: Run each example in `examples/python/`
2. **Read the docs**:
   - `docs/primitives.md` - Primitive command reference
   - `docs/architecture.md` - How pxOS works
   - `docs/pxcartridge_v0.md` - Pixel cartridge format
   - `PYTHON_ROADMAP.md` - Future Python features
3. **Write your own program**: Start with the template above
4. **Learn x86 assembly**: To understand what the compiler generates
5. **Contribute**: Add features to the Python compiler!

---

## Project Structure

```
pxos-v1.0/
├── build_pxos.py           # Binary builder
├── pxos_commands.txt       # Generated primitives (or write your own)
├── pxos.bin                # Bootable binary (output)
├── docs/
│   ├── primitives.md       # Primitive reference
│   ├── architecture.md     # System architecture
│   └── pxcartridge_v0.md   # Cartridge format spec
├── examples/
│   └── python/             # Python examples
│       ├── hello_simple.py
│       ├── hello_multiline.py
│       ├── hello_colors.py
│       └── bootloader_demo.py
├── tools/
│   ├── pxpyc.py            # Python compiler
│   ├── make_pxcart.py      # Create pixel cartridges
│   ├── read_pxcart.py      # Read pixel cartridges
│   └── pxos/
│       └── __init__.py     # Python API module
├── PYTHON_ROADMAP.md       # Development roadmap
└── QUICK_START.md          # This file!
```

---

## Getting Help

- **Documentation**: See `docs/` folder
- **Examples**: See `examples/python/`
- **Issues**: Check if PIL/numpy/QEMU are installed
- **API Reference**: See `tools/pxos/__init__.py` docstrings

---

## What's Next?

The Python compiler is just the beginning! See `PYTHON_ROADMAP.md` for:

- **Track A**: Enhanced Python compiler (variables, loops, conditionals)
- **Track B**: Native MicroPython runtime on pxOS
- **Track C**: Pixel cartridge porting system

---

**Welcome to pxOS development! 🚀**

For the full development roadmap and architecture details, see:
- `PYTHON_ROADMAP.md`
- `docs/architecture.md`
