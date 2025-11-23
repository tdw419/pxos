# pxOS v1.0 — Revolutionary OS Architecture

**pxOS** is an educational operating system project demonstrating innovative architectural patterns:

1. **Bootloader Shell** - A minimal x86 bootloader built using custom assembly primitives (`WRITE`, `DEFINE`, `CALL`)
2. **Shader VM Runtime** - A revolutionary GPU programming system using VM architecture (NEW!)

## What's NEW: Shader VM Runtime 🎉

We've added a groundbreaking GPU programming system based on the same architecture as JVM, .NET CLR, and WebAssembly:

**Instead of compiling languages TO shaders, we built a VM that runs ON shaders!**

### Key Features:
- **Language Independence** - Multiple frontends (Effects, Math DSL, Python-like, LOGO)
- **Hot-Reloading** - Change effects at runtime without recompilation
- **GPU Performance** - Full parallel execution on GPU
- **Simple Compilation** - Just emit bytecode, no complex SPIR-V generation
- **Debuggable** - Disassemble bytecode, step through execution

See [runtime/README.md](runtime/README.md) for complete documentation!

### Quick Start - Shader VM:

```bash
# Test the compiler (no GPU required)
cd runtime
python demo.py --test

# Run on GPU (requires wgpu-py)
pip install wgpu glfw
python demo.py --interactive
```

---

## Features

✓ **Direct BIOS boot** - Works on real hardware and emulators
✓ **Interactive shell** - Character echo with Enter key support
✓ **< 1KB code size** - Minimal and educational
✓ **Primitive-based build** - Uses WRITE/DEFINE commands instead of assembly
✓ **Fully documented** - Every byte is traceable to a primitive command

---

## Quick Start

### Boot in Emulator

```bash
# Install QEMU (if not already installed)
sudo apt install qemu-system-x86

# Boot pxOS
./tests/boot_qemu.sh
```

### Build from Source

```bash
# Build the bootable binary
python3 build_pxos.py

# Output: pxos.bin (ready to boot)
```

### Create a Bootable USB (⚠️ Use with caution!)

```bash
# DANGER: This will overwrite the target device!
# Double-check your device path!
sudo dd if=pxos.bin of=/dev/sdX bs=512 count=1 conv=notrunc

# Verify
sudo fdisk -l /dev/sdX
```

> **Warning**: Replace `/dev/sdX` with your actual USB device. This will destroy all data on the target device!

---

## What Does It Do?

1. **Boots**: BIOS loads the first 512 bytes from disk
2. **Clears screen**: Fills VGA text buffer with spaces
3. **Prints welcome**: "pxOS v1> "
4. **Shell loop**:
   - Waits for keyboard input
   - Echoes characters back to screen
   - On Enter: moves to new line and reprints prompt

---

## Memory Map

| Address Range | Label          | Purpose                    |
|---------------|----------------|----------------------------|
| `0x0050`      | cursor_pos     | Cursor position (unused)   |
| `0x7C00-7C27` | Boot loader    | Entry point, setup, clear  |
| `0x7C28-7C33` | print_string   | Print null-terminated str  |
| `0x7C38-7C58` | shell_loop     | Interactive keyboard loop  |
| `0x7C40-7C49` | welcome_msg    | "pxOS v1> " string        |
| `0x7E00`      | shell_prompt   | (reserved for future use)  |
| `0x7E10`      | input_buffer   | (reserved for future use)  |
| `0x01FE-01FF` | Boot signature | `0x55 0xAA` (required)    |

---

## How It's Built: The Primitive System

Traditional OS development uses assembly:

```nasm
mov ah, 0x0E
int 0x10
```

pxOS uses **primitives** during initial development:

```
WRITE 0x7C2D 0xB4    COMMENT mov ah, 0x0E
WRITE 0x7C2E 0x0E
WRITE 0x7C2F 0xCD    COMMENT int 0x10
WRITE 0x7C30 0x10
DEFINE print_string 0x7C28
```

### Advantages

- **Educational**: See exactly what bytes go where
- **Transparent**: No "magic" assembler transformations
- **Hackable**: Easy to modify with any text editor
- **Debuggable**: Direct mapping from command to memory
- **Minimal tooling**: Just Python 3

### Supported Primitives

| Command | Format | Description |
|---------|--------|-------------|
| `WRITE` | `WRITE <addr> <value>` | Write a byte to memory |
| `DEFINE` | `DEFINE <label> <addr>` | Create symbolic address |
| `CALL` | `CALL <label>` | Documentation only |
| `COMMENT` | `COMMENT <text>` | Inline or full-line comment |

---

## Project Structure

```
pxos-v1.0/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── build_pxos.py            # Build system (converts primitives → binary)
├── pxos_commands.txt        # Primitive source code
├── pxos.bin                 # Bootable binary (512 bytes + padding)
├── runtime/                 # NEW: Shader VM Runtime
│   ├── README.md            # Shader VM documentation
│   ├── shader_vm.py         # VM instruction set & compiler
│   ├── shader_vm.wgsl       # WGSL VM runtime
│   ├── webgpu_runtime.py    # WebGPU integration
│   ├── language_frontends.py # Math DSL, Python, LOGO compilers
│   └── demo.py              # Comprehensive demo
├── tests/
│   ├── boot_qemu.sh         # Boot in QEMU
│   ├── boot_bochs.sh        # Boot in Bochs
│   └── test_input.sh        # Automated input testing
├── docs/
│   ├── architecture.md      # System design & memory layout
│   ├── primitives.md        # Primitive command reference
│   └── extensions.md        # How to extend pxOS
└── examples/
    └── hello_world_module.txt  # Example extension
```

---

## Testing

### QEMU (Recommended)

```bash
./tests/boot_qemu.sh
```

### Bochs

```bash
./tests/boot_bochs.sh
```

### Automated Input Test

```bash
./tests/test_input.sh
```

---

## Extending pxOS

See [docs/extensions.md](docs/extensions.md) for:

- Adding new commands
- Implementing command parser
- Loading modules from disk
- Upgrading to protected mode
- Adding FAT12 filesystem support

Example extension in [examples/hello_world_module.txt](examples/hello_world_module.txt)

---

## Development Roadmap

### ✅ v1.0 (Current)
- [x] Bootable shell
- [x] Character echo
- [x] Primitive-based build system
- [x] Documentation

### 🚧 v1.1 (Planned)
- [ ] Command parser (recognize typed commands)
- [ ] Help command
- [ ] Clear screen command
- [ ] Backspace support

### 🔮 v2.0 (Future)
- [ ] FAT12 driver (read files from disk)
- [ ] Module loading system
- [ ] Protected mode (32-bit)
- [ ] NASM assembly generator (auto-convert primitives)

---

## Technical Details

### Boot Process

1. BIOS loads sector 1 (512 bytes) to `0x7C00`
2. CPU jumps to `0x7C00` (boot_start)
3. Setup: CLI, stack at `0x9000:0xFFFF`, STI
4. Clear VGA text buffer (`0xB800:0000`)
5. Print welcome message via BIOS interrupt
6. Enter infinite keyboard loop

### Character Output

Uses BIOS interrupt `0x10`, function `0x0E` (teletype):
- `AH = 0x0E`: teletype output
- `AL = character`: character to print
- Automatically advances cursor

### Keyboard Input

Uses BIOS interrupt `0x16`, function `0x00`:
- `AH = 0x00`: wait for keypress
- Returns: `AL = ASCII`, `AH = scan code`

---

## System Requirements

### To Build
- Python 3.6+
- Text editor

### To Run
- x86 PC (or emulator)
- QEMU, Bochs, VirtualBox, or real hardware
- 32KB RAM minimum

### Optional Tools
- `qemu-system-i386` — Testing
- `genisoimage` — ISO creation
- `expect` — Automated testing

---

## FAQ

**Q: Can this boot on real hardware?**
A: Yes! Write `pxos.bin` to a USB drive with `dd` and boot from it.

**Q: Why not use NASM/FASM/etc?**
A: The primitive system is educational and makes every byte explicit. You can convert to NASM if you want (see [docs/extensions.md](docs/extensions.md)).

**Q: Is this a "real" OS?**
A: It's a minimal bootloader with a shell. No multitasking, memory management, or filesystem yet—but it's a foundation!

**Q: How do I add commands?**
A: Currently it just echoes. See [docs/extensions.md](docs/extensions.md) for adding a command parser.

**Q: Can I boot this in VirtualBox/VMware?**
A: Yes! Attach `pxos.bin` as a floppy disk image and boot from it.

---

## Contributing

Ideas for contributions:

- 🐛 Bug fixes
- 📝 Documentation improvements
- ✨ New primitive commands
- 🔧 Command parser implementation
- 🎨 Better welcome screen
- 🧪 More test cases
- 📦 Module system design

---

## License

MIT License — See [LICENSE](LICENSE) file

---

## Credits

**pxOS** is an educational project demonstrating minimal OS development with a unique primitive-based build system.

Built with inspiration from:
- [OSDev Wiki](https://wiki.osdev.org/)
- Classic bootloader tutorials
- Bare metal programming community

---

## Resources

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **Build System**: [build_pxos.py](build_pxos.py)
- **Source Code**: [pxos_commands.txt](pxos_commands.txt)

---

**Made with ❤️ in real-mode assembly**

*"Every operating system starts with a single boot sector..."*
