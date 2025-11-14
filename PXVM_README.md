# pxVM - Bytecode VM for pxOS

pxVM is a minimal bytecode virtual machine that integrates with the pxOS graphics system via syscalls.

## Architecture

```
┌─────────────────────────────────────┐
│ pxVM Bytecode Program               │
├─────────────────────────────────────┤
│ pxVM Interpreter (pxvm.py)          │
│ - 8 registers (R0-R7)               │
│ - 64KB memory                       │
│ - Syscalls emit PXTERM              │
├─────────────────────────────────────┤
│ PXTERM Instructions                 │
├─────────────────────────────────────┤
│ pxos_llm_terminal.py                │
├─────────────────────────────────────┤
│ PxOSTerminalGPU (layered rendering) │
├─────────────────────────────────────┤
│ Frozen WGSL Shader → GPU → PNG      │
└─────────────────────────────────────┘
```

## Features

**v0.1 Syscalls:**
- `SYS_PRINT_ID (1)` - Print predefined messages
- `SYS_RECT_ID (2)` - Draw rectangles with color IDs
- `SYS_TEXT_ID (3)` - Render text with message IDs
- `SYS_LAYER_USE_ID (4)` - Switch rendering layers

**Imperfect Computing:**
- Unknown syscalls → warning, continue
- Invalid args → safe defaults
- VM errors logged, never crash host

## Quick Start

### 1. Write a Program (Python API)

```python
from pxvm_asm import ProgramBuilder

prog = ProgramBuilder()

# Print boot message
prog.sys_print(1)  # "PXVM booting..."

# Switch to VM layer
prog.sys_layer(3)

# Draw window
prog.sys_rect(150, 150, 500, 300, 1)  # x, y, w, h, color_id

# Draw title bar
prog.sys_rect(150, 150, 500, 40, 2)

# Add title
prog.sys_text(170, 160, 4, 2)  # x, y, color_id, message_id

prog.halt()

# Save bytecode
prog.save("my_program.pxvm")
```

### 2. Run the Program

```bash
python pxvm_run.py my_program.pxvm
# Output: my_program.png
```

## Opcodes

### Data Movement

- `0x10 IMM8 reg, val` - Load 8-bit immediate
- `0x11 IMM32 reg, val` - Load 32-bit immediate
- `0x20 MOV dst, src` - Copy register

### Arithmetic

- `0x30 ADD dst, src1, src2` - Add registers
- `0x31 SUB dst, src1, src2` - Subtract registers

### System

- `0xF0 SYSCALL num` - Invoke system call
- `0x00 HALT` - Stop execution
- `0x01 NOP` - No operation

## Syscall Details

### SYS_PRINT_ID (1)

Print a predefined message to console.

**Args:**
- `R1` = message_id

**Message IDs:**
```
1 = "PXVM booting..."
2 = "PXVM ready."
3 = "Task complete."
4 = "Kernel init done."
5 = "Process started."
6 = "Process terminated."
```

**Example:**
```python
prog.imm32(1, 1)      # R1 = 1
prog.syscall(1)       # SYS_PRINT_ID
```

### SYS_RECT_ID (2)

Draw a filled rectangle.

**Args:**
- `R1` = x position
- `R2` = y position
- `R3` = width
- `R4` = height
- `R5` = color_id

**Color IDs:**
```
1 = (40, 40, 100, 255)   # window frame
2 = (20, 20, 60, 255)    # title bar
3 = (0, 0, 40, 255)      # background
4 = (255, 255, 255, 255) # white text
5 = (200, 200, 255, 255) # light blue text
```

**Example:**
```python
prog.imm32(1, 100)    # x = 100
prog.imm32(2, 100)    # y = 100
prog.imm32(3, 200)    # w = 200
prog.imm32(4, 150)    # h = 150
prog.imm32(5, 1)      # color_id = 1
prog.syscall(2)       # SYS_RECT_ID
```

### SYS_TEXT_ID (3)

Render text at a position.

**Args:**
- `R1` = x position
- `R2` = y position
- `R3` = color_id
- `R4` = message_id

**Example:**
```python
prog.imm32(1, 120)    # x = 120
prog.imm32(2, 150)    # y = 150
prog.imm32(3, 4)      # color_id = 4 (white)
prog.imm32(4, 2)      # message_id = 2 ("PXVM ready.")
prog.syscall(3)       # SYS_TEXT_ID
```

### SYS_LAYER_USE_ID (4)

Switch the active rendering layer.

**Args:**
- `R1` = layer_id

**Layer IDs:**
```
1 = "background"
2 = "ui"
3 = "vm"
4 = "overlay"
```

**Example:**
```python
prog.imm32(1, 3)      # layer_id = 3 ("vm")
prog.syscall(4)       # SYS_LAYER_USE_ID
```

## Example Programs

### Hello World

```python
from pxvm_asm import build_hello_program

bytecode = build_hello_program()
with open("hello.pxvm", "wb") as f:
    f.write(bytecode)
```

Run:
```bash
python pxvm_run.py hello.pxvm
```

### Complex Window

```python
from pxvm_asm import build_window_program

bytecode = build_window_program()
with open("window.pxvm", "wb") as f:
    f.write(bytecode)
```

Run:
```bash
python pxvm_run.py window.pxvm
```

## Files

- `pxvm.py` - VM interpreter core
- `pxvm_asm.py` - Assembler / program builder
- `pxvm_run.py` - Runner that connects VM → PXTERM → GPU
- `PXVM_SYSCALL_SPEC.md` - Detailed syscall specification
- `examples/` - Example programs

## Imperfect Mode

pxVM inherits pxOS's imperfect computing philosophy:

**Never crashes on:**
- Unknown syscalls → warning comment
- Invalid register indices → ignored
- Out-of-range coordinates → clipped by drawing layer
- Unknown message/color/layer IDs → fallback values

**Example:**
```python
prog.syscall(99)  # unknown syscall
# Emits: # WARNING: unknown syscall 99 with args R1-R7=[...]
# VM continues execution
```

## Extending the VM

### Adding New Syscalls

1. Define syscall number in `pxvm.py`
2. Add handler in `PxVM.handle_syscall()`
3. Update `PXVM_SYSCALL_SPEC.md`

### Adding New Opcodes

1. Define opcode constant in `pxvm.py`
2. Add case in `PxVM.step()`
3. Add assembler helper in `pxvm_asm.py`

### Adding Messages/Colors/Layers

Edit lookup tables in `PxVM.__init__()`:
- `self.sys_messages`
- `self.sys_colors`
- `self.sys_layers`

## Future Extensions (v0.2+)

- **String syscalls**: SYS_PRINT_STR, SYS_TEXT_STR from memory
- **Dynamic layers**: SYS_LAYER_CREATE
- **Frame control**: SYS_DRAW to trigger renders
- **Input**: SYS_INPUT for keyboard/mouse
- **Process management**: SYS_PROCESS_*
- **Memory management**: Heap allocation
- **Jump/Branch**: Conditional execution

## Performance

- VM is interpreted Python (not JIT)
- Suitable for UI generation, not computation
- Syscalls emit PXTERM (text format)
- Rendering happens in GPU terminal (fast)

Typical program: ~100-300 bytes, renders in <100ms.

## License

Same as pxOS (see LICENSE file).
