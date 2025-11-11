# pxOS Architecture

This document describes the internal architecture, memory layout, and design decisions of pxOS v1.0.

---

## Overview

pxOS is a **real-mode x86 bootloader** with an interactive shell. It demonstrates:

- **Minimal design** — < 1KB of code
- **Educational approach** — Every byte is documented
- **BIOS-based I/O** — Uses standard BIOS interrupts
- **Primitive-based build** — Built from WRITE commands, not assembly

---

## Boot Sequence

```
┌─────────────────┐
│  BIOS Power-On  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Read Sector 1   │ ← Boot sector (512 bytes)
│ Load to 0x7C00  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Jump to 0x7C00  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ pxOS Entry      │
│ - Disable int   │
│ - Setup stack   │
│ - Enable int    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Clear Screen    │
│ Fill VGA buffer │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Print Welcome   │
│ "pxOS v1> "     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Shell Loop      │
│ - Read key      │
│ - Echo char     │
│ - Repeat        │
└─────────────────┘
```

---

## Memory Map

### Physical Memory Layout

```
0x00000000  ┌──────────────────────────┐
            │ BIOS Data Area           │
0x00000500  ├──────────────────────────┤
            │ Available RAM            │
0x00007C00  ├──────────────────────────┤
            │ Boot Sector (pxOS)       │ ← BIOS loads here
0x00007E00  ├──────────────────────────┤
            │ Available RAM            │
0x00090000  ├──────────────────────────┤
            │ Stack (grows down)       │
0x000A0000  ├──────────────────────────┤
            │ Video RAM / BIOS ROM     │
0x00100000  └──────────────────────────┘
```

### pxOS Code Layout (0x7C00 - 0x7DFF)

| Offset | Address | Label         | Size | Description                    |
|--------|---------|---------------|------|--------------------------------|
| 0x00   | 0x7C00  | boot_start    | 26   | Entry, stack setup, clear      |
| 0x1A   | 0x7C1A  | -             | 6    | Print welcome, jump to shell   |
| 0x28   | 0x7C28  | print_string  | 12   | Print null-terminated string   |
| 0x38   | 0x7C38  | shell_loop    | 33   | Keyboard read and echo         |
| 0x40   | 0x7C40  | welcome_msg   | 10   | "pxOS v1> " + null             |
| 0xFE   | 0x7DFE  | boot_sig      | 2    | 0x55 0xAA (required)           |

### VGA Text Buffer

```
0xB8000 ┌────────────────────────────┐
        │ Character + Attribute      │
        │ 80 columns × 25 rows       │
        │ = 2000 words (4000 bytes)  │
        │                            │
        │ Format: [char][attr]       │
        │   char: ASCII              │
        │   attr: color/blink        │
0xB8FA0 └────────────────────────────┘
```

**Attribute Byte Format:**
```
 7  6  5  4  3  2  1  0
┌──┬────────┬───────────┐
│BL│  BG   │    FG     │
└──┴────────┴───────────┘
BL = Blink (1 bit)
BG = Background color (3 bits)
FG = Foreground color (4 bits)
```

Used in pxOS: `0x07` = Light gray on black

---

## Code Sections

### 1. Boot Loader (0x7C00 - 0x7C19)

**Purpose**: System initialization

**Operations**:
1. `CLI` - Disable interrupts (safety)
2. Set stack segment (SS) to `0x9000`
3. Set stack pointer (SP) to `0xFFFF` (64KB stack)
4. `STI` - Re-enable interrupts
5. Set ES to `0xB800` (video memory)
6. Clear screen: Fill 2000 words with `0x0720` (space + attribute)
7. Print welcome message
8. Jump to shell

**Assembly Equivalent**:
```nasm
cli
mov ax, 0x9000
mov ss, ax
mov sp, 0xFFFF
sti

mov ax, 0xB800
mov es, ax
xor di, di
mov cx, 2000
mov ax, 0x0720
rep stosw

mov si, welcome_msg
call print_string
jmp shell_loop
```

---

### 2. print_string (0x7C28 - 0x7C33)

**Purpose**: Print null-terminated string

**Parameters**:
- `SI` = pointer to string

**Algorithm**:
```
loop:
    AL = [SI++]          ; Load byte
    if AL == 0: return   ; Null terminator?
    AH = 0x0E            ; BIOS teletype
    int 0x10             ; Print character
    goto loop
```

**BIOS Interrupt Used**:
- `INT 0x10, AH=0x0E`: Teletype output
  - Input: `AL` = character
  - Automatically advances cursor

---

### 3. shell_loop (0x7C38 - 0x7C58)

**Purpose**: Interactive keyboard echo

**Algorithm**:
```
loop:
    AH = 0x00
    int 0x16            ; Read key → AL

    if AL == 0x0D:      ; Enter key?
        print('\r')     ; Carriage return
        print('\n')     ; Line feed
        print prompt
    else:
        AH = 0x0E
        int 0x10        ; Echo character

    goto loop
```

**BIOS Interrupts Used**:
- `INT 0x16, AH=0x00`: Wait for keypress
  - Output: `AL` = ASCII code, `AH` = scan code
- `INT 0x10, AH=0x0E`: Print character

---

## CPU State

### Registers at Boot

| Register | Value    | Purpose                    |
|----------|----------|----------------------------|
| CS       | 0x0000   | Code segment               |
| DS       | 0x0000   | Data segment               |
| ES       | varies   | Extra segment (we set it)  |
| SS       | varies   | Stack segment (we set it)  |
| SP       | varies   | Stack pointer (we set it)  |
| IP       | 0x7C00   | Instruction pointer (BIOS) |

### Real Mode Addressing

Address = (Segment × 16) + Offset

Examples:
- `0x7C00:0x0000` = Physical `0x7C00`
- `0x0000:0x7C00` = Physical `0x7C00` (same)
- `0xB800:0x0000` = Physical `0xB8000` (VGA)

---

## BIOS Interrupts Used

### INT 0x10 - Video Services

| Function | AH  | Description      | Inputs         | Outputs |
|----------|-----|------------------|----------------|---------|
| Teletype | 0x0E| Print character  | AL=char        | -       |

Features:
- Automatic cursor advancement
- Handles special chars: `\r`, `\n`, `\b`, `\t`, `\a`
- Scrolls screen when cursor reaches bottom

### INT 0x16 - Keyboard Services

| Function | AH  | Description     | Inputs | Outputs             |
|----------|-----|-----------------|--------|---------------------|
| Read key | 0x00| Wait for key    | -      | AL=ASCII, AH=scan   |

Behavior:
- Blocks until key is pressed
- Returns both ASCII code and scan code
- Does NOT echo (we do that manually)

---

## Stack

**Location**: `0x9000:0xFFFF`

**Physical Address**: `0x9FFFF` (approximately 640KB mark)

**Size**: Grows downward, 64KB available

**Usage**:
- Function calls: `CALL` pushes return address
- `RET` pops return address
- Currently minimal (only print_string uses it)

---

## Design Decisions

### Why Real Mode?

- **Simplicity**: No paging, no protected mode setup
- **BIOS access**: Can use interrupts directly
- **Educational**: Easier to understand for beginners
- **Minimal**: No need for GDT, IDT, etc.

### Why BIOS Interrupts?

- **No driver code needed**: BIOS provides keyboard, video
- **Portable**: Works on any x86 PC
- **Small**: Saves space in boot sector

### Why No Command Parser?

- **Size**: 512 bytes is very limited
- **Simplicity**: Focus on core boot and I/O
- **Extensibility**: Can be added in v1.1

### Why Primitives?

- **Educational**: See every byte explicitly
- **Debugging**: Easy to trace problems
- **Minimal tools**: Just Python, no assembler
- **Unique**: Different approach than traditional

---

## Limitations

### Current (v1.0)

- ❌ No command recognition
- ❌ No backspace support
- ❌ No cursor positioning control
- ❌ No color control
- ❌ No disk I/O
- ❌ Single boot sector only

### Real Mode Limits

- ❌ Max 1MB addressable memory
- ❌ No memory protection
- ❌ No multitasking
- ❌ 16-bit registers only

---

## Future Architecture

### v1.1 Enhancements

```
┌──────────────┐
│ Shell Loop   │
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐
│ Input Buffer │────▶│ Parser       │
└──────────────┘     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ Jump Table   │
                     └──────┬───────┘
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
┌──────────┐         ┌──────────┐        ┌──────────┐
│ cmd_help │         │ cmd_cls  │        │ cmd_echo │
└──────────┘         └──────────┘        └──────────┘
```

### v2.0 Protected Mode

- Load GDT (Global Descriptor Table)
- Switch to 32-bit protected mode
- Access > 1MB memory
- Implement multitasking

---

## References

- [OSDev Wiki - Boot Sequence](https://wiki.osdev.org/Boot_Sequence)
- [OSDev Wiki - Real Mode](https://wiki.osdev.org/Real_Mode)
- [BIOS Interrupt Calls](https://en.wikipedia.org/wiki/BIOS_interrupt_call)
- [x86 Instruction Reference](https://www.felixcloutier.com/x86/)

---

**Next**: [primitives.md](primitives.md) — Primitive command reference
