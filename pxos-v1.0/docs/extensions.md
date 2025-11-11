# Extending pxOS

This guide shows how to extend pxOS with new features, commands, and capabilities.

---

## Quick Extension Ideas

| Difficulty | Feature | Lines | Description |
|------------|---------|-------|-------------|
| ⭐ Easy | **Backspace** | ~10 | Handle 0x08, move cursor back |
| ⭐ Easy | **Color change** | ~5 | Modify attribute byte |
| ⭐⭐ Medium | **Command parser** | ~50 | Recognize "help", "cls", etc. |
| ⭐⭐ Medium | **Help command** | ~30 | Print available commands |
| ⭐⭐⭐ Hard | **FAT12 reader** | ~200 | Read files from floppy |
| ⭐⭐⭐ Hard | **Protected mode** | ~100 | Enter 32-bit mode |

---

## Extension 1: Backspace Support

### Goal
Handle backspace key (0x08) to erase characters.

### Implementation

Add to pxos_commands.txt after keyboard read:

```
COMMENT Check for backspace
WRITE 0x7C3C 0x3C           COMMENT CMP AL, 0x08
WRITE 0x7C3D 0x08
WRITE 0x7C3E 0x74           COMMENT JE handle_backspace
WRITE 0x7C3F 0x0A

COMMENT ... (rest of shell code)

COMMENT Handle backspace
DEFINE handle_backspace 0x7C5A
WRITE 0x7C5A 0xB4           COMMENT MOV AH, 0x0E
WRITE 0x7C5B 0x0E
WRITE 0x7C5C 0xB0           COMMENT MOV AL, 0x08 (backspace)
WRITE 0x7C5D 0x08
WRITE 0x7C5E 0xCD           COMMENT INT 0x10 (move cursor back)
WRITE 0x7C5F 0x10
WRITE 0x7C60 0xB0           COMMENT MOV AL, ' ' (space)
WRITE 0x7C61 0x20
WRITE 0x7C62 0xCD           COMMENT INT 0x10 (erase character)
WRITE 0x7C63 0x10
WRITE 0x7C64 0xB0           COMMENT MOV AL, 0x08 (backspace again)
WRITE 0x7C65 0x08
WRITE 0x7C66 0xCD           COMMENT INT 0x10 (move cursor back)
WRITE 0x7C67 0x10
WRITE 0x7C68 0xE9           COMMENT JMP shell_loop
WRITE 0x7C69 0xCD
WRITE 0x7C6A 0xFF
```

**Result**: Backspace now erases characters!

---

## Extension 2: Command Parser

### Goal
Recognize typed commands like "help", "cls", "echo".

### Design

```
┌─────────────┐
│ Shell Loop  │
│ Read chars  │
└──────┬──────┘
       │
       ▼ (Enter pressed)
┌──────────────┐
│ Parse Buffer │
│ Compare cmds │
└──────┬───────┘
       │
       ├──▶ "help" ──▶ Show help
       ├──▶ "cls"  ──▶ Clear screen
       ├──▶ "echo" ──▶ Echo args
       └──▶ unknown ──▶ "Unknown command"
```

### Memory Layout

```
0x7E00  input_buffer[80]   Command input
0x7E50  cmd_help[5]        "help\0"
0x7E55  cmd_cls[4]         "cls\0"
0x7E59  cmd_echo[5]        "echo\0"
```

### Pseudo-code

```nasm
shell_loop:
    ; Clear buffer
    mov di, input_buffer
    mov cx, 80
    xor al, al
    rep stosb

    ; Read line
.read_loop:
    mov ah, 0x00
    int 0x16                ; Read key
    cmp al, 0x0D            ; Enter?
    je .parse_command
    cmp al, 0x08            ; Backspace?
    je .handle_backspace
    ; Store and echo
    stosb                   ; Store in buffer
    mov ah, 0x0E
    int 0x10                ; Echo
    jmp .read_loop

.parse_command:
    ; Print newline
    mov al, 0x0D
    call print_char
    mov al, 0x0A
    call print_char

    ; Compare with "help"
    mov si, input_buffer
    mov di, cmd_help
    mov cx, 4
    repe cmpsb
    je do_help

    ; Compare with "cls"
    mov si, input_buffer
    mov di, cmd_cls
    mov cx, 3
    repe cmpsb
    je do_cls

    ; Unknown command
    mov si, msg_unknown
    call print_string
    jmp shell_loop
```

### Implementation Size
~100 bytes of code + command strings

---

## Extension 3: Help Command

### Goal
Print available commands when user types "help".

### Implementation

```
DEFINE do_help 0x7D00
DEFINE help_text 0x7D50

COMMENT Print help text
WRITE 0x7D00 0xBE           COMMENT MOV SI, help_text
WRITE 0x7D01 0x50
WRITE 0x7D02 0x7D
WRITE 0x7D03 0xE8           COMMENT CALL print_string
WRITE 0x7D04 0x23
WRITE 0x7D05 0xFF
WRITE 0x7D06 0xE9           COMMENT JMP shell_loop
WRITE 0x7D07 0x2E
WRITE 0x7D08 0xFF

COMMENT Help text
WRITE 0x7D50 0x0D           COMMENT \r
WRITE 0x7D51 0x0A           COMMENT \n
WRITE 0x7D52 0x41           COMMENT 'A'
WRITE 0x7D53 0x76           COMMENT 'v'
WRITE 0x7D54 0x61           COMMENT 'a'
WRITE 0x7D55 0x69           COMMENT 'i'
WRITE 0x7D56 0x6C           COMMENT 'l'
WRITE 0x7D57 0x61           COMMENT 'a'
WRITE 0x7D58 0x62           COMMENT 'b'
WRITE 0x7D59 0x6C           COMMENT 'l'
WRITE 0x7D5A 0x65           COMMENT 'e'
WRITE 0x7D5B 0x20           COMMENT ' '
WRITE 0x7D5C 0x63           COMMENT 'c'
WRITE 0x7D5D 0x6F           COMMENT 'o'
WRITE 0x7D5E 0x6D           COMMENT 'm'
WRITE 0x7D5F 0x6D           COMMENT 'm'
WRITE 0x7D60 0x61           COMMENT 'a'
WRITE 0x7D61 0x6E           COMMENT 'n'
WRITE 0x7D62 0x64           COMMENT 'd'
WRITE 0x7D63 0x73           COMMENT 's'
WRITE 0x7D64 0x3A           COMMENT ':'
WRITE 0x7D65 0x0D           COMMENT \r
WRITE 0x7D66 0x0A           COMMENT \n
WRITE 0x7D67 0x68           COMMENT 'h'
WRITE 0x7D68 0x65           COMMENT 'e'
WRITE 0x7D69 0x6C           COMMENT 'l'
WRITE 0x7D6A 0x70           COMMENT 'p'
WRITE 0x7D6B 0x20           COMMENT ' '
WRITE 0x7D6C 0x2D           COMMENT '-'
WRITE 0x7D6D 0x20           COMMENT ' '
WRITE 0x7D6E 0x53           COMMENT 'S'
... (continue for full help text)
WRITE 0x7D90 0x00           COMMENT Null terminator
```

**Tip**: Use a Python script to generate string bytes:

```python
text = "Available commands:\nhelp - Show this help\ncls - Clear screen\n"
addr = 0x7D50
for char in text:
    print(f"WRITE 0x{addr:04X} 0x{ord(char):02X}   COMMENT '{char}'")
    addr += 1
print(f"WRITE 0x{addr:04X} 0x00           COMMENT Null terminator")
```

---

## Extension 4: Clear Screen Command

### Goal
Clear screen when user types "cls".

### Implementation

Reuse existing clear screen code from boot:

```
DEFINE do_cls 0x7D20

WRITE 0x7D20 0xB8           COMMENT MOV AX, 0xB800
WRITE 0x7D21 0x00
WRITE 0x7D22 0xB8
WRITE 0x7D23 0x8E           COMMENT MOV ES, AX
WRITE 0x7D24 0xC0
WRITE 0x7D25 0x31           COMMENT XOR DI, DI
WRITE 0x7D26 0xFF
WRITE 0x7D27 0xB9           COMMENT MOV CX, 2000
WRITE 0x7D28 0xD0
WRITE 0x7D29 0x07
WRITE 0x7D2A 0xB0           COMMENT MOV AL, ' '
WRITE 0x7D2B 0x20
WRITE 0x7D2C 0xB4           COMMENT MOV AH, 0x07
WRITE 0x7D2D 0x07
WRITE 0x7D2E 0xF3           COMMENT REP STOSW
WRITE 0x7D2F 0xAB
WRITE 0x7D30 0xE9           COMMENT JMP shell_loop
WRITE 0x7D31 0x05
WRITE 0x7D32 0xFF
```

---

## Extension 5: Loading from Disk

### Goal
Load additional sectors from floppy disk.

### BIOS INT 0x13 - Disk Services

**Function 0x02**: Read sectors

| Register | Value |
|----------|-------|
| AH | 0x02 (read) |
| AL | Number of sectors |
| CH | Cylinder (0-based) |
| CL | Sector (1-based) |
| DH | Head (0 or 1) |
| DL | Drive (0x00=floppy A) |
| ES:BX | Buffer address |

### Example: Load Sector 2

```nasm
DEFINE load_sector2 0x7D40

load_sector2:
    mov ah, 0x02            ; Read sectors
    mov al, 1               ; 1 sector
    mov ch, 0               ; Cylinder 0
    mov cl, 2               ; Sector 2
    mov dh, 0               ; Head 0
    mov dl, 0               ; Drive A:
    mov bx, 0x8000          ; Load to 0x8000
    int 0x13                ; BIOS disk read
    jc .error               ; Check carry flag
    ret
.error:
    ; Handle error (AH = error code)
    ret
```

### Primitive Implementation

```
DEFINE load_sector2 0x7D40
WRITE 0x7D40 0xB4           COMMENT MOV AH, 0x02
WRITE 0x7D41 0x02
WRITE 0x7D42 0xB0           COMMENT MOV AL, 1
WRITE 0x7D43 0x01
WRITE 0x7D44 0xB5           COMMENT MOV CH, 0
WRITE 0x7D45 0x00
WRITE 0x7D46 0xB1           COMMENT MOV CL, 2
WRITE 0x7D47 0x02
WRITE 0x7D48 0xB6           COMMENT MOV DH, 0
WRITE 0x7D49 0x00
WRITE 0x7D4A 0xB2           COMMENT MOV DL, 0
WRITE 0x7D4B 0x00
WRITE 0x7D4C 0xBB           COMMENT MOV BX, 0x8000
WRITE 0x7D4D 0x00
WRITE 0x7D4E 0x80
WRITE 0x7D4F 0xCD           COMMENT INT 0x13
WRITE 0x7D50 0x13
WRITE 0x7D51 0x72           COMMENT JC error
WRITE 0x7D52 0x02
WRITE 0x7D53 0xC3           COMMENT RET
WRITE 0x7D54 0xC3           COMMENT error: RET
```

---

## Extension 6: Multi-Sector Boot

### Goal
Boot sector loads additional sectors automatically.

### Disk Layout

```
Sector 1: Boot sector (512 bytes) ← BIOS loads this
Sector 2: Extended code (512 bytes)
Sector 3: Data/strings (512 bytes)
Sector 4+: Modules
```

### Build Multi-Sector Image

```bash
# Build boot sector
python3 build_pxos.py

# Create sector 2 (extended code)
python3 build_sector2.py > sector2.bin

# Concatenate
cat pxos.bin sector2.bin > pxos_multi.bin
```

### Boot Sector Modification

Add at end of boot sector (before signature):

```
COMMENT Load sector 2
WRITE 0x7DF0 0xB4           COMMENT MOV AH, 0x02
WRITE 0x7DF1 0x02
WRITE 0x7DF2 0xB0           COMMENT MOV AL, 1
WRITE 0x7DF3 0x01
WRITE 0x7DF4 0xB5           COMMENT MOV CH, 0
WRITE 0x7DF5 0x00
WRITE 0x7DF6 0xB1           COMMENT MOV CL, 2
WRITE 0x7DF7 0x02
WRITE 0x7DF8 0xB6           COMMENT MOV DH, 0
WRITE 0x7DF9 0x00
WRITE 0x7DFA 0xB2           COMMENT MOV DL, 0
WRITE 0x7DFB 0x00
WRITE 0x7DFC 0xBB           COMMENT MOV BX, 0x8000
WRITE 0x7DFD 0x00
WRITE 0x7DFE 0x80
WRITE 0x7DFF 0xCD           COMMENT INT 0x13
WRITE 0x7E00 0x13
WRITE 0x7E01 0xE9           COMMENT JMP 0x8000 (sector 2 code)
WRITE 0x7E02 0xFC
WRITE 0x7E03 0x01
```

---

## Extension 7: Protected Mode

### Goal
Switch from 16-bit real mode to 32-bit protected mode.

### Requirements

1. **GDT** (Global Descriptor Table)
2. **Disable interrupts**
3. **Enable A20 line**
4. **Load GDT**
5. **Set PE bit in CR0**
6. **Far jump to 32-bit code**

### Minimal GDT

```nasm
gdt_start:
    ; Null descriptor
    dq 0

    ; Code segment descriptor
    dw 0xFFFF           ; Limit (low)
    dw 0x0000           ; Base (low)
    db 0x00             ; Base (middle)
    db 10011010b        ; Access (present, code, execute/read)
    db 11001111b        ; Granularity (4KB blocks) + Limit (high)
    db 0x00             ; Base (high)

    ; Data segment descriptor
    dw 0xFFFF           ; Limit (low)
    dw 0x0000           ; Base (low)
    db 0x00             ; Base (middle)
    db 10010010b        ; Access (present, data, read/write)
    db 11001111b        ; Granularity + Limit (high)
    db 0x00             ; Base (high)

gdt_descriptor:
    dw gdt_descriptor - gdt_start - 1   ; Size
    dd gdt_start                         ; Offset
```

### Switch Code

```nasm
    cli                     ; Disable interrupts
    lgdt [gdt_descriptor]   ; Load GDT
    mov eax, cr0
    or al, 1                ; Set PE bit
    mov cr0, eax
    jmp 0x08:protected_mode ; Far jump (0x08 = code selector)

[BITS 32]
protected_mode:
    mov ax, 0x10            ; Data selector
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    ; Now in 32-bit mode!
```

**Note**: This requires > 512 bytes. Use multi-sector boot.

---

## Extension 8: NASM Converter

### Goal
Auto-generate NASM assembly from primitives.

### Python Script

```python
#!/usr/bin/env python3
"""Convert pxOS primitives to NASM assembly"""

with open('pxos_commands.txt', 'r') as f:
    for line in f:
        line = line.strip()

        if line.startswith('COMMENT'):
            comment = line[8:].strip()
            print(f"; {comment}")

        elif line.startswith('DEFINE'):
            parts = line.split()
            label = parts[1]
            print(f"{label}:")

        elif line.startswith('WRITE'):
            parts = line.split('COMMENT')
            cmd = parts[0].strip().split()
            addr = cmd[1]
            value = cmd[2]
            comment = parts[1].strip() if len(parts) > 1 else ""
            print(f"    db {value}    ; {comment}")
```

**Output**:
```nasm
; pxOS Boot Sector
boot_start:
    db 0xFA    ; CLI
    db 0xB8    ; MOV AX, 0x9000
    db 0x00
    db 0x90
...
```

---

## Tools and Resources

### Assemblers
- **NASM**: Modern, widely used
- **FASM**: Fast, included IDE
- **GAS**: GNU Assembler (AT&T syntax)

### Emulators
- **QEMU**: Fast, scriptable
- **Bochs**: Detailed debugging
- **VirtualBox**: Full virtualization

### Debuggers
- **GDB**: With QEMU remote debugging
- **Bochs debugger**: Built-in, powerful
- **ndisasm**: Disassembler (included with NASM)

### Hexdump Tools
```bash
hexdump -C pxos.bin          # Canonical hex
od -t x1z pxos.bin           # Octal dump
xxd pxos.bin                 # Vim-style hex
```

---

## Example Projects

### 1. Calculator
Add basic arithmetic: `calc 2+2` → `4`

### 2. Text Editor
Line-based editor with save/load

### 3. Game
Snake, Tetris, or Pong in 16-bit

### 4. Network Boot
PXE boot loader

### 5. Multi-Boot Menu
Select from multiple OS images

---

## Further Reading

- [OSDev Wiki](https://wiki.osdev.org/)
- [Writing a Simple Operating System from Scratch](https://www.cs.bham.ac.uk/~exr/lectures/opsys/10_11/lectures/os-dev.pdf)
- [Linux Insides](https://0xax.gitbooks.io/linux-insides/)
- [Bran's Kernel Development Tutorial](http://www.osdever.net/bkerndev/Docs/title.htm)

---

**Need help?** See [examples/hello_world_module.txt](../examples/hello_world_module.txt) for a complete extension example.
