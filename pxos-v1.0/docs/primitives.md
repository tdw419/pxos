# pxOS Primitive Command Reference

This document describes the primitive command language used to build pxOS.

---

## Overview

The pxOS primitive system is a **domain-specific language (DSL)** for directly manipulating memory without requiring a traditional assembler. It provides a transparent, educational approach to bootloader development.

---

## Command Syntax

### General Format

```
COMMAND <argument1> <argument2> ... COMMENT optional description
```

**Rules**:
- Commands are case-insensitive (`WRITE` = `write`)
- Arguments are space-separated
- Inline comments start with `COMMENT`
- Full-line comments start with `COMMENT` or `#`
- Blank lines are ignored

---

## Commands

### WRITE

**Purpose**: Write a single byte to memory

**Syntax**:
```
WRITE <address> <value> [COMMENT description]
```

**Parameters**:
- `address`: Memory address (hex or decimal)
- `value`: Byte value 0x00-0xFF (hex or decimal)

**Examples**:
```
WRITE 0x7C00 0xFA                    COMMENT cli instruction
WRITE 0x7C01 0xB8                    COMMENT mov ax, ... (part 1)
WRITE 31744 250                      COMMENT Same as 0x7C00 0xFA (decimal)
```

**Notes**:
- Address must be < 0x10000 (64KB limit in builder)
- Value must be 0x00-0xFF (single byte)
- Writes are immediate (no delayed evaluation)
- Later writes to same address overwrite previous

**Use Cases**:
- Writing opcodes (machine code bytes)
- Writing data (strings, numbers, tables)
- Filling memory regions

---

### DEFINE

**Purpose**: Create a symbolic label for an address

**Syntax**:
```
DEFINE <label> <address> [COMMENT description]
```

**Parameters**:
- `label`: Symbol name (alphanumeric, no spaces)
- `address`: Memory address (hex or decimal)

**Examples**:
```
DEFINE boot_start 0x7C00             COMMENT Boot sector entry
DEFINE print_string 0x7C28           COMMENT Function address
DEFINE video_mem 0xB800              COMMENT VGA text buffer
```

**Notes**:
- Labels can be used in subsequent WRITE commands (future enhancement)
- Labels are case-sensitive by convention (use lowercase)
- No duplicate labels (later definitions overwrite)
- Address must be < 0x10000

**Use Cases**:
- Marking function entry points
- Documenting memory regions
- Creating jump tables
- Symbol table generation

---

### CALL

**Purpose**: Documentation of function calls (currently no-op)

**Syntax**:
```
CALL <label> [COMMENT description]
```

**Parameters**:
- `label`: Function label (defined via DEFINE)

**Examples**:
```
CALL print_string                    COMMENT Print welcome message
CALL clear_screen                    COMMENT Initialize display
```

**Notes**:
- Currently only for documentation
- Does not generate code
- Future: May resolve to WRITE commands for call instructions
- Use in comments to document control flow

---

### COMMENT

**Purpose**: Add human-readable documentation

**Syntax**:
```
COMMENT <any text>
```

**Forms**:

**Full-line comment**:
```
COMMENT ============================================
COMMENT Boot Loader Section
COMMENT ============================================
```

**Inline comment**:
```
WRITE 0x7C00 0xFA    COMMENT CLI instruction (disable interrupts)
```

**Empty comment**:
```
COMMENT
```

**Notes**:
- Everything after `COMMENT` is ignored
- Can appear at start of line or after command
- No nesting or multi-line support
- Alternative: `#` for full-line comments

---

## Value Formats

### Hexadecimal

```
0x7C00          COMMENT Prefix with 0x
0X7C00          COMMENT 0X also works
0x00 - 0xFFFF   COMMENT Range supported
```

### Decimal

```
31744           COMMENT No prefix = decimal
0 - 65535       COMMENT Range supported
```

### Symbol References (Future)

```
DEFINE start 0x7C00
WRITE start 0xFA        COMMENT Writes to 0x7C00 (future)
```

Currently symbols are only for documentation. Future versions may support symbolic addressing.

---

## Complete Example

```
COMMENT ========================================
COMMENT pxOS Minimal Boot Sector
COMMENT ========================================

DEFINE boot_start 0x7C00

COMMENT Initialize system
WRITE 0x7C00 0xFA           COMMENT CLI (disable interrupts)
WRITE 0x7C01 0xB8           COMMENT MOV AX, 0x9000
WRITE 0x7C02 0x00
WRITE 0x7C03 0x90
WRITE 0x7C04 0x8E           COMMENT MOV SS, AX
WRITE 0x7C05 0xD0
WRITE 0x7C06 0xBC           COMMENT MOV SP, 0xFFFF
WRITE 0x7C07 0xFF
WRITE 0x7C08 0xFF
WRITE 0x7C09 0xFB           COMMENT STI (enable interrupts)

COMMENT Define print function
DEFINE print_string 0x7C10

WRITE 0x7C10 0xAC           COMMENT LODSB
WRITE 0x7C11 0x08           COMMENT OR AL, AL
WRITE 0x7C12 0xC0
WRITE 0x7C13 0x74           COMMENT JZ done
WRITE 0x7C14 0x06
WRITE 0x7C15 0xB4           COMMENT MOV AH, 0x0E
WRITE 0x7C16 0x0E
WRITE 0x7C17 0xCD           COMMENT INT 0x10
WRITE 0x7C18 0x10
WRITE 0x7C19 0xEB           COMMENT JMP print_string
WRITE 0x7C1A 0xF5
WRITE 0x7C1B 0xC3           COMMENT RET

COMMENT Boot signature
COMMENT (Builder adds automatically at 0x1FE-0x1FF)
```

---

## Best Practices

### 1. Use Consistent Addressing

**Good**:
```
DEFINE print_char 0x0200
WRITE 0x0200 0xB4
WRITE 0x0201 0x0E
```

**Avoid**:
```
WRITE 0x0200 0xB4
WRITE 513 0x0E        COMMENT Mixing hex/decimal is confusing
```

### 2. Document Every Instruction

**Good**:
```
WRITE 0x7C00 0xFA     COMMENT CLI - disable interrupts for stack setup
```

**Avoid**:
```
WRITE 0x7C00 0xFA     COMMENT FA
```

### 3. Group Related Code

```
COMMENT ========================================
COMMENT Screen Clearing Routine
COMMENT ========================================
DEFINE clear_screen 0x0100

WRITE 0x0100 0xB8     COMMENT MOV AX, 0xB800
...
```

### 4. Use Symbol Definitions

**Good**:
```
DEFINE shell_loop 0x7C38
DEFINE input_buffer 0x7E00
DEFINE max_input 80
```

**Avoid**:
```
COMMENT Function at 0x7C38
COMMENT Buffer at 0x7E00
COMMENT Max 80 chars
```

### 5. Align Data Structures

```
COMMENT Jump table (aligned to 16-byte boundary)
DEFINE jump_table 0x0700
WRITE 0x0700 0x00     COMMENT cmd_cls low byte
WRITE 0x0701 0x08     COMMENT cmd_cls high byte
WRITE 0x0702 0x50     COMMENT cmd_help low byte
WRITE 0x0703 0x08     COMMENT cmd_help high byte
```

---

## Conversion Table

### Common x86 Instructions

| Assembly | Opcode | Primitive |
|----------|--------|-----------|
| `cli` | 0xFA | `WRITE addr 0xFA` |
| `sti` | 0xFB | `WRITE addr 0xFB` |
| `ret` | 0xC3 | `WRITE addr 0xC3` |
| `nop` | 0x90 | `WRITE addr 0x90` |
| `hlt` | 0xF4 | `WRITE addr 0xF4` |
| `int 0x10` | 0xCD 0x10 | `WRITE addr 0xCD` `WRITE addr+1 0x10` |
| `jmp short $` | 0xEB 0xFE | `WRITE addr 0xEB` `WRITE addr+1 0xFE` |

---

## Advanced Patterns

### Multi-Byte Instructions

```
COMMENT MOV AX, 0x1234 (3 bytes: B8 34 12)
WRITE 0x7C00 0xB8
WRITE 0x7C01 0x34     COMMENT Low byte first (little-endian)
WRITE 0x7C02 0x12     COMMENT High byte second
```

### Conditional Jumps

```
COMMENT JZ target (jump if zero)
COMMENT Opcode: 74 <offset>
WRITE 0x7C10 0x74
WRITE 0x7C11 0x05     COMMENT Jump forward 5 bytes
```

### String Data

```
DEFINE message 0x7E00
WRITE 0x7E00 0x48     COMMENT 'H'
WRITE 0x7E01 0x65     COMMENT 'e'
WRITE 0x7E02 0x6C     COMMENT 'l'
WRITE 0x7E03 0x6C     COMMENT 'l'
WRITE 0x7E04 0x6F     COMMENT 'o'
WRITE 0x7E05 0x00     COMMENT Null terminator
```

### Jump Tables

```
DEFINE jump_table 0x0700
COMMENT Entry 0: CLS command
WRITE 0x0700 0x00     COMMENT Address low byte
WRITE 0x0701 0x08     COMMENT Address high byte (0x0800)
COMMENT Entry 1: HELP command
WRITE 0x0702 0x50
WRITE 0x0703 0x08     COMMENT 0x0850
```

---

## Limitations

### Current

- No arithmetic expressions: `WRITE 0x7C00+2 0xFA` ❌
- No symbol references in WRITE: `WRITE start 0xFA` ❌
- No macros or includes
- No string literals: `WRITE 0x7E00 "Hello"` ❌
- No auto-addressing: Must specify every byte

### Workarounds

**Use Python for generation**:
```python
address = 0x7C00
for byte in [0xFA, 0xB8, 0x00, 0x90]:
    print(f"WRITE 0x{address:04X} 0x{byte:02X}")
    address += 1
```

**Use external tools**:
```bash
echo "Hello" | hexdump -v -e '/1 "WRITE 0x7E%02_ax 0x%02x\n"'
```

---

## Error Handling

### Build-Time Errors

**Out of bounds address**:
```
WRITE 0x10000 0xFA
Error: Address 0x10000 out of bounds (max 0xFFFF)
```

**Invalid byte value**:
```
WRITE 0x7C00 0x1FF
Error: Byte value 0x1FF out of range (max 0xFF)
```

**Syntax error**:
```
WRITE 0x7C00
Error: WRITE requires 2 arguments: address and value
```

### Runtime Errors

- Builder validates syntax, not semantics
- Invalid opcodes won't cause build errors
- CPU will crash on invalid instructions at runtime

---

## Debugging Tips

### 1. Use Verbose Comments

```
WRITE 0x7C00 0xB4     COMMENT MOV AH, 0x00 (BIOS read key function)
WRITE 0x7C01 0x00     COMMENT   ^^ This sets up INT 0x16 call
```

### 2. Mark Section Boundaries

```
COMMENT ======== SECTION START: 0x7C00 ========
...
COMMENT ======== SECTION END: 0x7C19 ========
```

### 3. Use Build Summary

```bash
python3 build_pxos.py
# Check symbol table output
# Verify operation count
```

### 4. Hexdump the Output

```bash
hexdump -C pxos.bin | head -20
```

### 5. Disassemble

```bash
ndisasm -b 16 -o 0x7C00 pxos.bin | head -30
```

---

## Further Reading

- [build_pxos.py](../build_pxos.py) — Builder implementation
- [pxos_commands.txt](../pxos_commands.txt) — Full source
- [architecture.md](architecture.md) — Memory layout
- [x86 Opcode Reference](http://ref.x86asm.net/)

---

**Next**: [extensions.md](extensions.md) — Extending pxOS
