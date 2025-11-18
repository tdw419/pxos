# pxOS Milestone Roadmap (AI-Facing)

**Machine-readable OS development milestones**

The AI build system should prioritize milestones in this order.
Each milestone has strict acceptance criteria and test cases.

---

## M0: Stable Boot Sector

**Status**: ✅ COMPLETE

**Description**: Reliable boot from BIOS with minimal "Hello pxOS" output

**Constraints**:
- Boot sector at 0x7C00-0x7DFF (512 bytes)
- Boot signature 0x55AA at bytes 510-511
- Stack properly initialized
- Segment registers configured

**Acceptance Criteria**:
- [ ] QEMU boots 10/10 runs without crash
- [ ] "pxOS v1>" prompt displays
- [ ] No triple faults or general protection faults
- [ ] Boot time < 1 second

**Test Command**:
```bash
qemu-system-i386 -fda pxos.bin -nographic -monitor none
```

**Do Not Change**:
- Stack setup (0x7C00-0x7C09)
- Boot signature (0x1FE-0x1FF)
- Initial segment registers

**Address Range**: 0x7C00-0x7DFF

---

## M1: Reliable Text Output

**Status**: ✅ COMPLETE

**Description**: Known-good print_string primitive for displaying messages

**Constraints**:
- Use BIOS INT 0x10, AH=0x0E (teletype)
- Support null-terminated strings
- Handle newlines correctly

**Acceptance Criteria**:
- [ ] Can print "Hello pxOS" reliably
- [ ] Newline (\\n) moves to next line
- [ ] No screen corruption
- [ ] Cursor advances correctly

**Test Cases**:
```
print_string("Hello")      → "Hello" on screen
print_string("A\\nB")       → "A" then "B" on new line
print_string("Test 123")   → "Test 123" on screen
```

**Primitives to Use**:
- `primitives/print_char.json`

**Address Range**: 0x7C28-0x7C37 (current), can extend to 0x7E00+

---

## M2: Keyboard Input + Backspace

**Status**: 🚧 IN PROGRESS

**Description**: Read keyboard input and support backspace for line editing

**Constraints**:
- Use BIOS INT 0x16, AH=0x00 (wait for key)
- Backspace erases previous character
- Enter key processes input line

**Acceptance Criteria**:
- [ ] Typing characters displays them on screen
- [ ] Backspace erases last character
- [ ] Enter key moves to new line
- [ ] Special keys (arrows, function keys) handled gracefully

**Test Cases**:
```
Type "ABC"          → "ABC" displayed
Type "ABC", BS      → "AB" displayed
Type "Test", Enter  → New line, prompt reappears
```

**Primitives to Use**:
- `primitives/wait_for_key.json`
- `primitives/backspace.json`
- `primitives/print_char.json`

**Address Range**: 0x7E00-0x7E50 (50 bytes budget)

---

## M3: Command Parser Skeleton

**Status**: 🔜 PENDING

**Description**: Basic command parser that recognizes typed commands

**Constraints**:
- Input buffer max 80 characters
- Command comparison (strcmp-like)
- Dispatch to command handlers

**Acceptance Criteria**:
- [ ] Recognizes "help", "clear", "exit" commands
- [ ] Displays "Unknown command" for invalid input
- [ ] Command history (optional, nice-to-have)

**Test Cases**:
```
Type "help", Enter   → Display help text
Type "clear", Enter  → Clear screen
Type "xyz", Enter    → "Unknown command: xyz"
```

**Primitives to Use**:
- Custom string comparison routine (new)
- Command dispatch table (new)

**Address Range**: 0x7E50-0x7EB0 (96 bytes budget)

---

## M4: Built-in Commands (help, clear, reboot)

**Status**: 🔜 PENDING

**Description**: Implement 3 core commands with proper handlers

**Commands**:
1. **help** - Display available commands
2. **clear** - Clear screen
3. **reboot** - Warm reboot via BIOS

**Acceptance Criteria**:
- [ ] `help` displays command list
- [ ] `clear` clears screen and resets cursor
- [ ] `reboot` performs warm reboot (INT 0x19)

**Test Cases**:
```
> help
Available commands:
  help   - Show this help
  clear  - Clear screen
  reboot - Restart system

> clear
[Screen cleared, cursor at top]
pxOS v1>

> reboot
[System reboots]
```

**Primitives to Use**:
- `primitives/clear_screen.json`
- Custom help text printer
- Reboot routine (INT 0x19)

**Address Range**: 0x7EB0-0x7F50 (160 bytes budget)

---

## M5: Error Reporting to Screen

**Status**: 🔜 PENDING

**Description**: Display user-friendly error messages

**Constraints**:
- Error messages fit on screen
- Clear indication of error type
- Return to prompt after error

**Acceptance Criteria**:
- [ ] Unknown commands show error
- [ ] Invalid input handled gracefully
- [ ] No silent failures

**Test Cases**:
```
> badcommand
Error: Unknown command 'badcommand'
pxOS v1>

> [invalid character]
Error: Invalid input
pxOS v1>
```

**Address Range**: 0x7F50-0x7FA0 (80 bytes budget)

---

## M6: Extended Boot Code (2nd Sector Load)

**Status**: 🔮 FUTURE

**Description**: Load additional code from disk sector 2

**Constraints**:
- Use BIOS INT 0x13, AH=0x02 (read sectors)
- Load to 0x8000+ (above boot sector)
- Verify checksum before jumping

**Acceptance Criteria**:
- [ ] Successfully loads sector 2
- [ ] Detects read errors
- [ ] Jumps to extended code

**Test Cases**:
```
Boot → Load sector 2 → Jump to 0x8000 → Extended features run
```

**Address Range**: 0x8000+ (new address space)

---

## M7: FAT12 Filesystem Driver

**Status**: 🔮 FUTURE

**Description**: Read files from FAT12 floppy disk

**Constraints**:
- Parse FAT12 boot sector
- Follow file allocation table
- Read root directory entries

**Acceptance Criteria**:
- [ ] Can list files in root directory
- [ ] Can read file contents
- [ ] Handles fragmented files

**Test Cases**:
```
> dir
README.TXT    1024 bytes
HELLO.TXT      128 bytes

> type README.TXT
[File contents displayed]
```

**Address Range**: 0x8000-0x9000 (4KB budget for FS driver)

---

## Milestone Priority Order

For AI automation, implement in this order:

1. **M0** - Stable Boot (prerequisite for everything)
2. **M1** - Text Output (needed for user feedback)
3. **M2** - Keyboard Input (needed for interactivity)
4. **M3** - Command Parser (framework for features)
5. **M4** - Built-in Commands (user-facing functionality)
6. **M5** - Error Reporting (robustness)
7. **M6** - Extended Code (scalability)
8. **M7** - Filesystem (real OS functionality)

---

## AI Usage

When the AI build system runs with `--auto`, it should:

1. Check which milestones are complete
2. Identify the next incomplete milestone
3. Generate primitives to satisfy that milestone's acceptance criteria
4. Build and test
5. Mark milestone complete if all criteria met
6. Move to next milestone

**Command**:
```bash
python3 tools/auto_build_pxos.py --auto --milestone M2
```

---

## Validation

Each milestone has a test script:

```bash
./tests/milestone_M0.sh  # Test M0 completion
./tests/milestone_M1.sh  # Test M1 completion
# etc.
```

These scripts return exit code 0 if milestone is complete, 1 otherwise.

---

**Last Updated**: 2025-11-18
**Total Milestones**: 8
**Complete**: 2 (M0, M1)
**In Progress**: 1 (M2)
**Pending**: 5 (M3-M7)
