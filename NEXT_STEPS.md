# pxOS Next Steps - Immediate Priorities

**Status:** pxVM v0.2 complete and functional
**Goal:** Get to pxVM v0.3 with syscalls and assembler

---

## Week 1: Build the Assembler

### Day 1-2: Basic Assembler Structure
```python
# File: pxvm_assembler.py

class PxVmAssembler:
    def assemble(self, source_code):
        """Convert assembly to bytecode"""
        lines = self.preprocess(source_code)
        labels = self.extract_labels(lines)
        bytecode = self.generate_code(lines, labels)
        return bytecode
```

**Example input (hello.pxasm):**
```asm
    IMM32 R0, 5
    CALL factorial
    PRINT R0
    HALT

factorial:
    IMM32 R1, 1
    SUB R2, R0, R1
    JZ R2, base_case
    PUSH R0
    MOV R0, R2
    CALL factorial
    POP R1
    MUL R0, R0, R1
    RET

base_case:
    IMM32 R0, 1
    RET
```

**Output:** `factorial.pxvm` (same 48 bytes we created manually)

### Day 3-4: Assembler Features
- [ ] Instruction mnemonics → opcodes
- [ ] Label resolution (forward and backward)
- [ ] Comments (`;` or `#`)
- [ ] Immediate value parsing (decimal, hex, binary)
- [ ] Register name validation
- [ ] Error messages with line numbers

### Day 5: Integration
- [ ] Add `ASM <name> <path.pxasm>` command to terminal
- [ ] Auto-generates `.pxvm` file
- [ ] Test with existing factorial program
- [ ] Write 3+ new test programs in assembly

**Deliverable:** Never write raw hex again!

---

## Week 2: System Calls

### Day 1-2: Syscall Infrastructure
```python
# Add to spirv_terminal.py

class SystemCalls:
    # Syscall numbers
    SYS_WRITE = 1
    SYS_READ = 2
    SYS_OPEN = 3
    SYS_CLOSE = 4

    def __init__(self):
        self.open_files = {
            0: sys.stdin,   # stdin
            1: sys.stdout,  # stdout
            2: sys.stderr   # stderr
        }
        self.next_fd = 3

    def syscall(self, num, arg1, arg2, arg3):
        if num == self.SYS_WRITE:
            return self.sys_write(arg1, arg2, arg3)
        elif num == self.SYS_READ:
            return self.sys_read(arg1, arg2, arg3)
        # ... etc
```

### Day 3: Implement Core Syscalls

**SYS_WRITE (1)**
```
Input:  R1 = file descriptor
        R2 = memory address (buffer)
        R3 = byte count
Output: R0 = bytes written (or -1 on error)
        R1 = errno
```

**SYS_READ (2)**
```
Input:  R1 = file descriptor
        R2 = memory address (buffer)
        R3 = byte count
Output: R0 = bytes read (or -1 on error)
        R1 = errno
```

### Day 4-5: Test Programs

**hello.pxasm** - "Hello, World!" via syscall
```asm
.data
msg: .string "Hello, World!\n"

.text
    IMM32 R1, 1              ; STDOUT
    IMM32 R2, msg            ; message address
    IMM32 R3, 14             ; message length
    SYSCALL SYS_WRITE        ; write(1, msg, 14)
    HALT
```

**echo.pxasm** - Read from stdin, write to stdout
```asm
    IMM32 R1, 0              ; STDIN
    IMM32 R2, buffer         ; read buffer
    IMM32 R3, 256            ; max bytes
    SYSCALL SYS_READ         ; read(0, buffer, 256)

    MOV R3, R0               ; bytes read → write count
    IMM32 R1, 1              ; STDOUT
    IMM32 R2, buffer         ; same buffer
    SYSCALL SYS_WRITE        ; write(1, buffer, count)
    HALT

.bss
buffer: .space 256
```

**Deliverable:** Real I/O from pxVM programs!

---

## Quick Wins (Do These Anytime)

### Easy Improvements
1. **Better PRINT formatting**
   - `PRINT R0` → shows decimal AND hex
   - `PRINTHEX R0` → hex only
   - `PRINTSTR <addr>` → print null-terminated string from memory

2. **Memory dump command**
   ```
   DUMPMEM 0 64    # Show first 64 bytes of VM memory
   ```

3. **Execution statistics**
   ```
   STATS factorial  # Show: cycles, instructions, syscalls
   ```

4. **Multi-line REPL**
   ```
   > ASM
   ... IMM32 R0, 42
   ... PRINT R0
   ... HALT
   ... END
   OK ASM inline (6 bytes)
   > RUN inline
   PRINT R0 = 42
   ```

---

## Architecture Decisions Needed

### 1. Memory Layout
Current: 4KB flat memory (0x0000-0x0FFF)

Proposed for v0.3:
```
0x0000-0x00FF: Interrupt vector table
0x0100-0x01FF: System reserved
0x0200-0x7FFF: User code & data
0x8000-0xFFFF: Stack (grows downward from 0xFFFF)
```

**Decision:** Implement now or wait for Phase 4 (processes)?
**Recommendation:** Implement now - simple and enables better programs

### 2. Syscall Calling Convention
Option A: Registers only (current plan)
```asm
IMM32 R1, 1          ; fd
IMM32 R2, buffer     ; buf
IMM32 R3, 14         ; count
SYSCALL SYS_WRITE    ; uses R1-R3
```

Option B: Stack-based (more extensible)
```asm
PUSH 14              ; count
PUSH buffer          ; buf
PUSH 1               ; fd
SYSCALL SYS_WRITE    ; pops args
```

**Decision:** A for simplicity, B for compatibility?
**Recommendation:** Start with A, can add B later

### 3. String/Data Directives
How to embed strings in assembly?

Option A: Manual bytes
```asm
msg: 48 65 6C 6C 6F  ; "Hello"
```

Option B: Assembler directives
```asm
msg: .string "Hello"
msglen: .word 5
```

**Decision:** Needed before we can write real programs
**Recommendation:** Implement Option B in assembler

---

## Metrics for Success

### Week 1 Complete:
- [ ] Assembler converts `.pxasm` → `.pxvm`
- [ ] factorial.pxasm assembles to same bytecode
- [ ] 3+ new programs written in assembly
- [ ] No more manual hex editing!

### Week 2 Complete:
- [ ] SYS_WRITE and SYS_READ working
- [ ] "Hello, World!" via syscall
- [ ] Echo program works
- [ ] Can read/write files in virtual filesystem

---

## Files to Create

### Week 1 (Assembler)
```
pxvm_assembler.py        # The assembler
examples/factorial.pxasm # Rewrite in assembly
examples/fibonacci.pxasm # New program
examples/loops.pxasm     # Control flow demo
test_assembler.py        # Unit tests
```

### Week 2 (Syscalls)
```
syscalls.py              # Syscall implementation
examples/hello.pxasm     # Hello world
examples/echo.pxasm      # I/O demo
examples/cat.pxasm       # File reader
test_syscalls.py         # Syscall tests
```

---

## Commands to Add to Terminal

```python
# New commands for Phase 1-2

def cmd_ASM(self, name, path):
    """Assemble .pxasm to .pxvm: ASM <name> <path>"""
    from pxvm_assembler import PxVmAssembler
    asm = PxVmAssembler()
    with open(path, 'r') as f:
        source = f.read()
    bytecode = asm.assemble(source)
    pxvm_path = path.replace('.pxasm', '.pxvm')
    with open(pxvm_path, 'wb') as f:
        f.write(bytecode)
    self.programs[name] = bytecode
    print(f"OK ASM {name} ({len(bytecode)} bytes)")

def cmd_DISASM(self, name):
    """Disassemble program: DISASM <name>"""
    if name not in self.programs:
        print(f"ERR program {name} not found")
        return
    # Show assembly representation
    disassemble(self.programs[name])

def cmd_DUMPMEM(self, start_str, count_str):
    """Dump VM memory: DUMPMEM <start> <count>"""
    # Show memory contents during/after execution

def cmd_STATS(self, name):
    """Show execution stats: STATS <name>"""
    # Cycle count, instruction count, syscalls
```

---

## The Goal

By end of Week 2, you should be able to:

```bash
$ cat hello.pxasm
    IMM32 R1, 1
    IMM32 R2, msg
    IMM32 R3, 14
    SYSCALL 1
    HALT
msg: .string "Hello, World!\n"

$ python3 spirv_terminal.py
> ASM hello hello.pxasm
OK ASM hello (32 bytes)
> RUN hello
Hello, World!
>
```

**That's the milestone: Real programs with real I/O!**

---

## Resources Needed

### Time: ~40-60 hours
- Assembler: 20-25 hours
- Syscalls: 15-20 hours
- Testing: 5-10 hours
- Documentation: 5 hours

### Skills Required:
- Python (already have ✓)
- Assembly language concepts (learning as we go ✓)
- System call knowledge (will learn ✓)

### Dependencies:
- None! Everything builds on what's already done

---

## Why This Matters

Right now, writing programs requires manual hex encoding:
```
WRITEFILE factorial.pxvm 30 00 05 00 00 00 60 0C 00 F0 00 FF ...
```

After Week 1, you write this instead:
```asm
IMM32 R0, 5
CALL factorial
PRINT R0
HALT
```

After Week 2, you can do this:
```asm
SYSCALL SYS_WRITE  # Real I/O!
```

**Each step makes pxOS more real and less theoretical.**

---

## Start Here

1. **Create pxvm_assembler.py** (scaffold)
2. **Test with factorial** (known-good program)
3. **Add one feature at a time** (mnemonics, then labels, then directives)
4. **Write new programs** to test each feature

**First commit:** "Add basic pxVM assembler (mnemonics only)"
**Second commit:** "Add label support to assembler"
**Third commit:** "Add string directives to assembler"

Small, incremental, testable progress.

Let's build this! 🚀
