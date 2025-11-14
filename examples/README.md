# pxVM Example Programs

This directory contains example programs written in pxVM assembly language.

## Running Examples

### Command Line
```bash
# Assemble a program
python3 ../pxvm_assembler.py factorial.pxasm

# Assemble and run
python3 ../pxvm_assembler.py factorial.pxasm factorial.pxvm
python3 ../spirv_terminal.py
> LOADIMG factorial factorial.pxvm
> RUN factorial
```

### Using the Terminal
```bash
python3 ../spirv_terminal.py
> ASM factorial examples/factorial.pxasm
> RUN factorial
```

### Using Scripts
```bash
python3 ../spirv_terminal.py < test_all.script
```

## Available Examples

### factorial.pxasm
Computes factorial(5) = 120 using recursion.

**Features demonstrated:**
- Recursive function calls
- CALL/RET instructions
- PUSH/POP stack operations
- Conditional branching (JZ)

**Output:** `PRINT R0 = 120`

### fibonacci.pxasm
Computes Fibonacci(10) = 55 using recursion.

**Features demonstrated:**
- Complex recursion (two recursive calls)
- Multiple base cases
- Stack-heavy computation
- Nested function calls

**Output:** `PRINT R0 = 55`

### loop.pxasm
Counts from 1 to 10 using a loop.

**Features demonstrated:**
- Simple loop structure
- Counter increment
- Conditional jump (JNZ)
- Multiple PRINT statements

**Output:**
```
PRINT R0 = 1
PRINT R0 = 2
...
PRINT R0 = 10
```

## Writing Your Own Programs

### Basic Template
```asm
; myprogram.pxasm - Description

main:
    IMM32 R0, 42        ; Load value
    PRINT R0            ; Print it
    HALT                ; Exit

; Add your functions here
my_function:
    ; Function code
    RET
```

### Available Instructions

**Arithmetic:**
- `ADD R0, R1, R2` - R0 = R1 + R2
- `SUB R0, R1, R2` - R0 = R1 - R2
- `MUL R0, R1, R2` - R0 = R1 * R2
- `DIV R0, R1, R2` - R0 = R1 / R2

**Data Movement:**
- `MOV R0, R1` - R0 = R1
- `IMM32 R0, 123` - R0 = immediate value

**Memory:**
- `LOAD R0, [R1]` - Load from memory address in R1
- `STORE [R0], R1` - Store to memory address in R0

**Stack:**
- `PUSH R0` - Push R0 onto stack
- `POP R0` - Pop from stack into R0

**Control Flow:**
- `CALL label` - Call function
- `RET` - Return from function
- `JMP label` - Unconditional jump
- `JZ R0, label` - Jump if R0 == 0
- `JNZ R0, label` - Jump if R0 != 0

**System:**
- `PRINT R0` - Print register value
- `HALT` - Stop execution

### Labels
```asm
main:               ; Label at current address
    CALL subroutine ; Jump to label
    HALT

subroutine:
    ; Code here
    RET
```

### Comments
```asm
; This is a comment
IMM32 R0, 5         ; Inline comment
```

## Next Steps

After mastering these examples, check out:
- **NEXT_STEPS.md** - Week 2 plan for syscalls
- **ROADMAP.md** - Long-term vision
- **ARCHITECTURE.md** - Technical deep dive
