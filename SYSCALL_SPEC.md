# pxVM System Call Specification

**Version:** v0.3 (Phase 2.1)
**Date:** November 14, 2025

---

## Overview

System calls enable pxVM programs to interact with the outside world through a controlled interface. The SYSCALL instruction (opcode 0x80) provides access to file I/O, terminal interaction, and future OS services.

## Calling Convention

### SYSCALL Instruction

**Opcode:** `0x80`
**Format:** `SYSCALL` (no operands)

**Registers:**
- **R0**: Syscall number (input) → Return value (output)
- **R1-R6**: Arguments (up to 6 parameters)
- **R7**: Reserved for future use

**Return Values:**
- **Success**: Non-negative value (≥ 0)
- **Error**: Negative value (-1 to -255)

**Behavior:**
1. Save syscall number from R0
2. Execute syscall handler
3. Write result back to R0
4. Continue execution at PC+1

---

## Error Codes

```
ERR_SUCCESS       =  0   # No error
ERR_INVALID       = -1   # Invalid syscall number
ERR_BADFD         = -2   # Bad file descriptor
ERR_NOMEM         = -3   # Out of memory
ERR_FAULT         = -4   # Memory access violation
ERR_NOTFOUND      = -5   # File not found
ERR_NOSPACE       = -6   # No space left
ERR_BADADDR       = -7   # Invalid memory address
```

---

## File Descriptors

### Standard Streams

```
FD_STDIN  = 0   # Standard input
FD_STDOUT = 1   # Standard output
FD_STDERR = 2   # Standard error
```

### File Descriptor Table

- **Range:** 0-255
- **Allocation:** Dynamic (3+)
- **Lifetime:** Process-scoped
- **Inheritance:** Not inherited (v0.3)

---

## System Calls

### SYS_WRITE (1)

Write data to a file descriptor.

**Signature:**
```
syscall_number = 1
R1 = fd (file descriptor)
R2 = buf_addr (memory address of data)
R3 = count (number of bytes to write)
→ R0 = bytes_written (or error code)
```

**Description:**
- Writes up to `count` bytes from memory address `buf_addr` to file descriptor `fd`
- Returns number of bytes successfully written
- For STDOUT/STDERR: prints to terminal

**Errors:**
- `ERR_BADFD`: Invalid file descriptor
- `ERR_BADADDR`: Buffer address out of bounds
- `ERR_FAULT`: Memory access violation

**Example:**
```asm
; Write "Hello" to stdout
main:
    ; Store "Hello" in memory
    IMM32 R0, 0x48      ; 'H'
    STORE [100], R0
    IMM32 R0, 0x65      ; 'e'
    STORE [101], R0
    IMM32 R0, 0x6C      ; 'l'
    STORE [102], R0
    STORE [103], R0     ; 'l'
    IMM32 R0, 0x6F      ; 'o'
    STORE [104], R0

    ; syscall(SYS_WRITE, STDOUT, &buf, 5)
    IMM32 R0, 1         ; SYS_WRITE
    IMM32 R1, 1         ; STDOUT
    IMM32 R2, 100       ; buffer address
    IMM32 R3, 5         ; length
    SYSCALL

    HALT
```

---

### SYS_READ (2)

Read data from a file descriptor.

**Signature:**
```
syscall_number = 2
R1 = fd (file descriptor)
R2 = buf_addr (memory address to store data)
R3 = count (max bytes to read)
→ R0 = bytes_read (0 = EOF, or error code)
```

**Description:**
- Reads up to `count` bytes from file descriptor `fd` into memory at `buf_addr`
- Returns number of bytes actually read
- Returns 0 on EOF (end of file)
- For STDIN: reads from terminal input buffer

**Errors:**
- `ERR_BADFD`: Invalid file descriptor
- `ERR_BADADDR`: Buffer address out of bounds
- `ERR_FAULT`: Memory access violation

**Example:**
```asm
; Read line from stdin
main:
    IMM32 R0, 2         ; SYS_READ
    IMM32 R1, 0         ; STDIN
    IMM32 R2, 1000      ; buffer at address 1000
    IMM32 R3, 256       ; read up to 256 bytes
    SYSCALL

    ; R0 now contains bytes read
    PRINT R0
    HALT
```

---

### SYS_OPEN (3)

Open a file and return a file descriptor.

**Signature:**
```
syscall_number = 3
R1 = path_addr (address of null-terminated filename string)
R2 = flags (O_RDONLY=0, O_WRONLY=1, O_RDWR=2, O_CREATE=64)
→ R0 = fd (or error code)
```

**Description:**
- Opens file at path pointed to by `path_addr`
- String must be null-terminated (byte 0x00)
- Returns new file descriptor (≥ 3)
- Creates file if O_CREATE flag set and file doesn't exist

**Errors:**
- `ERR_NOTFOUND`: File not found (and O_CREATE not set)
- `ERR_BADADDR`: Path address out of bounds
- `ERR_NOMEM`: Too many open files

**Flags:**
```
O_RDONLY  = 0x00    # Read only
O_WRONLY  = 0x01    # Write only
O_RDWR    = 0x02    # Read and write
O_CREATE  = 0x40    # Create if not exists
O_TRUNC   = 0x80    # Truncate to zero length
```

**Example:**
```asm
; Open file "test.txt" for writing
main:
    ; Store filename in memory
    IMM32 R0, 0x74      ; 't'
    STORE [200], R0
    IMM32 R0, 0x65      ; 'e'
    STORE [201], R0
    IMM32 R0, 0x73      ; 's'
    STORE [202], R0
    IMM32 R0, 0x74      ; 't'
    STORE [203], R0
    IMM32 R0, 0x2E      ; '.'
    STORE [204], R0
    IMM32 R0, 0x74      ; 't'
    STORE [205], R0
    IMM32 R0, 0x78      ; 'x'
    STORE [206], R0
    IMM32 R0, 0x74      ; 't'
    STORE [207], R0
    IMM32 R0, 0x00      ; null terminator
    STORE [208], R0

    ; syscall(SYS_OPEN, &path, O_WRONLY | O_CREATE)
    IMM32 R0, 3         ; SYS_OPEN
    IMM32 R1, 200       ; path address
    IMM32 R2, 0x41      ; O_WRONLY | O_CREATE
    SYSCALL

    ; R0 now contains fd (or error)
    PRINT R0
    HALT
```

---

### SYS_CLOSE (4)

Close a file descriptor.

**Signature:**
```
syscall_number = 4
R1 = fd (file descriptor to close)
→ R0 = 0 (success) or error code
```

**Description:**
- Closes the file descriptor `fd`
- Releases resources associated with the file
- Cannot close STDIN/STDOUT/STDERR (returns ERR_BADFD)

**Errors:**
- `ERR_BADFD`: Invalid file descriptor or already closed

**Example:**
```asm
; Close file descriptor 3
main:
    IMM32 R0, 4         ; SYS_CLOSE
    IMM32 R1, 3         ; fd to close
    SYSCALL

    ; R0 = 0 if success
    PRINT R0
    HALT
```

---

## Virtual Filesystem (v0.3)

### Architecture

```
VirtualFS
├── stdin_buffer: bytearray        # Input buffer
├── stdout_buffer: bytearray       # Output buffer
├── stderr_buffer: bytearray       # Error buffer
├── files: dict[str, bytearray]    # In-memory files
└── fd_table: list[FileHandle]     # Open file descriptors
```

### FileHandle Structure

```python
class FileHandle:
    name: str           # File path (or "<stdin>", "<stdout>", "<stderr>")
    mode: int           # O_RDONLY, O_WRONLY, O_RDWR
    pos: int            # Current read/write position
    data: bytearray     # File contents (reference to files[name])
```

### Limitations (v0.3)

- **In-memory only:** No persistent storage
- **No directories:** Flat namespace
- **No permissions:** All files accessible
- **No locking:** Single-process model
- **Max file size:** Limited by Python memory
- **Max open files:** 256 file descriptors

### Future Enhancements

**v0.5 - Persistent Storage:**
- Write files to host filesystem
- Mount host directories
- Path translation

**v0.8 - Multi-process:**
- Per-process file descriptor tables
- File locking
- Shared memory files

**v1.0 - Full POSIX:**
- Directory operations (mkdir, rmdir, readdir)
- File metadata (stat, chmod, chown)
- Symbolic links
- Pipes and sockets

---

## Memory Layout

### String Storage

Strings in pxVM are null-terminated byte sequences:

```
Address:  1000  1001  1002  1003  1004
Content:  0x48  0x65  0x6C  0x6C  0x6F  0x00
String:   'H'   'e'   'l'   'l'   'o'   '\0'
```

### Buffer Management

**Recommendation:** Reserve memory regions for I/O:
- **0-999:** Code and data
- **1000-1999:** String buffers
- **2000-2999:** I/O buffers
- **3000-4095:** Stack and heap

---

## Example Programs

### hello.pxasm

Classic "Hello, World!" using syscalls:

```asm
; hello.pxasm - First real I/O program
; Prints "Hello, World!\n" to stdout

main:
    ; Store "Hello, World!\n" in memory
    IMM32 R0, 0x48      ; 'H'
    STORE [1000], R0
    IMM32 R0, 0x65      ; 'e'
    STORE [1001], R0
    IMM32 R0, 0x6C      ; 'l'
    STORE [1002], R0
    STORE [1003], R0    ; 'l'
    IMM32 R0, 0x6F      ; 'o'
    STORE [1004], R0
    IMM32 R0, 0x2C      ; ','
    STORE [1005], R0
    IMM32 R0, 0x20      ; ' '
    STORE [1006], R0
    IMM32 R0, 0x57      ; 'W'
    STORE [1007], R0
    IMM32 R0, 0x6F      ; 'o'
    STORE [1008], R0
    IMM32 R0, 0x72      ; 'r'
    STORE [1009], R0
    IMM32 R0, 0x6C      ; 'l'
    STORE [1010], R0
    IMM32 R0, 0x64      ; 'd'
    STORE [1011], R0
    IMM32 R0, 0x21      ; '!'
    STORE [1012], R0
    IMM32 R0, 0x0A      ; '\n'
    STORE [1013], R0

    ; syscall(SYS_WRITE, STDOUT, &msg, 14)
    IMM32 R0, 1         ; SYS_WRITE
    IMM32 R1, 1         ; STDOUT
    IMM32 R2, 1000      ; message address
    IMM32 R3, 14        ; length (including newline)
    SYSCALL

    HALT
```

### echo.pxasm

Read from stdin, write to stdout:

```asm
; echo.pxasm - Echo stdin to stdout
; Reads a line and prints it back

main:
    ; Read from stdin
    IMM32 R0, 2         ; SYS_READ
    IMM32 R1, 0         ; STDIN
    IMM32 R2, 2000      ; buffer address
    IMM32 R3, 256       ; max bytes
    SYSCALL

    ; Save bytes read
    MOV R4, R0

    ; Write to stdout
    IMM32 R0, 1         ; SYS_WRITE
    IMM32 R1, 1         ; STDOUT
    IMM32 R2, 2000      ; buffer address
    MOV R3, R4          ; bytes read
    SYSCALL

    HALT
```

---

## Testing

### Test Cases

1. **hello.pxasm** - Basic output
   - Expected: "Hello, World!\n"
   - Validates: SYS_WRITE, STDOUT

2. **echo.pxasm** - Input/output
   - Input: "test\n"
   - Expected: "test\n"
   - Validates: SYS_READ, SYS_WRITE, STDIN, STDOUT

3. **file_create.pxasm** - File creation
   - Creates "test.txt" with content
   - Validates: SYS_OPEN, SYS_WRITE, SYS_CLOSE

4. **file_read.pxasm** - File reading
   - Reads "test.txt" and prints
   - Validates: SYS_OPEN, SYS_READ, SYS_CLOSE

### Success Metrics

- ✅ All 4 syscalls implemented
- ✅ Standard streams (STDIN/STDOUT/STDERR) working
- ✅ File operations (open/read/write/close) functional
- ✅ Error handling for invalid inputs
- ✅ Example programs run successfully

---

## Implementation Notes

### Performance

- **Syscall overhead:** ~10 Python function calls per syscall
- **Memory copy cost:** O(n) for read/write operations
- **String search:** O(n) for null terminator finding

### Security (v0.3)

⚠️ **No sandboxing in v0.3:**
- Memory bounds checked but no permissions
- All files accessible to all code
- No resource limits

**Future:** Phase 6 will add security model.

### Debugging

New terminal commands for v0.3:
- `FEED <text>` - Add text to stdin buffer
- `STDOUT` - Print stdout buffer contents
- `STDERR` - Print stderr buffer contents
- `LS` - List virtual filesystem files
- `CAT <file>` - Print file contents

---

**End of Specification**
