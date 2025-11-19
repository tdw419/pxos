# Hardware Mailbox Protocol for pxOS

## Overview

The **mailbox protocol** is the core communication mechanism between the CPU microkernel and the GPU runtime in pxOS. It enables the CPU to make "system calls" to the GPU, inverting the traditional privilege model.

**Traditional OS:** CPU (ring 0) → GPU (servant)
**pxOS:** GPU (ring 0) ← CPU (ring 3) via mailbox

---

## Architecture

### Memory-Mapped Registers

The mailbox is implemented as four 32-bit registers mapped at BAR0:

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| 0x0000 | CMD_REG | CPU Write | Command register (opcode + payload) |
| 0x0004 | STATUS_REG | GPU Write | Status flags (ready, busy, complete, error) |
| 0x0008 | RESP_REG | GPU Write | Response data from GPU |
| 0x000C | DOORBELL | CPU Write | Doorbell to notify GPU |

### Command Format (32-bit)

```
Bits 31-24: Opcode (operation code)
Bits 23-16: Thread ID (for multi-threading)
Bits 15-0:  Payload (parameter/data)
```

Example:
```
0x80004800 = OP_UART_WRITE | tid=0 | char='H'
```

### Status Flags

| Bit | Name | Description |
|-----|------|-------------|
| 0 | READY | GPU is ready for commands |
| 1 | BUSY | GPU is processing a command |
| 2 | COMPLETE | Command completed successfully |
| 31 | ERROR | Error occurred during processing |

---

## Opcodes

| Opcode | Name | Description | Payload | Response |
|--------|------|-------------|---------|----------|
| 0x80 | OP_UART_WRITE | Write character to serial port | char | None |
| 0x81 | OP_UART_READ | Read character from serial port | 0 | char |
| 0x82 | OP_GPU_EXECUTE | Execute GPU kernel | kernel_id | status |
| 0x83 | OP_MMIO_READ | Read from MMIO register | offset | value |
| 0x84 | OP_MMIO_WRITE | Write to MMIO register | value | None |
| 0x8F | OP_HALT | Halt the system | 0 | None |

---

## Protocol Flow

### 1. Command Submission

```nasm
; CPU code
mailbox_send_command:
    ; Wait for GPU to be ready (STATUS_REG & BUSY == 0)
    mov rbx, [gpu_bar0_virt]
.wait_ready:
    mov eax, [rbx + STATUS_REG]
    test eax, STATUS_BUSY
    jz .send
    pause
    loop .wait_ready

.send:
    ; Write command to CMD_REG
    mov eax, command_value
    mov [rbx + CMD_REG], eax
    mfence

    ; Ring doorbell
    mov dword [rbx + DOORBELL], 1
    mfence
```

### 2. GPU Processing

```wgsl
// GPU runtime (WGSL shader)
@compute @workgroup_size(1)
fn mailbox_handler() {
    let cmd = textureLoad(mailbox_cmd, vec2<u32>(0, 0)).r;
    let opcode = (cmd >> 24) & 0xFF;
    let tid = (cmd >> 16) & 0xFF;
    let payload = cmd & 0xFFFF;

    // Dispatch based on opcode
    if (opcode == OP_UART_WRITE) {
        uart_write(payload);
        textureStore(mailbox_status, vec2<u32>(0, 0), STATUS_COMPLETE);
    }
}
```

### 3. Response Polling

```nasm
; CPU code
mailbox_poll_complete:
    mov rbx, [gpu_bar0_virt]
    mov rcx, 1000000    ; Timeout

.poll_loop:
    mov eax, [rbx + STATUS_REG]
    test eax, STATUS_COMPLETE
    jnz .complete

    test eax, STATUS_ERROR
    jnz .error

    pause
    loop .poll_loop

.complete:
    ; Read response
    mov eax, [rbx + RESP_REG]
    ret
```

---

## Performance Characteristics

### Latency Targets

| Operation | Target | Typical |
|-----------|--------|---------|
| Command write | < 10 cycles | 5 cycles |
| GPU processing | < 1000 cycles | 500-800 cycles |
| Total round-trip | < 1 μs | 0.5-0.8 μs |

### Optimization Techniques

**1. Batching**
```nasm
; BAD: Individual commands (high overhead)
call mailbox_send_command  ; 'H'
call mailbox_poll_complete
call mailbox_send_command  ; 'e'
call mailbox_poll_complete
; Total: 2 round-trips = 2 μs

; GOOD: Batch commands
call mailbox_send_batch    ; "Hello"
call mailbox_poll_complete
; Total: 1 round-trip = 1 μs
```

**2. Write-Combining**
```nasm
; Use WC memory for command buffer
; Write multiple commands without fencing
mov [rbx + CMD_BUF + 0], eax
mov [rbx + CMD_BUF + 4], ebx
mov [rbx + CMD_BUF + 8], ecx
mfence                      ; Single fence at end
mov [rbx + DOORBELL], 1
```

**3. Polling Optimization**
```nasm
; Use PAUSE instruction to reduce bus contention
.poll_loop:
    pause                   ; Hint to CPU: spinning
    mov eax, [rbx + STATUS_REG]
    test eax, STATUS_COMPLETE
    jz .poll_loop
```

---

## Memory Ordering

### Critical: MFENCE After Writes

MMIO writes MUST be fenced to ensure proper ordering:

```nasm
; CORRECT
mov [rbx + CMD_REG], eax
mfence                      ; Ensures write completes
mov [rbx + DOORBELL], 1
mfence

; INCORRECT - GPU may see doorbell before command!
mov [rbx + CMD_REG], eax
mov [rbx + DOORBELL], 1     ; May overtake CMD_REG write
```

### Why MFENCE Is Required

1. **Out-of-Order Execution**: CPU may reorder writes
2. **Write Buffers**: Writes may be buffered
3. **PCIe Ordering**: PCIe allows relaxed ordering
4. **Uncacheable Memory**: UC doesn't enforce strict ordering

---

## Error Handling

### Timeout Handling

```nasm
mailbox_send_command:
    mov rcx, 100000         ; Timeout counter
.wait_ready:
    mov eax, [rbx + STATUS_REG]
    test eax, STATUS_BUSY
    jz .send
    pause
    loop .wait_ready

    ; Timeout occurred
    mov rax, 0xFFFFFFFF     ; Error code
    ret
```

### Error Detection

```nasm
mailbox_poll_complete:
    mov eax, [rbx + STATUS_REG]
    test eax, STATUS_ERROR
    jz .check_complete

    ; GPU reported error
    mov eax, [rbx + RESP_REG]
    or eax, 0x80000000      ; Set error flag
    ret
```

---

## Statistics and Monitoring

### Performance Counters

```nasm
section .data
mailbox_stats:
    .commands_sent:     dq 0
    .commands_complete: dq 0
    .commands_failed:   dq 0
    .total_cycles:      dq 0
```

### Cycle Measurement

```nasm
; Before command
rdtsc
shl rdx, 32
or rax, rdx
mov r15, rax            ; Save start time

; Send command
call mailbox_send_command
call mailbox_poll_complete

; After command
rdtsc
shl rdx, 32
or rax, rdx
sub rax, r15            ; Elapsed cycles
add [mailbox_stats.total_cycles], rax
```

---

## Testing

### Basic Functionality Test

```nasm
mailbox_test:
    ; Send UART write command
    mov rdi, (OP_UART_WRITE << 24) | 'H'
    call mailbox_send_command

    ; Wait for completion
    call mailbox_poll_complete

    ; Check result
    test rax, rax
    jnz .failed
    ; Success!
```

### Stress Test

```nasm
mailbox_stress_test:
    mov rcx, 10000          ; 10,000 commands
.loop:
    mov rdi, (OP_UART_WRITE << 24) | 'A'
    call mailbox_send_command
    call mailbox_poll_complete
    loop .loop

    ; Calculate average latency
    mov rax, [mailbox_stats.total_cycles]
    xor rdx, rdx
    mov rbx, 10000
    div rbx
    ; RAX = average cycles per command
```

---

## Debug Output

### Serial Logging

```nasm
; Log command submission
lea rsi, [rel msg_mailbox_cmd]
call serial_print_64

mov rax, rdi            ; Command value
call print_hex_64

; Log completion
lea rsi, [rel msg_mailbox_done]
call serial_print_64
```

### Expected Output

```
Initializing mailbox protocol... READY
Testing mailbox: sending 'H' via UART... OK
Hello from GPU OS!
Mailbox stats:
  Commands sent: 5
  Commands complete: 5
  Average latency: 750 cycles (0.31 μs @ 2.4 GHz)
```

---

## Future Enhancements

### Phase 3: Command Buffers

Instead of single commands, use a ring buffer:

```
Mailbox Layout:
  0x0000: CMD_QUEUE_HEAD (CPU writes)
  0x0004: CMD_QUEUE_TAIL (GPU writes)
  0x0008: CMD_BUFFER[1024]
```

**Benefits:**
- Batch 100+ commands per doorbell
- Reduce round-trip overhead from 1 μs to 10 ns per command
- Achieve <1% CPU overhead target

### Phase 4: Interrupt-Driven

Replace polling with GPU-to-CPU interrupts:

```nasm
; GPU signals completion via MSI-X interrupt
; CPU interrupt handler:
mailbox_irq_handler:
    mov eax, [rbx + RESP_REG]
    mov [command_response], eax
    ; Wake up waiting thread
```

**Benefits:**
- CPU can do other work instead of polling
- Better power efficiency
- Lower latency for async operations

---

## Resources

- [PCIe Base Specification](https://pcisig.com/specifications)
- [Intel SDM - Memory Ordering](https://www.intel.com/content/www/us/en/architecture-and-technology/64-ia-32-architectures-software-developer-manual-325462.html)
- [OSDev Wiki - MMIO](https://wiki.osdev.org/MMIO)

---

**pxOS Phase 2 - Hardware Mailbox Protocol** 🚀

*"The GPU is the kernel, the CPU is the syscall"* - pxOS Architecture
