# Phase 2: GPU Hardware Integration Architecture

**Date**: 2025-11-18
**Status**: Design phase (pending GRUB boot validation)
**Goal**: Real GPU-CPU communication on bare metal

---

## 🎯 Objectives

Phase 2 extends the validated Phase 1 architecture (Python test harness) to real GPU hardware:

1. **Map GPU MMIO regions** (BAR0, BAR2)
2. **Implement hardware mailbox** (shared memory)
3. **Submit GPU commands** (via command buffers)
4. **Execute pixel programs** (on real GPU shaders)
5. **Test on QEMU** (bochs-display or virtio-gpu)

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     pxOS Phase 2                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐              ┌──────────────┐            │
│  │              │   Mailbox    │              │            │
│  │  CPU Core    │◄────────────►│  GPU Shader  │            │
│  │  (Broker)    │  (UC-MMIO)   │  (Pixel ISA) │            │
│  │              │              │              │            │
│  └──────┬───────┘              └───────▲──────┘            │
│         │                              │                   │
│         │ BAR0 Map                     │ Command           │
│         │ (MMIO)                       │ Buffer            │
│         ▼                              │                   │
│  ┌──────────────────────────────────────────┐              │
│  │     GPU Memory (VRAM / BAR0)             │              │
│  ├──────────────────────────────────────────┤              │
│  │ 0x0000: Mailbox (4 bytes, UC)            │              │
│  │ 0x1000: Command Buffer (4KB, WC)         │              │
│  │ 0x2000: Pixel Program (os.pxi)           │              │
│  │ 0x3000: Framebuffer (WC)                 │              │
│  └──────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧱 Component 1: BAR Memory Mapping

### PCIe BAR Detection (Already Working)

From Phase 1, we have PCIe enumeration code that reads:
- **BAR0**: GPU memory-mapped I/O (framebuffer + MMIO)
- **BAR2**: GPU control registers (optional)

**Example output from QEMU**:
```
Device 00:02.0: VGA compatible controller
  BAR0: 0xE0000000 (size 16MB, MMIO)
  BAR2: 0xFEBF0000 (size 4KB, MMIO)
```

### Memory Mapping Strategy

We need to map BAR0 into CPU virtual address space with **correct cache attributes**:

#### Region 1: Mailbox (Uncacheable)
- **Physical**: BAR0 + 0x0000 (first 4KB)
- **Size**: 4 bytes (32-bit mailbox word)
- **Cache**: UC (Uncacheable) - ensures CPU-GPU synchronization
- **Virtual**: Identity-mapped at BAR0 address

#### Region 2: Command Buffer (Write-Combining)
- **Physical**: BAR0 + 0x1000
- **Size**: 4KB (1024 commands)
- **Cache**: WC (Write-Combining) - batches writes for performance
- **Virtual**: Identity-mapped

#### Region 3: Pixel Program Storage
- **Physical**: BAR0 + 0x2000
- **Size**: 64KB (16,384 pixel instructions)
- **Cache**: WC
- **Virtual**: Identity-mapped

#### Region 4: Framebuffer
- **Physical**: BAR0 + 0x10000
- **Size**: Remaining BAR0 space (up to 16MB)
- **Cache**: WC
- **Virtual**: Identity-mapped

### Page Table Attributes

**PAT (Page Attribute Table) Setup**:
```nasm
; PAT MSR (0x277) setup for cache types
; PAT0 = WB (Write-Back)       - default
; PAT1 = WT (Write-Through)    - not used
; PAT2 = UC (Uncacheable)      - mailbox
; PAT3 = UC- (Uncacheable)     - not used
; PAT4 = WB (Write-Back)       - not used
; PAT5 = WT (Write-Through)    - not used
; PAT6 = UC (Uncacheable)      - not used
; PAT7 = WC (Write-Combining)  - buffers/framebuffer

mov ecx, 0x277
rdmsr
; Set PAT2 = UC (0x00), PAT7 = WC (0x01)
and eax, 0x00FFFFFF
or  eax, 0x00000000  ; PAT2 = UC
and edx, 0xFF00FFFF
or  edx, 0x00010000  ; PAT7 = WC
wrmsr
```

**PTE Flags for Each Region**:

```nasm
; Mailbox (UC)
; PTE = base | present | writable | PAT2
mov eax, [bar0_base]
or  eax, 0x83 | (1 << 7)  ; Present, RW, PS, PAT

; Command Buffer (WC)
; PTE = base | present | writable | PAT7
mov eax, [bar0_base]
add eax, 0x1000
or  eax, 0x83 | (1 << 12) ; Present, RW, PS, PAT7
```

---

## 🧱 Component 2: Mailbox Protocol (Hardware)

### Mailbox Memory Layout

**Physical Address**: `BAR0 + 0x0000`
**Size**: 32 bits (4 bytes)
**Format**: Same as Phase 1

```
┌───────────┬──────────┬─────────────────┐
│  Opcode   │  TID     │    Payload      │
│  (8 bits) │ (8 bits) │   (16 bits)     │
└───────────┴──────────┴─────────────────┘
  31:24       23:16         15:0
```

### CPU Write Sequence

```nasm
; CPU writes GPU request
mov dword [BAR0 + 0x0000], 0x80000048  ; Request to print 'H'

; CPU polls for completion
.wait:
    mov eax, [BAR0 + 0x0000]
    test eax, eax
    jnz .wait  ; Wait until GPU clears mailbox
```

### GPU Read Sequence (Shader Pseudocode)

```c
// GPU shader reads mailbox
uint32_t request = *((volatile uint32_t*)(BAR0 + 0x0000));

if (request != 0) {
    uint8_t opcode = (request >> 24) & 0xFF;
    uint8_t tid = (request >> 16) & 0xFF;
    uint16_t payload = request & 0xFFFF;

    // Handle request
    if (opcode == OP_MMIO_WRITE_UART) {
        uart_write((char)payload);
    }

    // Clear mailbox to signal completion
    *((volatile uint32_t*)(BAR0 + 0x0000)) = 0;
}
```

### Synchronization

**Critical**: Mailbox must be mapped **UC (Uncacheable)** to ensure:
- CPU writes are immediately visible to GPU
- GPU writes are immediately visible to CPU
- No cache coherency issues

---

## 🧱 Component 3: Command Buffer

### Purpose

Instead of polling a single mailbox, use a **ring buffer** for batched GPU commands.

### Layout

**Physical Address**: `BAR0 + 0x1000`
**Size**: 4KB (1024 commands, each 4 bytes)
**Cache**: WC (Write-Combining) for performance

```
┌────────────────────────────────────────┐
│  Head Pointer (4 bytes, offset 0x0)    │  ← CPU writes here
├────────────────────────────────────────┤
│  Tail Pointer (4 bytes, offset 0x4)    │  ← GPU writes here
├────────────────────────────────────────┤
│  Command 0    (4 bytes, offset 0x8)    │
│  Command 1    (4 bytes, offset 0xC)    │
│  ...                                   │
│  Command 1022 (4 bytes, offset 0xFFC)  │
└────────────────────────────────────────┘
```

### CPU Producer Code

```nasm
; Get current head
mov eax, [BAR0 + 0x1000]  ; Read head pointer
mov ebx, eax
inc ebx
and ebx, 0x3FF            ; Wrap at 1024
cmp ebx, [BAR0 + 0x1004]  ; Check if buffer full (head+1 == tail)
je .buffer_full

; Write command
shl eax, 2                ; head * 4 (4 bytes per command)
add eax, 0x1008           ; Offset to command array
mov dword [BAR0 + eax], 0x80000048  ; Write command

; Update head
mov dword [BAR0 + 0x1000], ebx

.buffer_full:
; Handle full buffer (wait or drop)
```

### GPU Consumer Code (Shader)

```c
volatile uint32_t* head = (uint32_t*)(BAR0 + 0x1000);
volatile uint32_t* tail = (uint32_t*)(BAR0 + 0x1004);
volatile uint32_t* commands = (uint32_t*)(BAR0 + 0x1008);

while (*tail != *head) {
    uint32_t cmd = commands[*tail];

    // Process command
    handle_command(cmd);

    // Advance tail
    *tail = (*tail + 1) & 0x3FF;  // Wrap at 1024
}
```

### Doorbell Register

To notify GPU of new commands, write to a doorbell register:

```nasm
; After writing commands, ring doorbell
mov dword [BAR2 + 0x0], 1  ; GPU interrupt/wake
```

---

## 🧱 Component 4: Pixel Program Upload

### Upload Process

1. **Generate PXI file** (already working from Phase 1)
2. **Copy to GPU memory** via BAR0

```nasm
; Load os.pxi into GPU memory
mov esi, os_pxi_data      ; Source: RAM buffer
mov edi, BAR0 + 0x2000    ; Dest: GPU memory
mov ecx, os_pxi_size / 4  ; Size in dwords
rep movsd                 ; Copy to GPU (WC mapping)
```

### GPU Execution

GPU shader fetches pixel instructions:

```c
uint8_t* pixel_program = (uint8_t*)(BAR0 + 0x2000);
uint32_t pc = 0;  // Program counter

while (running) {
    uint8_t opcode = pixel_program[pc * 4];
    uint8_t r = pixel_program[pc * 4 + 1];
    uint8_t g = pixel_program[pc * 4 + 2];
    uint8_t b = pixel_program[pc * 4 + 3];

    execute_instruction(opcode, r, g, b);
    pc++;
}
```

---

## 🧱 Component 5: GPU Command Submission

### Minimal GPU Command Format

For simple GPUs (QEMU bochs-display, virtio-gpu), use simple MMIO writes:

```nasm
; Set framebuffer base
mov dword [BAR2 + 0x10], BAR0 + 0x10000  ; FB address

; Set resolution
mov dword [BAR2 + 0x14], 1024  ; Width
mov dword [BAR2 + 0x18], 768   ; Height

; Enable display
mov dword [BAR2 + 0x00], 1     ; Enable bit
```

### For Real GPUs (Intel/AMD/NVIDIA)

Use proper command submission:

```c
struct gpu_command {
    uint32_t opcode;      // SHADER_DISPATCH, MEMORY_WRITE, etc.
    uint32_t args[7];     // Command-specific arguments
};

// Example: Dispatch shader
struct gpu_command cmd = {
    .opcode = GPU_CMD_DISPATCH_SHADER,
    .args = {
        shader_address,    // Pixel program location
        thread_count,      // Number of GPU threads
        0, 0, 0, 0, 0
    }
};

// Submit via command buffer
submit_gpu_command(&cmd);
```

---

## 🧪 Testing Strategy

### Test 1: Mailbox Echo Test
```
CPU writes: 0x80000048 ('H')
GPU reads, prints, clears
CPU verifies: mailbox == 0
✅ Pass if echo successful
```

### Test 2: Command Buffer Stress Test
```
CPU writes 100 commands rapidly
GPU processes all commands
CPU verifies: head == tail
✅ Pass if all commands processed
```

### Test 3: Pixel Program Execution
```
CPU uploads os.pxi to GPU memory
CPU triggers GPU dispatch
GPU executes "Hello from GPU OS!"
CPU reads UART output
✅ Pass if output matches
```

### Test 4: Framebuffer Write
```
CPU writes pattern to framebuffer
GPU copies pattern to screen
CPU verifies pixel values
✅ Pass if display matches
```

---

## 🎯 Implementation Roadmap

### Week 1: BAR Mapping
- [ ] Implement PAT setup (cache types)
- [ ] Map BAR0 with UC for mailbox
- [ ] Map BAR0 with WC for buffers
- [ ] Verify with memory tests

### Week 2: Mailbox Protocol
- [ ] Implement CPU write/poll code
- [ ] Test with QEMU virtio-gpu
- [ ] Measure latency
- [ ] Optimize synchronization

### Week 3: Command Buffer
- [ ] Implement ring buffer producer
- [ ] Implement GPU consumer (shader)
- [ ] Test throughput
- [ ] Add doorbell mechanism

### Week 4: Pixel Programs
- [ ] Upload os.pxi to GPU memory
- [ ] Trigger shader dispatch
- [ ] Execute pixel instructions
- [ ] Verify "Hello from GPU OS!" output

---

## 📊 Performance Targets

| Metric | Target | Why |
|--------|--------|-----|
| Mailbox latency | < 1μs | Real-time GPU-CPU sync |
| Command throughput | > 1M cmds/sec | Batched operations |
| Program upload | < 100μs for 64KB | Fast shader updates |
| CPU overhead | < 5% | Maintain GPU-centric design |

---

## 🔧 Debug Tools

### Memory Dump Tool
```nasm
; Dump BAR0 region to UART
mov esi, BAR0
mov ecx, 256  ; Dump 256 bytes
call hexdump_to_uart
```

### Mailbox Monitor
```nasm
; Print mailbox state every 100ms
.loop:
    mov eax, [BAR0 + 0x0000]
    call print_hex
    mov eax, 100
    call sleep_ms
    jmp .loop
```

### GPU Register Dump
```nasm
; Dump all GPU control registers
mov esi, BAR2
mov ecx, 64   ; 64 registers
call dump_registers
```

---

## 📁 Files to Create

| File | Purpose |
|------|---------|
| `bar_map.asm` | BAR0/BAR2 mapping code |
| `mailbox_hw.asm` | Hardware mailbox implementation |
| `cmdbuf.asm` | Command buffer producer |
| `pxi_upload.asm` | Pixel program upload |
| `gpu_dispatch.asm` | GPU shader dispatch |
| `test_mailbox_hw.sh` | Hardware mailbox tests |
| `test_cmdbuf.sh` | Command buffer tests |

---

## 🚀 Success Criteria

Phase 2 is complete when:

1. ✅ **BAR0 mapped** with correct cache attributes
2. ✅ **Mailbox working** on real hardware
3. ✅ **"Hello from GPU OS!"** executes on GPU shader
4. ✅ **Command buffer** processes 1M+ commands/sec
5. ✅ **Framebuffer updates** visible on display

---

**File**: `PHASE2_ARCHITECTURE.md`
**Branch**: `claude/pxos-gpu-centric-014hDDyJqnxLejmBJcAXbbN3`
**Status**: Design ready, pending GRUB boot validation
