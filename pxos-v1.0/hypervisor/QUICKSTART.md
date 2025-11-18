# pxHV Quick Start & Stage 3 Guide

## Current Status: ✅ Stages 1-2 Complete

You have a **working hypervisor** that:
- ✅ Boots from disk
- ✅ Enters 64-bit long mode
- ✅ Enables Intel VT-x
- ✅ Executes VMXON successfully

**Test it now:**
```bash
make run-kvm
```

**You should see:** "VMXON executed successfully!"

## What's Missing: Stage 3

To boot a guest OS, you need to implement:

### 1. VMCS (Virtual Machine Control Structure)

The VMCS is the "control panel" for the virtual machine.

**Essential fields to set:**

```c
// Guest state (where guest starts)
GUEST_RIP       = 0x681E    // → guest entry point
GUEST_RSP       = 0x681C    // → guest stack
GUEST_RFLAGS    = 0x6820    // → 0x2 (bit 1 always set)
GUEST_CR0       = 0x6800    // → guest CR0
GUEST_CR3       = 0x6802    // → guest page table
GUEST_CR4       = 0x6804    // → guest CR4

// Segment registers (CS, DS, ES, SS, FS, GS)
GUEST_CS_SELECTOR    = 0x0802
GUEST_CS_BASE        = 0x6808
GUEST_CS_LIMIT       = 0x4800
GUEST_CS_AR_BYTES    = 0x4816
// ... (repeat for DS, ES, SS, FS, GS)

// Host state (where to return on VM exit)
HOST_RIP        = 0x6C16    // → vm_exit_handler
HOST_RSP        = 0x6C14    // → host stack
HOST_CR0        = 0x6C00
HOST_CR3        = 0x6C02
HOST_CR4        = 0x6C04

// Controls
PIN_BASED_VM_EXEC_CONTROL   = 0x4000
CPU_BASED_VM_EXEC_CONTROL   = 0x4002
VM_EXIT_CONTROLS            = 0x400C
VM_ENTRY_CONTROLS           = 0x4012
```

### 2. EPT (Extended Page Tables)

Maps guest physical → host physical addresses.

```
Guest thinks: 0x0000_0000
Host gives:   0x0020_0000
EPT translates transparently
```

### 3. Guest Loading

Load a kernel into memory:
- **FreeDOS**: Easier, expects real mode
- **Linux bzImage**: Harder, needs boot protocol

### 4. VMLAUNCH

```asm
vmlaunch
jc .vmlaunch_failed   ; CF=1: VMfailInvalid
jz .vmlaunch_failed   ; ZF=1: VMfailValid
```

### 5. VM Exit Handler

```asm
vm_exit_handler:
    ; Read exit reason
    mov rdx, 0x4402          ; VM_EXIT_REASON
    vmread rax, rdx

    ; Handle common exits
    cmp rax, 12              ; HLT
    je handle_hlt
    cmp rax, 10              ; CPUID
    je handle_cpuid
    cmp rax, 30              ; I/O instruction
    je handle_io

    ; Resume guest
    vmresume
```

## Implementation Roadmap

### Phase 1: Minimal Guest (1-2 days)

**Goal**: Execute single HLT instruction

Create `pxhv_stage3.asm`:

```asm
BITS 64

stage3_entry:
    ; 1. Clear VMCS region
    mov rdi, VMCS_REGION
    mov rcx, 1024
    xor rax, rax
    rep stosq

    ; 2. Write VMCS revision ID
    mov ecx, 0x480           ; IA32_VMX_BASIC MSR
    rdmsr
    mov [VMCS_REGION], eax

    ; 3. Load VMCS pointer
    mov rax, VMCS_REGION
    vmptrld [rax]

    ; 4. Place HLT at guest entry
    mov byte [0x200000], 0xF4  ; HLT opcode

    ; 5. Setup minimal VMCS fields
    ; ... (see Intel SDM for all required fields)

    ; 6. VMLAUNCH
    vmlaunch

    ; If we get here, launch failed
    jmp vm_launch_error
```

**Success criteria**: VMLAUNCH executes without error

### Phase 2: Handle VM Exits (2-3 days)

```asm
vm_exit_handler:
    ; Read exit reason
    mov rdx, 0x4402
    vmread rax, rdx

    ; Check if HLT (reason = 12)
    cmp rax, 12
    je handle_hlt

    jmp unknown_exit

handle_hlt:
    mov rsi, msg_guest_halt
    call print_string_64
    cli
    hlt

msg_guest_halt: db 'Guest executed HLT!', 0
```

**Success criteria**: See "Guest executed HLT!" message

### Phase 3: Boot FreeDOS (3-5 days)

**Why FreeDOS first?**
- Smaller (~100KB vs Linux ~5MB)
- Expects real mode (simpler setup)
- No complex boot protocol

**Steps:**
1. Download FreeDOS kernel (`kernel.sys`)
2. Load to 0x200000
3. Setup guest in real mode (CR0.PE = 0)
4. Set guest RIP to 0x200000
5. VMLAUNCH
6. Handle I/O exits

**Success criteria**: FreeDOS boots to prompt

### Phase 4: Boot Linux (1-2 weeks)

**Steps:**
1. Download Linux bzImage
2. Implement Linux boot protocol
3. Pass kernel command line: `console=ttyS0`
4. Setup guest in protected mode
5. VMLAUNCH
6. Handle many VM exits

**Success criteria**: Linux boots to shell

## Essential VMCS Fields Reference

Read these from Intel SDM Vol 3C, Appendix B:

### Guest State Area
```c
GUEST_CR0       = 0x6800
GUEST_CR3       = 0x6802
GUEST_CR4       = 0x6804
GUEST_RIP       = 0x681E
GUEST_RSP       = 0x681C
GUEST_RFLAGS    = 0x6820

GUEST_CS_SELECTOR    = 0x0802
GUEST_CS_BASE        = 0x6808
GUEST_CS_LIMIT       = 0x4800
GUEST_CS_AR_BYTES    = 0x4816

GUEST_GDTR_BASE      = 0x6816
GUEST_GDTR_LIMIT     = 0x4810
GUEST_IDTR_BASE      = 0x6818
GUEST_IDTR_LIMIT     = 0x4812
```

### Host State Area
```c
HOST_CR0        = 0x6C00
HOST_CR3        = 0x6C02
HOST_CR4        = 0x6C04
HOST_RIP        = 0x6C16  // → vm_exit_handler
HOST_RSP        = 0x6C14  // → host stack
```

### VM-Execution Controls
```c
PIN_BASED_VM_EXEC_CONTROL     = 0x4000
CPU_BASED_VM_EXEC_CONTROL     = 0x4002
SECONDARY_VM_EXEC_CONTROL     = 0x401E
```

Read allowed bits from these MSRs:
- `IA32_VMX_PINBASED_CTLS` (0x481)
- `IA32_VMX_PROCBASED_CTLS` (0x482)
- `IA32_VMX_PROCBASED_CTLS2` (0x48B)

### Entry/Exit Controls
```c
VM_EXIT_CONTROLS     = 0x400C
VM_ENTRY_CONTROLS    = 0x4012
```

## Debugging Tips

### VMLAUNCH fails with VMfailInvalid
- Check VMCS revision ID
- Verify VMPTRLD succeeded
- Ensure VMCS is 4KB aligned

### VMLAUNCH fails with VMfailValid
Read `VM_INSTRUCTION_ERROR` (0x4400):
- Error 7: Invalid host state
- Error 8: Invalid guest state
- Error 12: Invalid control fields

### Guest doesn't execute
- Print guest RIP: `vmread rax, 0x681E`
- Verify guest code is there
- Check memory is readable

### Unexpected VM exits
- Always print exit reason
- Handle HLT, CPUID, I/O first

## Testing Checklist

After each phase:
- [ ] Builds without errors
- [ ] Boots in QEMU
- [ ] Prints expected messages
- [ ] VMLAUNCH succeeds
- [ ] Doesn't triple-fault

## The End Goal

```
$ make run-kvm
pxHV: Pixel Hypervisor v0.1
Jumping to hypervisor...
Stage 2: Hypervisor loader
VT-x supported
Long mode enabled
VMXON executed successfully!
Loading Linux kernel...
Guest launched!

[    0.000000] Linux version 5.15.0
[    0.000000] Command line: console=ttyS0
...
login: root
# uname -a
Linux pxhv-guest 5.15.0 x86_64 GNU/Linux
```

**That's a working hypervisor in ~10KB!**

## Resources

- **Intel SDM Vol 3C**: Required reading
  - Chapter 24: VMCS structure
  - Appendix B: Field encodings
- **OSDev Wiki**: Helpful for x86 details
- **GitHub**: Search for "simple hypervisor" examples

## Next Steps

1. Read Intel SDM Chapter 24
2. Create `pxhv_stage3.asm` skeleton
3. Implement minimal VMCS init
4. Test VMLAUNCH
5. Add VM exit handler
6. Boot trivial guest (HLT)
7. Boot FreeDOS
8. Boot Linux

**Time estimate**: 1-2 weeks to working Linux guest

---

**You're 60% done. The hard part (boot + VT-x) is complete!** 🚀
