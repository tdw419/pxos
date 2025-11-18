# pxHV Quick Start Guide

## What You Have

You now have a **working bare-metal hypervisor** that:

✅ Boots from disk
✅ Enters 64-bit long mode
✅ Enables Intel VT-x
✅ Executes VMXON successfully

**This is Stages 1 and 2 complete!**

## Test It Right Now

```bash
chmod +x build_pxhv.sh
make

# If you have QEMU with KVM support:
make run-kvm

# Or without KVM:
make run
```

**Expected output:**
```
pxHV: Pixel Hypervisor v0.1
Jumping to hypervisor...
Stage 2: Hypervisor loader
VT-x supported
Long mode enabled
VMXON executed successfully!
```

If you see "VMXON executed successfully!" → **IT WORKS!** 🎉

## What's Missing (Stage 3)

To actually boot a guest OS, you need:

### 1. VMCS (Virtual Machine Control Structure) Setup
This is the "control panel" for the virtual machine. It contains:
- **Guest state**: Where guest execution starts (RIP), stack (RSP), flags, etc.
- **Host state**: Where to return on VM exit
- **Execution controls**: What events cause VM exits (I/O, CPUID, etc.)

### 2. EPT (Extended Page Tables)
Maps guest physical addresses to host physical addresses.
- Guest thinks it has memory at 0x0
- Host actually gives it memory at 0x200000
- EPT does the translation transparently

### 3. Guest Loading
Load a kernel (Linux bzImage or FreeDOS) into memory where the guest expects it.

### 4. VMLAUNCH
Execute the VMLAUNCH instruction to start the guest running.

### 5. VM Exit Handler
Handle events when the guest needs help:
- I/O instructions (guest wants to talk to hardware)
- CPUID (guest wants CPU info)
- HLT (guest is idle)
- EPT violations (guest accessed unmapped memory)

## Implementation Roadmap

### Phase 1: Minimal Guest (1-2 days)

**Goal**: Run the simplest possible guest code

Create `pxhv_stage3.asm` with:

```asm
BITS 64

stage3_entry:
    ; 1. Allocate and clear VMCS region
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

    ; 4. Place HLT instruction at guest entry point
    mov rax, 0xF4            ; HLT opcode
    mov [0x200000], al

    ; 5. Write mandatory VMCS fields
    ; Guest RIP
    mov rax, 0x200000
    mov rdx, 0x681E          ; GUEST_RIP encoding
    vmwrite rdx, rax

    ; Guest RSP
    mov rax, 0x7000
    mov rdx, 0x681C          ; GUEST_RSP encoding
    vmwrite rdx, rax

    ; Guest RFLAGS
    mov rax, 0x2             ; Bit 1 is always set
    mov rdx, 0x6820          ; GUEST_RFLAGS encoding
    vmwrite rdx, rax

    ; ... (more fields - see full implementation)

    ; 6. VMLAUNCH
    vmlaunch

    ; If we get here, launch failed
    jmp vm_launch_error
```

**Success criteria**: VMLAUNCH executes without VMfailInvalid/VMfailValid

### Phase 2: Handle VM Exits (2-3 days)

```asm
vm_exit_handler:
    ; Read VM exit reason
    mov rdx, 0x4402          ; VM_EXIT_REASON encoding
    vmread rax, rdx

    ; Check if it's HLT (exit reason = 12)
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

FreeDOS is easier than Linux because:
- No UEFI/boot protocol complexity
- Expects real mode (no need for protected mode setup)
- Small (~100KB)

Steps:
1. Download FreeDOS kernel (kernel.sys)
2. Load it to guest memory at 0x200000
3. Setup guest in real mode (CR0.PE = 0)
4. Set guest RIP to 0x200000
5. VMLAUNCH
6. Handle I/O VM exits

**Success criteria**: FreeDOS boots to command prompt

### Phase 4: Boot Linux (1-2 weeks)

1. Download minimal Linux kernel (bzImage)
2. Implement Linux boot protocol
3. Pass command line: "console=ttyS0"
4. Load initrd (optional)
5. Setup guest in protected mode
6. VMLAUNCH
7. Handle many more VM exits

**Success criteria**: Linux boots to shell

## VMCS Field Reference (Essential)

You MUST set these fields for VMLAUNCH to succeed:

### Guest State Area
```c
// Control registers
GUEST_CR0       = 0x6800
GUEST_CR3       = 0x6802
GUEST_CR4       = 0x6804

// Instruction/stack pointers
GUEST_RIP       = 0x681E
GUEST_RSP       = 0x681C
GUEST_RFLAGS    = 0x6820

// Segment registers
GUEST_CS_SELECTOR    = 0x0802
GUEST_CS_BASE        = 0x6808
GUEST_CS_LIMIT       = 0x4800
GUEST_CS_AR_BYTES    = 0x4816
// DS, ES, FS, GS, SS follow same pattern

// Descriptor tables
GUEST_GDTR_BASE      = 0x6816
GUEST_GDTR_LIMIT     = 0x4810
GUEST_IDTR_BASE      = 0x6818
GUEST_IDTR_LIMIT     = 0x4812
```

### Host State Area
```c
// Where to return on VM exit
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

// Read allowed bits from MSRs:
// IA32_VMX_PINBASED_CTLS       = 0x481
// IA32_VMX_PROCBASED_CTLS      = 0x482
// IA32_VMX_PROCBASED_CTLS2     = 0x48B
```

### Entry/Exit Controls
```c
VM_EXIT_CONTROLS     = 0x400C
VM_ENTRY_CONTROLS    = 0x4012
```

**Pro tip**: Read Intel SDM Vol 3C, Appendix B for all field encodings!

## Debugging Tips

### 1. VMLAUNCH Fails with VMfailInvalid
- Check VMCS revision ID matches CPU
- Verify VMPTRLD succeeded
- Make sure VMCS region is 4KB aligned

### 2. VMLAUNCH Fails with VMfailValid
- Read VM_INSTRUCTION_ERROR field (encoding 0x4400)
- Common errors:
  - Error 7: Invalid host state
  - Error 8: Invalid guest state
  - Error 12: Invalid control fields

### 3. Guest Doesn't Execute
- Print guest RIP after VMLAUNCH
- Verify guest memory is readable
- Check that guest code is actually there

### 4. Unexpected VM Exits
- Always print exit reason
- Implement handlers for common exits first (HLT, CPUID, I/O)

## Resources

### Must Read
- **Intel SDM Volume 3C**: Chapters 23-33 (VMX architecture)
- **Appendix B**: VMCS field encodings

### Code Examples
- Search GitHub for "simple hypervisor" or "bare metal VMX"

### OSDev Wiki
- https://wiki.osdev.org/ - Great for low-level x86 details

## Testing Checklist

After each phase, verify:

- [ ] Builds without errors
- [ ] Boots in QEMU
- [ ] Prints expected messages
- [ ] Doesn't triple-fault
- [ ] VMLAUNCH succeeds (or fails with understandable error)

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

[    0.000000] Linux version 5.15.0 (...)
[    0.000000] Command line: console=ttyS0
...
login: root
# uname -a
Linux pxhv-guest 5.15.0 x86_64 GNU/Linux
#
```

**That's a working hypervisor in ~10KB!**

---

You now have:
1. ✅ Working boot sector
2. ✅ Long mode transition
3. ✅ VT-x enabled
4. ✅ VMXON executed
5. 🚧 VMCS setup (next)
6. 🚧 Guest launch (after that)

**Go forth and virtualize!** 🚀
