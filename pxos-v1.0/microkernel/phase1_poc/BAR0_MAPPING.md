# GPU BAR0 Memory Mapping for pxOS

## Overview

This document describes the BAR0 mapping implementation in pxOS Phase 2, which enables the CPU microkernel to communicate with the GPU through Memory-Mapped I/O (MMIO).

## Architecture

### Memory Regions

| Region | Address | Size | Purpose |
|--------|---------|------|---------|
| **BAR0 Physical** | 0xE0000000 | 16 MB | GPU control registers (physical) |
| **BAR0 Virtual** | 0xE0000000 | 16 MB | GPU control registers (mapped) |
| **Mailbox** | BAR0 + 0x0000 | 16 bytes | CPU-GPU command interface |

### Mapping Strategy

**Current Implementation: Identity Mapping**

For Phase 2, BAR0 is identity-mapped (virtual address = physical address). This simplifies early bring-up and debugging.

```
Physical: 0xE0000000 → Virtual: 0xE0000000
```

**Future: High-Half Kernel Mapping**

For production, BAR0 will be mapped into the kernel's high-half virtual address space:

```
Physical: 0xE0000000 → Virtual: 0xFFFF800000000000 + offset
```

## Implementation

### 1. PCIe Discovery

The PCIe enumeration code scans bus 0, devices 0-31 to find the GPU (VGA controller, class 0x0300):

```nasm
; In pcie_scan_64:
; Build config address for BAR0
mov eax, 0x80000000    ; Enable bit
mov ecx, device_num
shl ecx, 11            ; Device << 11
or eax, ecx
or eax, 0x10           ; Offset 0x10 (BAR0)

; Read BAR0 from PCI config space
mov dx, 0xCF8          ; Config address port
out dx, eax
mov dx, 0xCFC          ; Config data port
in eax, dx

; Mask off flags, save physical address
and eax, 0xFFFFFFF0
mov [gpu_bar0_phys], rax
```

### 2. Memory Mapping

After PCIe scan completes, BAR0 is mapped into kernel virtual address space:

```nasm
; In map_gpu_bar0:
mov rax, [gpu_bar0_phys]
test rax, rax
jz .no_bar

; Identity map for simplicity
mov [gpu_bar0_virt], rax

; Map first 4KB page with UC (uncacheable) attributes
mov rdi, rax           ; phys_addr
mov rsi, rax           ; virt_addr (identity)
mov rdx, PAT_UC        ; Uncacheable for MMIO
call map_mmio_page
```

### 3. Page Table Entry Flags

For MMIO regions, the following page table flags are used:

| Flag | Value | Purpose |
|------|-------|---------|
| **Present** | 0x001 | Page is present in memory |
| **Write** | 0x002 | Writable (required for MMIO) |
| **Cache-Disable** | 0x010 | Disable caching (UC) |
| **Writethrough** | 0x008 | Write-through policy |

Combined flags for BAR0: `0x001 | 0x002 | 0x010 | 0x008 = 0x01B`

### 4. Usage Example

Once BAR0 is mapped, the mailbox protocol can use it:

```nasm
; Write to mailbox (BAR0 + 0x0000)
mov rdi, [gpu_bar0_virt]
test rdi, rdi
jz .no_hardware

mov dword [rdi], eax   ; Write command
mfence                 ; Ensure write completes
```

## Memory Attributes

### PAT (Page Attribute Table) Configuration

pxOS configures the PAT to support different memory types:

| Index | Type | Use Case |
|-------|------|----------|
| 0 | UC (Uncacheable) | Control registers, mailbox |
| 1 | WC (Write-Combining) | GPU framebuffer |
| 2 | WT (Write-Through) | Shared buffers |
| 3 | WB (Write-Back) | Normal memory |

### Why UC for Mailbox?

**Uncacheable (UC)** is required for the mailbox region because:

1. **Synchronization** - GPU writes must be immediately visible to CPU
2. **Ordering** - Commands must execute in program order
3. **Side Effects** - MMIO reads/writes can trigger hardware state changes
4. **No Speculation** - CPU must not prefetch or reorder MMIO operations

### Why WC for Framebuffer?

**Write-Combining (WC)** is optimal for the framebuffer because:

1. **Batching** - Multiple pixel writes can be combined
2. **Bandwidth** - Reduces memory bus traffic
3. **Performance** - 2-3x faster than UC for large transfers

## Boot Sequence

The complete boot sequence with BAR0 mapping:

```
1. BIOS/GRUB loads kernel
2. Kernel enters 32-bit protected mode
3. Setup page tables (identity map first 1GB)
4. Enter 64-bit long mode
5. Scan PCIe bus → discover GPU → read BAR0 physical address
6. Map BAR0 into kernel virtual address space (UC)
7. Initialize mailbox protocol using BAR0 virtual address
8. GPU is ready for commands
```

## Verification

### VGA Status Markers

The kernel displays status markers on VGA screen:

| Marker | Meaning |
|--------|---------|
| M | Multiboot entry successful |
| L | Long mode activated |
| P | PCIe scan started |
| G | GPU found, BAR0 read |
| **(B)** | **BAR0 mapped successfully** |

### Serial Debug Output

Serial port (COM1) shows detailed mapping info:

```
pxOS CPU Microkernel v0.4
Entering Long Mode...
Scanning PCIe bus 0...
Mapping GPU BAR0... OK (virt=0xE0000000)
Hello from GPU OS!
```

## Error Handling

### Graceful Degradation

If BAR0 mapping fails (GPU not present, mapping error), the system:

1. Detects null `gpu_bar0_virt`
2. Falls back to simulation mode
3. Continues booting with reduced functionality
4. Logs error to serial port

```nasm
; All MMIO access code checks for null
mov rdi, [gpu_bar0_virt]
test rdi, rdi
jz .use_simulation     ; Fallback path
```

## Performance

### Memory Access Latency

| Operation | Cached (WB) | Uncached (UC) | Write-Combining (WC) |
|-----------|-------------|---------------|---------------------|
| Read | ~4 cycles | ~200 cycles | ~200 cycles |
| Write | ~4 cycles | ~200 cycles | ~50 cycles |
| Burst Write (64B) | ~16 cycles | ~12,800 cycles | ~800 cycles |

### MMIO Access Patterns

**Good: Minimize round-trips**
```nasm
; Write 4 commands in batch
mov rdi, [gpu_bar0_virt]
mov [rdi + 0], eax
mov [rdi + 4], ebx
mov [rdi + 8], ecx
mov [rdi + 12], edx
mfence
```

**Bad: Excessive fencing**
```nasm
; Each write has separate fence - slow!
mov [rdi], eax
mfence
mov [rdi + 4], ebx
mfence
```

## Future Enhancements

### Phase 3 Improvements

1. **Multi-page mapping** - Map entire 16MB BAR0 region
2. **High-half kernel** - Map BAR0 above 0xFFFF800000000000
3. **TLB optimization** - Use huge pages (2MB) for framebuffer
4. **DMA buffers** - Allocate contiguous physical memory for GPU DMA
5. **IOMMU support** - Enable IOMMU for GPU address translation

### Example: Multi-page Mapping

```nasm
map_gpu_bar0_full:
    mov rcx, 16                ; 16 MB = 4096 pages
    mov rax, [gpu_bar0_phys]
    mov rbx, [gpu_bar0_virt]

.map_loop:
    mov rdi, rax
    mov rsi, rbx
    mov rdx, PAT_UC
    call map_mmio_page

    add rax, 0x1000            ; Next 4KB
    add rbx, 0x1000
    loop .map_loop
```

## Debugging

### Common Issues

**1. Triple Fault After Mapping**

- Check that page tables cover BAR0 address range
- Verify PML4/PDP/PD entries are present
- Ensure BAR0 address is aligned to 4KB

**2. MMIO Writes Not Visible to GPU**

- Verify UC attribute is set (not WB)
- Add `mfence` after writes
- Check that BAR0 address is correct

**3. BAR0 = 0x00000000**

- PCIe scan failed to find GPU
- Check QEMU device configuration
- Verify PCI config space access works

### Debug Techniques

**Print BAR0 address**
```nasm
mov rax, [gpu_bar0_phys]
call print_hex_64
```

**Verify page table entry**
```nasm
; Walk page tables manually
mov rax, [gpu_bar0_virt]
shr rax, 39
and rax, 0x1FF         ; PML4 index
; Read PML4[index] and verify Present bit
```

**Test MMIO access**
```nasm
; Write pattern, read back
mov rdi, [gpu_bar0_virt]
mov dword [rdi], 0x12345678
mfence
mov eax, [rdi]
cmp eax, 0x12345678
jne .mmio_broken
```

## Resources

- [OSDev Wiki - PCI](https://wiki.osdev.org/PCI)
- [OSDev Wiki - Paging](https://wiki.osdev.org/Paging)
- [Intel SDM Vol 3A - Memory Types](https://www.intel.com/content/www/us/en/architecture-and-technology/64-ia-32-architectures-software-developer-manual-325462.html)
- [Write-Combining Memory](https://en.wikipedia.org/wiki/Write_combining)

---

**pxOS Phase 2 - GPU-Centric OS Architecture** 🚀

*"Map it, use it, revolutionize it"* - pxOS Team
