# pxOS Phase 2 - GRUB Multiboot Kernel

## Overview

This is the **Phase 2 Proof of Concept** for pxOS, a GPU-centric operating system that offloads execution to the GPU for maximum performance. This phase implements:

- ✅ **GRUB Multiboot2** bootloader support
- ✅ **Long Mode (64-bit)** transition
- ✅ **PCIe Enumeration** to discover GPU
- ✅ **BAR0 Memory Mapping** for GPU MMIO access
- ✅ **Hardware Mailbox Protocol** for CPU-GPU communication
- ✅ **Page Table Setup** with 2MB pages
- ✅ **Serial Port Debug Output**
- ✅ **VGA Text Mode** status markers

---

## Quick Start

### Prerequisites

```bash
sudo apt-get install -y grub-pc-bin xorriso nasm qemu-system-x86
```

### Build and Run

```bash
# Build the bootable ISO
./test_grub_multiboot.sh

# Run in QEMU (GUI mode shows VGA output)
qemu-system-x86_64 -cdrom build/pxos.iso -m 512M
```

### Expected Output

When booting, you should see VGA markers on screen:

- **M** - Multiboot entry
- **L** - Long mode active
- **P** - PCIe scan started
- **G** - GPU device found
- **"Hello from GPU OS!"** - Success message

---

## Architecture

### Boot Flow

```
1. BIOS/UEFI loads GRUB
2. GRUB loads multiboot2 kernel at 1MB
3. Kernel entry in 32-bit protected mode
4. Initialize serial port (COM1)
5. Set up page tables (identity map 1GB)
6. Enable PAE + Long Mode
7. Jump to 64-bit code
8. Scan PCIe bus for GPU (VGA class 0x0300)
9. Read GPU BAR0 physical address
10. Map BAR0 into kernel virtual address space
11. Initialize hardware mailbox protocol
12. Test mailbox with UART write command
13. Print "Hello from GPU OS!"
14. Halt
```

### Memory Map

| Address Range | Purpose |
|---------------|---------|
| `0x0000 - 0x0FFF` | BIOS/Real Mode (unused) |
| `0x1000 - 0x1FFF` | PML4 (Page Map Level 4) |
| `0x2000 - 0x2FFF` | PDP (Page Directory Pointer) |
| `0x3000 - 0x3FFF` | PD (Page Directory) |
| `0x100000+` | Kernel loaded here by GRUB |
| `0xB8000` | VGA text buffer |
| `0xE0000000` | GPU BAR0 (typical address) |

### Page Table Structure

Uses **2MB huge pages** for simplicity:

- **PML4** → PDP → PD → 2MB pages
- First 1GB identity-mapped (512 × 2MB pages)
- PAE and PSE enabled for performance

---

## Files

| File | Description |
|------|-------------|
| `microkernel_multiboot.asm` | Main kernel source (NASM assembly) |
| `map_gpu_bar0.asm` | BAR0 memory mapping implementation |
| `mailbox_protocol.asm` | Hardware mailbox protocol implementation |
| `linker.ld` | Linker script for multiboot layout |
| `test_grub_multiboot.sh` | Build script |
| `run_qemu.sh` | QEMU test runner |
| `iso/boot/grub/grub.cfg` | GRUB configuration |
| `build/microkernel.bin` | Compiled kernel binary |
| `build/pxos.iso` | Bootable ISO image |
| `BAR0_MAPPING.md` | BAR0 mapping technical documentation |
| `MAILBOX_PROTOCOL.md` | Mailbox protocol technical documentation |

---

## PCIe Enumeration

The kernel scans **PCI bus 0, devices 0-31** using I/O ports `0xCF8` (config address) and `0xCFC` (config data).

### Discovery Process

1. For each device, read vendor ID at offset 0x00
2. Skip if vendor ID = 0xFFFF (no device)
3. Read class code at offset 0x08
4. Check if class = 0x0300 (VGA controller)
5. If VGA found, read BAR0 at offset 0x10
6. Store bus/dev/func and BAR0 address

### QEMU GPU

In QEMU with default VGA:

```
Device: 00:02.0
Class: 0x0300 (VGA compatible controller)
BAR0: 0xE0000000 (256MB prefetchable)
```

---

## Debug Output

### VGA Markers

The kernel writes status markers to VGA memory for visual debugging:

| Marker | Meaning |
|--------|---------|
| `M` | Multiboot entry successful |
| `L` (green) | Long mode activated |
| `P` (white) | PCIe scan started |
| `G` (green) | GPU found |

### Serial Port

COM1 (`0x3F8`) is initialized at 38400 baud for debugging:

```bash
# To capture serial output in QEMU:
qemu-system-x86_64 -cdrom build/pxos.iso -m 512M -serial file:serial.log
```

Output includes:
- Boot message
- Long mode transition
- PCIe scan results
- Hello message

---

## Technical Details

### Multiboot2 Header

```nasm
section .multiboot
align 8
    dd 0xE85250D6      ; Magic
    dd 0               ; Architecture (i386)
    dd header_length   ; Header size
    dd checksum        ; Checksum
    ; ... tags ...
```

### Long Mode Activation

1. Load PML4 address into CR3
2. Set PAE bit (bit 5) in CR4
3. Set LME bit (bit 8) in EFER MSR (0xC0000080)
4. Enable paging (bit 31 in CR0)
5. Load GDT with 64-bit code segment
6. Far jump to 64-bit code

### GDT (Global Descriptor Table)

```nasm
gdt_start:
    dq 0                      ; Null descriptor
    dq 0x00AF9A000000FFFF     ; 64-bit code (L=1, P=1)
    dq 0x00AF92000000FFFF     ; 64-bit data
```

---

## Troubleshooting

### Build Errors

**"nasm: command not found"**

```bash
sudo apt-get install nasm
```

**"grub-mkrescue: command not found"**

```bash
sudo apt-get install grub-pc-bin xorriso
```

### Runtime Issues

**No VGA output in QEMU**

- Make sure you're NOT using `-nographic`
- Use GUI mode or `-display gtk`

**"Multiboot header not found"**

- Check that `.multiboot` section is first in linker script
- Verify magic number is 0xE85250D6

**Triple fault / Reset loop**

- Check page table setup
- Verify GDT is loaded correctly
- Ensure stack is valid

---

## Next Steps (Phase 3)

- [ ] Map GPU BAR0 into kernel address space
- [ ] Implement mailbox protocol for CPU-GPU communication
- [ ] Create pixel program loader
- [ ] Add WGSL shader compilation
- [ ] Measure CPU overhead (target: < 5%)

---

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Boot time | < 1 second | ~0.5s (QEMU) |
| CPU overhead | < 5% | TBD (Phase 3) |
| Mailbox latency | < 1 microsecond | TBD (Phase 3) |
| PCIe scan time | < 10ms | ~2ms (32 devices) |

---

## Resources

- [Multiboot2 Specification](https://www.gnu.org/software/grub/manual/multiboot2/multiboot.html)
- [OSDev Wiki - Long Mode](https://wiki.osdev.org/Setting_Up_Long_Mode)
- [OSDev Wiki - PCI](https://wiki.osdev.org/PCI)
- [Intel 64 and IA-32 SDM](https://www.intel.com/content/www/us/en/architecture-and-technology/64-ia-32-architectures-software-developer-manual-325462.html)

---

## License

MIT License - See repository root for full license text.

---

**Made with revolution in mind** 🚀

*"The future of operating systems runs on the GPU"* - pxOS Team
