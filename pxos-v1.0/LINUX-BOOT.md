# pxOS Linux Boot - Actually Boot Real Linux

This directory contains an enhanced version of pxOS that can **actually boot a real Linux kernel**.

## Overview

Instead of just simulating Linux boot messages, this version of pxOS acts as a real bootloader (like GRUB) that:

1. Loads a Linux kernel from disk into memory
2. Sets up the Linux boot protocol parameters
3. Switches the CPU to protected mode
4. Transfers control to the Linux kernel

## Architecture

```
┌─────────────────────────────────┐
│  BIOS (Real or QEMU)            │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  pxOS Bootloader (512 bytes)    │
│  - Load kernel from disk        │
│  - Setup boot parameters        │
│  - Enter protected mode         │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Linux Kernel (Real x86_64)     │
│  - Decompress itself            │
│  - Initialize hardware          │
│  - Mount root filesystem        │
│  - Start init process           │
└─────────────────────────────────┘
```

## Quick Start

### 1. Build the Bootable Image

```bash
./build-linux.sh
```

This will:
- Assemble the pxOS bootloader (512 bytes)
- Download a Tiny Core Linux kernel (~10MB)
- Create a minimal initrd with busybox
- Package everything into a bootable disk image

Output: `build/pxos-linux.img`

### 2. Test in QEMU

```bash
./test-linux.sh
```

Or manually:

```bash
qemu-system-x86_64 \
  -drive file=build/pxos-linux.img,format=raw \
  -m 512M \
  -serial stdio
```

### 3. Boot on Real Hardware (Advanced)

⚠️ **WARNING: This will overwrite the target device!**

```bash
# Find your USB device
lsblk

# Write the image (replace sdX with your device)
sudo dd if=build/pxos-linux.img of=/dev/sdX bs=4M status=progress

# Sync and eject
sudo sync
sudo eject /dev/sdX
```

Then boot from the USB device.

## Files

| File | Purpose |
|------|---------|
| `pxos-linux.asm` | Enhanced bootloader source code (NASM) |
| `build-linux.sh` | Build script to create bootable image |
| `test-linux.sh` | Test script to run in QEMU |
| `build/pxos-linux.bin` | Compiled 512-byte bootloader |
| `build/pxos-linux.img` | Complete bootable disk image |

## How It Works

### Boot Sector (pxos-linux.asm)

The bootloader performs these steps:

1. **Initialize**
   - Set up segments (DS, ES, SS)
   - Set up stack at 0x9000:0xFFFF
   - Save boot drive number

2. **Load Kernel**
   - Use BIOS INT 13h to read kernel sectors
   - Load to memory address 0x10000
   - Read 50 sectors (25KB) - enough for kernel setup code

3. **Setup Boot Parameters**
   - Create Linux boot protocol structure at 0x90000
   - Set "HdrS" signature and protocol version
   - Configure loader type and heap

4. **Enter Protected Mode**
   - Load GDT (Global Descriptor Table)
   - Enable PE bit in CR0 register
   - Far jump to 32-bit code

5. **Jump to Kernel**
   - Transfer control to kernel entry point
   - Linux takes over from here

### Disk Layout

```
Sector 0:     pxOS bootloader (512 bytes)
Sector 1-50:  Linux kernel (~25KB setup + compressed kernel)
Sector 10000: Initial ramdisk (initrd)
```

## Memory Map

```
0x00000000 - 0x000003FF  Real Mode IVT
0x00000400 - 0x000004FF  BIOS Data Area
0x00007C00 - 0x00007DFF  Boot Sector (pxOS)
0x00010000 - 0x0001FFFF  Kernel Setup Code
0x00090000 - 0x000901FF  Boot Parameters
0x00100000 - ...         Linux Kernel (after decompression)
```

## Requirements

### Build Requirements
- `nasm` - Netwide Assembler
- `wget` or `curl` - To download kernel
- `cpio` and `gzip` - To create initrd
- Standard Unix utilities (dd, stat)

Install on Ubuntu/Debian:
```bash
sudo apt install nasm qemu-system-x86 wget cpio gzip
```

### Runtime Requirements
- x86 or x86_64 CPU
- At least 64MB RAM (512MB recommended)
- BIOS-compatible boot (or QEMU)

## Customization

### Use a Different Kernel

Replace `vmlinuz` in the build directory:

```bash
# Copy your kernel
cp /path/to/your/vmlinuz build/vmlinuz

# Rebuild image
./build-linux.sh
```

### Modify Boot Parameters

Edit `setup_boot_params` in `pxos-linux.asm` to change:
- Kernel command line
- Video mode
- RAM disk location
- Heap size

### Change Kernel Load Size

Edit the constant in `pxos-linux.asm`:

```asm
%define KERNEL_SECTORS    50    ; Increase if kernel is larger
```

## Troubleshooting

### "DISK ERROR!" message

- Check that the kernel file exists in `build/vmlinuz`
- Verify disk image was built correctly
- Try rebuilding: `./build-linux.sh`

### Kernel doesn't boot

- Ensure you're using a compatible kernel (x86 or x86_64)
- Try a different kernel (Tiny Core Linux recommended)
- Increase memory: `qemu-system-x86_64 ... -m 1G`

### "Command not found: nasm"

Install build tools:
```bash
sudo apt install nasm build-essential
```

## Differences from Simulation

| Feature | Simulation | Real Boot |
|---------|-----------|-----------|
| Kernel | Pixel ISA VM | Real Linux (x86_64) |
| Boot Time | Instant | 5-30 seconds |
| Hardware | None | Real or emulated |
| Shell | Simulated | Real bash/sh |
| Commands | Limited | Full Linux |

## Next Steps

### Add More Features

1. **Multi-boot menu**
   - Display menu with multiple kernels
   - User can select which to boot

2. **Filesystem support**
   - Read kernel from FAT32/ext2
   - No hardcoded sector numbers

3. **Command line editor**
   - Allow user to edit kernel parameters
   - Pass custom arguments to kernel

4. **Graphical boot**
   - Show logo during boot
   - Progress bar

### Educational Extensions

1. **Add debug output**
   - Print memory addresses
   - Show GDT contents
   - Display registers before jump

2. **Step-through mode**
   - Pause at each boot stage
   - Show what's happening

3. **Interactive tutorial**
   - Explain each step as it runs
   - Link to documentation

## Resources

- [Linux x86 Boot Protocol](https://www.kernel.org/doc/html/latest/x86/boot.html)
- [OSDev Wiki - Bare Bones](https://wiki.osdev.org/Bare_Bones)
- [Intel x86 Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [NASM Documentation](https://www.nasm.us/docs.php)

## License

MIT License - Same as pxOS

---

**Made with ❤️ for educational purposes**

*"From pixels to real hardware - bridging the gap between simulation and reality"*
