# GRUB Multiboot Quick Start Guide

**Date**: 2025-11-18
**Status**: Ready to test (needs GRUB installation)
**Estimated Time**: 5 minutes

---

## ✅ Why GRUB?

Your custom bootloader has a subtle ES segment corruption bug during mode transitions. GRUB bypasses this entirely by:
- ✅ Handling all CPU mode transitions (Real → Protected → Long)
- ✅ Setting up proper page tables
- ✅ Loading your kernel as a standard ELF binary
- ✅ Used by Linux, BSD, and virtually all major operating systems

**Bottom line**: GRUB gets you testing on QEMU and bare metal immediately.

---

## 🚀 Installation (One-Time Setup)

```bash
sudo apt-get update
sudo apt-get install -y grub-pc-bin xorriso
```

**Time**: ~2 minutes

---

## 🔨 Build and Test

### Quick Test (Recommended)
```bash
cd /home/user/pxos/pxos-v1.0/microkernel/phase1_poc
./test_grub_multiboot.sh
```

### Manual Build
```bash
cd /home/user/pxos/pxos-v1.0/microkernel/phase1_poc

# 1. Build multiboot kernel
nasm -f elf32 microkernel_multiboot.asm -o build/microkernel_multiboot.o
ld -m elf_i386 -T linker.ld -o build/pxos_multiboot.elf build/microkernel_multiboot.o

# 2. Create ISO structure
mkdir -p iso/boot/grub
cp build/pxos_multiboot.elf iso/boot/pxos.elf

# 3. Create GRUB config
cat > iso/boot/grub/grub.cfg << 'EOF'
set timeout=0
set default=0

menuentry "pxOS Phase 1 POC" {
    multiboot /boot/pxos.elf
    boot
}
EOF

# 4. Generate bootable ISO
grub-mkrescue -o build/pxos.iso iso/

# 5. Test in QEMU
qemu-system-x86_64 -cdrom build/pxos.iso -m 512M -nographic
```

---

## 📊 Expected Output

### If Everything Works ✅
```
pxOS CPU Microkernel v0.3
Initializing GPU drivers...
Scanning PCIe bus 0...
  Device 00:00.0: Vendor=8086 Device=1237 Class=0600 (Host bridge)
  Device 00:01.0: Vendor=8086 Device=7000 Class=0601 (ISA bridge)
  Device 00:02.0: Vendor=1234 Device=1111 Class=0300 (VGA compatible)
    BAR0: 0xE0000000 (size 16MB, MMIO)
    BAR2: 0xFEBF0000 (size 4KB, MMIO)
GPU found at: 00:02.0
GPU BAR0: 0xE0000000
Executing GPU program...
Hello from GPU OS!
System halted.
```

### If Boot Fails ❌
Check the boot sequence:
- GRUB menu appears? → Kernel loading issue
- Triple fault/reboot? → Page table or long mode issue
- Nothing appears? → Multiboot header incorrect

---

## 🐛 Troubleshooting

### Issue: "grub-mkrescue not found"
**Fix**: Install GRUB tools (see Installation section above)

### Issue: "ld: cannot find -lelf_i386"
**Fix**: You might need to use `ld` without `-m elf_i386`:
```bash
ld -T linker.ld -o build/pxos_multiboot.elf build/microkernel_multiboot.o
```

### Issue: QEMU shows "No bootable device"
**Fix**: Use `-cdrom` instead of `-drive`:
```bash
qemu-system-x86_64 -cdrom build/pxos.iso -m 512M
```

### Issue: Boot succeeds but no output
**Check**:
1. Microkernel UART output working? (Check `microkernel_multiboot.asm:print_string`)
2. VGA output configured? (Should write to 0xB8000)
3. Serial port configured? (Port 0x3F8)

---

## 📁 Files Involved

| File | Purpose |
|------|---------|
| `microkernel_multiboot.asm` | Multiboot-compatible kernel (32→64 transition) |
| `linker.ld` | ELF linker script (defines memory layout) |
| `test_grub_multiboot.sh` | Automated build and test script |
| `build/pxos_multiboot.elf` | Compiled multiboot kernel (ELF format) |
| `build/pxos.iso` | Bootable ISO image |
| `iso/boot/grub/grub.cfg` | GRUB configuration |

---

## 🎯 What This Unlocks

Once GRUB boots successfully, you can:

### ✅ Test PCIe Enumeration
- See real PCIe devices discovered
- Verify BAR0 addresses
- Validate device class codes

### ✅ Test GPU Discovery
- Detect VGA controller (00:02.0 in QEMU)
- Read BAR0 MMIO address
- Prepare for framebuffer mapping

### ✅ Test on Real Hardware
- Write ISO to USB stick: `dd if=build/pxos.iso of=/dev/sdX bs=4M`
- Boot from USB
- See pxOS on bare metal!

### ✅ Begin Phase 2 Development
- BAR memory mapping (MMIO regions)
- GPU command buffer setup
- Real hardware mailbox protocol

---

## 🔄 Comparison: Custom Bootloader vs GRUB

| Aspect | Custom Bootloader | GRUB Multiboot |
|--------|-------------------|----------------|
| **Boot Success** | ❌ Triple fault at PAE | ✅ Proven reliable |
| **Development Time** | 🕐 4-8 hours to debug | ✅ 5 minutes |
| **Real Hardware** | ❌ Broken | ✅ Works |
| **QEMU Testing** | ❌ Broken | ✅ Works |
| **Learning Value** | ✅ High (x86 internals) | ⚠️ Medium |
| **Production Ready** | ❌ No | ✅ Yes |

---

## 📝 Next Steps After GRUB Works

1. **Verify PCIe enumeration** - See device discovery working
2. **Map GPU BAR0** - Set up MMIO access to framebuffer
3. **Test mailbox protocol** - CPU-GPU communication on real hardware
4. **Phase 2 development** - Command buffers, GPU dispatch

---

## 💡 Pro Tips

### For Faster Iteration
Use QEMU with serial output:
```bash
qemu-system-x86_64 -cdrom build/pxos.iso -m 512M -nographic -serial mon:stdio
```

### For Debugging
Enable QEMU logging:
```bash
qemu-system-x86_64 -cdrom build/pxos.iso -m 512M -d int,cpu_reset -D qemu.log
```

### For Real Hardware Testing
Check ISO integrity:
```bash
md5sum build/pxos.iso
# Verify checksum matches after writing to USB
```

---

## ✅ Success Criteria

You know GRUB boot worked when you see:
1. **"pxOS CPU Microkernel"** banner
2. **"Scanning PCIe bus 0..."** message
3. **Device enumeration** with vendor/device IDs
4. **"Hello from GPU OS!"** output

---

**File**: `GRUB_QUICKSTART.md`
**Branch**: `claude/pxos-gpu-centric-014hDDyJqnxLejmBJcAXbbN3`
**Status**: Ready for testing once GRUB is installed
