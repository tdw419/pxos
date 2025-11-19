# 🚀 How to Boot Linux with Virtio Device Detection

## Quick Start (3 Steps)

### **Step 1: Download TinyCore Linux**
```bash
./download_tinycore.sh
```
This downloads a tiny (10MB) Linux kernel perfect for testing.

### **Step 2: Boot with Virtio Device**
```bash
./boot_linux_virtio.sh
```
This boots Linux in QEMU with a Virtio console device attached.

### **Step 3: Check Virtio Detection**
```bash
grep "1af4:1003" boot_output.log
```
Look for: `pci 0000:00:04.0: [1af4:1003]` - this is the Virtio console!

---

## What You'll See

When Linux boots, it will detect the Virtio device:

```
pci 0000:00:04.0: [1af4:1003] type 00 class 0x078000
```

**This means:**
- `1af4` = Red Hat (Virtio vendor ID)
- `1003` = Virtio console device
- Successfully detected by Linux kernel!

---

## Manual QEMU Command

If you want to run it manually:

```bash
qemu-system-x86_64 \
  -kernel vmlinuz64 \
  -initrd corepure64.gz \
  -append 'console=ttyS0' \
  -device virtio-serial-pci \
  -m 512M \
  -display none \
  -serial stdio
```

---

## Understanding Virtio

**What is Virtio?**
- Para-virtualized device framework for VMs
- Faster than fully emulated hardware
- Standard for KVM/QEMU virtual devices

**Common Virtio Devices:**
- `1af4:1000` - Virtio network card
- `1af4:1001` - Virtio block device
- `1af4:1003` - Virtio console (what we're using)
- `1af4:1005` - Virtio RNG (random number generator)

**Why This Matters:**
- Real Linux kernel detects our virtual hardware
- Driver loads automatically (built into kernel)
- Foundation for custom device drivers

---

## Troubleshooting

**"No kernel found":**
```bash
./download_tinycore.sh
```

**"qemu-system-x86_64 not found":**
```bash
# Ubuntu/Debian:
sudo apt install qemu-system-x86

# Fedora/RHEL:
sudo dnf install qemu-system-x86
```

**"Timeout too short":**
Edit `boot_linux_virtio.sh` and increase timeout from 10 to 15 seconds.

---

## Next Steps

Once you see Virtio detection working:

1. **Explore other Virtio devices** - Try virtio-blk, virtio-net
2. **Build custom driver** - Load your own kernel module
3. **Test with real hardware** - Some servers support Virtio natively

---

## Files Created

- `download_tinycore.sh` - Downloads TinyCore Linux
- `boot_linux_virtio.sh` - Boots Linux with Virtio
- `boot_output.log` - Full boot log (created after running)
- `vmlinuz64` - Linux kernel (downloaded)
- `corepure64.gz` - Initial ramdisk (downloaded)
