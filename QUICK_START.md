# 🚀 Quick Start: Boot Linux with Virtio Detection

## What You Need

1. **QEMU** - Virtual machine emulator
2. **TinyCore Linux** - Lightweight Linux kernel (already downloaded ✅)

## Steps to Success

### **Step 1: Install QEMU** (if not already installed)

See [INSTALL_QEMU.md](INSTALL_QEMU.md) for your platform.

Quick command for Ubuntu/Debian:
```bash
sudo apt install qemu-system-x86
```

### **Step 2: Boot Linux**

```bash
./boot_linux_virtio.sh
```

### **Step 3: See Virtio Detection**

Look in the output or check the log:
```bash
grep "1af4:1003" boot_output.log
```

You should see:
```
pci 0000:00:04.0: [1af4:1003] type 00 class 0x078000
```

**This means Linux detected the Virtio console device!** ✅

## What's Happening

1. QEMU creates a virtual machine
2. Linux kernel boots in the VM
3. Kernel scans PCI bus for devices
4. Finds Virtio console at `1af4:1003`
5. Loads appropriate driver automatically

## Files in This Repo

- **download_tinycore.sh** - Downloads TinyCore Linux kernel ✅ Done
- **boot_linux_virtio.sh** - Boots Linux with Virtio device
- **VIRTIO_BOOT_GUIDE.md** - Detailed guide on Virtio
- **INSTALL_QEMU.md** - QEMU installation instructions
- **vmlinuz64** - Linux kernel (5.3 MB) ✅
- **corepure64.gz** - Initial ramdisk (13 MB) ✅

## Manual Boot Command

If you want to run QEMU directly:

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

Press Ctrl+C to stop the VM.

## What Virtio Devices Look Like

When you see device IDs like `1af4:XXXX`, that's a Virtio device:

- **1af4:1000** - Virtio network card
- **1af4:1001** - Virtio block device (disk)
- **1af4:1003** - Virtio console (what we're using)
- **1af4:1005** - Virtio RNG (random numbers)
- **1af4:1009** - Virtio filesystem

All Virtio devices use vendor ID `1af4` (Red Hat).

## Troubleshooting

**"qemu-system-x86_64 not found":**
- Install QEMU (see INSTALL_QEMU.md)

**"No kernel found":**
- Run `./download_tinycore.sh` first

**"Timeout":**
- Increase timeout in boot_linux_virtio.sh (change 10 to 20)

**No Virtio detection:**
- Check boot_output.log for errors
- Make sure `-device virtio-serial-pci` is in QEMU command

## Next Steps

Once you see Virtio working:

1. Try other Virtio devices (network, block storage)
2. Write a simple kernel module that interacts with Virtio
3. Test custom driver code
4. Explore Virtio specifications

## Learn More

- **Virtio Spec**: https://docs.oasis-open.org/virtio/virtio/v1.1/virtio-v1.1.html
- **QEMU Docs**: https://www.qemu.org/docs/master/
- **Linux Virtio**: https://www.kernel.org/doc/html/latest/driver-api/virtio/virtio.html
