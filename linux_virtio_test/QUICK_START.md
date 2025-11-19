# Quick Start - Linux + Virtio Test

## One-Time Setup

```bash
cd /home/user/pxos/linux_virtio_test
./setup_linux.sh
```

Downloads:
- `vmlinuz-test` (5.7MB) - TinyCore Linux kernel
- `corepure64.gz` (14MB) - Minimal initrd

---

## Run the Test

### Option 1: Direct QEMU Command

```bash
timeout 8 qemu-system-x86_64 \
  -kernel vmlinuz-test \
  -initrd corepure64.gz \
  -append 'console=ttyS0 loglevel=7' \
  -device virtio-serial-pci \
  -m 512M \
  -display none \
  -serial stdio \
  -no-reboot 2>&1 | tee virtio_boot_full.log
```

### Option 2: Python Script (Interactive)

```bash
./test_linux_virtio.py
```

---

## Verify Results

```bash
# Find Virtio device
grep "1af4:1003" virtio_boot_full.log

# Output:
# pci 0000:00:04.0: [1af4:1003] type 00 class 0x078000
```

**Interpretation:**
- `1af4` = Red Hat / Virtio vendor ID
- `1003` = Virtio Console device ID
- Device **actually detected** by Linux kernel!

---

## What You're Seeing

✅ **Real Linux kernel** booting in QEMU
✅ **Real PCI enumeration** finding the Virtio device
✅ **Real device detection** - not simulated!

All boot messages in `virtio_boot_full.log` are authentic kernel output.

---

## Files Created

- `virtio_boot_full.log` - Complete boot log
- `TEST_RESULTS.md` - Detailed test report
- `README.md` - Full documentation
- `QUICK_START.md` - This file

---

## Troubleshooting

**"QEMU not found"**
```bash
apt install qemu-system-x86
```

**"vmlinuz-test not found"**
```bash
./setup_linux.sh
```

**"No Virtio detected"**
Check the log manually:
```bash
grep -iE "virtio|1af4" virtio_boot_full.log
```

---

## Understanding PCI Device ID `1af4:1003`

- **Vendor `1af4`**: Red Hat (Virtio specification owner)
- **Device `1003`**: Virtio Console
- **Class `0x078000`**: Serial communication controller (other)

This device ID is **standardized** in the Virtio specification.

---

## Next Experiments

1. **Add more Virtio devices:**
   ```bash
   -device virtio-net-pci
   -device virtio-blk-pci,drive=test
   ```

2. **Monitor device initialization:**
   ```bash
   grep -A5 "1af4:1003" virtio_boot_full.log
   ```

3. **Test different Virtio types:**
   - virtio-serial-pci (console)
   - virtio-net-pci (network)
   - virtio-blk-pci (block storage)
   - virtio-scsi-pci (SCSI)

---

**Ready to test!** Run `./setup_linux.sh` then boot Linux! 🚀
