# Real Linux + Virtio Boot Test

This test demonstrates **actual Linux kernel boot** with **real Virtio device detection** - not a simulation!

## What This Test Does

✅ **Real**: Boots actual Linux kernel in QEMU
✅ **Real**: Linux PCI subsystem enumerates devices
✅ **Real**: Virtio device (PCI ID `1af4:1003`) is detected
✅ **Real**: All boot logs come from actual kernel

❌ **NOT**: Simulated or fake output
❌ **NOT**: Mock device detection

---

## Quick Start

### 1. Setup (Download Linux)

```bash
cd linux_virtio_test
chmod +x setup_linux.sh
./setup_linux.sh
```

This downloads:
- `vmlinuz-test` - TinyCore Linux kernel (~6MB)
- `corepure64.gz` - Minimal initrd (~12MB)

### 2. Run Test

```bash
chmod +x test_linux_virtio.py
./test_linux_virtio.py
```

### 3. Analyze Results

The test will:
1. Boot Linux in QEMU (12 second timeout)
2. Capture all boot messages to `virtio_boot_full.log`
3. Search for Virtio device detection evidence
4. Display findings

---

## Expected Output

You should see:

### Boot Messages
```
[    0.000000] Linux version 6.x.x ...
[    0.000000] Command line: console=ttyS0 loglevel=7
[    0.123456] PCI: Probing PCI hardware
[    0.234567] pci 0000:00:04.0: [1af4:1003] type 00 class 0x078000
```

### Virtio Detection
```
✅ Found: Virtio vendor ID (Red Hat/QEMU)
   pci 0000:00:04.0: [1af4:1003] type 00 class 0x078000

✅ Found: Virtio subsystem
   virtio-pci 0000:00:04.0: enabling device (0000 -> 0003)
```

---

## What Proves This is Real?

1. **PCI Device ID `1af4:1003`**
   - `1af4` = Red Hat (Virtio vendor)
   - `1003` = Virtio Console device
   - Detected by Linux PCI subsystem during real hardware enumeration

2. **Kernel Boot Messages**
   - Real kernel version strings
   - Authentic CPU detection (Spectre mitigations, etc.)
   - Genuine PCI bus scanning
   - Actual driver initialization

3. **QEMU Process**
   - Real `qemu-system-x86_64` process runs
   - Actual virtualization (KVM if available)
   - Real device emulation

---

## Verification Commands

After running the test:

```bash
# Find all Virtio references
grep -i virtio virtio_boot_full.log

# Find PCI device detection
grep -E "pci.*1af4" virtio_boot_full.log

# Show device type
grep "1af4:1003" virtio_boot_full.log

# Count kernel messages
wc -l virtio_boot_full.log
```

---

## Architecture

```
┌─────────────────────────────────────────┐
│ test_linux_virtio.py (Host)             │
│  └─ subprocess.run()                    │
│      └─ qemu-system-x86_64              │
│          ├─ Loads vmlinuz-test          │
│          ├─ Loads corepure64.gz         │
│          ├─ Creates virtio-serial-pci   │
│          └─ Boots Linux kernel          │
│              └─ PCI enumeration         │
│                  └─ Finds 1af4:1003 ✅  │
└─────────────────────────────────────────┘
```

---

## Files

- `setup_linux.sh` - Downloads TinyCore Linux components
- `test_linux_virtio.py` - Runs the boot test
- `vmlinuz-test` - Linux kernel (downloaded)
- `corepure64.gz` - Initrd (downloaded)
- `virtio_boot_full.log` - Boot output (generated)

---

## Requirements

- Python 3.6+
- QEMU (`sudo apt install qemu-system-x86`)
- Internet connection (for initial setup)
- ~20MB disk space

---

## Troubleshooting

### "QEMU not found"
```bash
sudo apt update
sudo apt install qemu-system-x86
```

### "vmlinuz-test not found"
```bash
./setup_linux.sh
```

### "No Virtio detected"
Check the log file manually:
```bash
cat virtio_boot_full.log | grep -i virtio
```

---

## Why This Matters

This test proves:

1. **Linux actually boots** - Not just print statements
2. **Virtio device is real** - Detected by actual PCI enumeration
3. **Output is authentic** - Direct from Linux kernel via serial console

This is the foundation for testing Virtio drivers, device models, and kernel modifications.

---

## Next Steps

Once you confirm Virtio detection works:

1. **Test custom Virtio drivers** - Load modules during boot
2. **Modify device model** - Change QEMU device parameters
3. **Capture traffic** - Monitor Virtio ring operations
4. **Performance testing** - Measure throughput

---

## License

This test setup is provided as-is for educational purposes.

TinyCore Linux is distributed under GPL - see http://tinycorelinux.net/

---

**This is REAL Linux running on REAL virtual hardware - enjoy! 🐧**
