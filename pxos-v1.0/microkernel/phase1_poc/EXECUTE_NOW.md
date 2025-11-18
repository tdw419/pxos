# IMMEDIATE EXECUTION PLAN

**Time: 5 minutes**
**Goal: Boot pxOS in QEMU via GRUB**

## Step 1: Install GRUB (1 minute)
```bash
sudo apt-get update && sudo apt-get install -y grub-pc-bin xorriso
```

## Step 2: Build GRUB ISO (2 minutes)
```bash
cd /home/user/pxos/pxos-v1.0/microkernel/phase1_poc
./test_grub_multiboot.sh
```

## Step 3: Boot in QEMU (30 seconds)
```bash
qemu-system-x86_64 -cdrom build/pxos.iso -m 512M -nographic
```

## Expected Output
```
pxOS Microkernel v0.3
Scanning PCIe bus 0...
Device 00:02.0: VGA compatible
BAR0: 0xE0000000
Hello from GPU OS!
```

## If This Fails
Fallback to simplified 32-bit bootloader (see BOOTLOADER_32BIT_FALLBACK.md)

---
**Status: READY TO EXECUTE**
**Next: Run the commands above**
