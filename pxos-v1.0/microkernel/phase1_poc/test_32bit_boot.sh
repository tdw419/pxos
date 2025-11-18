#!/bin/bash
# Test 32-bit fallback bootloader
# Bootloader: 16→32, Microkernel: 32→64

set -e

echo "==========================================="
echo "pxOS 32-bit Fallback Boot Test"
echo "==========================================="
echo ""

cd /home/user/pxos/pxos-v1.0/microkernel/phase1_poc

# Build bootloader
echo "[1/3] Building 32-bit bootloader..."
nasm -f bin boot_32bit.asm -o build/boot_32bit.bin
echo "✅ Built boot_32bit.bin ($(stat -c%s build/boot_32bit.bin) bytes)"
echo ""

# Build microkernel
echo "[2/3] Building 32-entry microkernel..."
nasm -f bin microkernel_32entry.asm -o build/microkernel_32entry.bin
echo "✅ Built microkernel_32entry.bin ($(stat -c%s build/microkernel_32entry.bin) bytes)"
echo ""

# Create disk image
echo "[3/3] Creating disk image..."
cat build/boot_32bit.bin build/microkernel_32entry.bin > build/pxos_32bit.img
echo "✅ Created build/pxos_32bit.img ($(stat -c%s build/pxos_32bit.img) bytes)"
echo ""

# Test in QEMU
echo "==========================================="
echo "Testing in QEMU..."
echo "==========================================="
echo ""
echo "Expected VGA markers:"
echo "  R A D P 3 K M T 6 G"
echo ""
echo "Expected output:"
echo "  'pxOS CPU Microkernel v0.3'"
echo "  'Scanning PCIe bus 0... OK'"
echo "  'Executing GPU program... OK'"
echo "  'Hello from GPU OS!'"
echo "  'System halted.'"
echo ""
echo "Press Ctrl-A then X to exit QEMU"
echo ""
echo "-------------------------------------------"

qemu-system-x86_64 \
    -drive file=build/pxos_32bit.img,format=raw \
    -m 512M \
    -nographic \
    -serial mon:stdio

echo ""
echo "==========================================="
echo "Test complete!"
echo "==========================================="
