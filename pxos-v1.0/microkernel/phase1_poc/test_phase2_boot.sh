#!/bin/bash
# Test Phase 2: GPU Hardware Integration
# Tests BAR mapping, PAT configuration, and PCIe enumeration

set -e

echo "==========================================="
echo "pxOS Phase 2 Boot Test"
echo "==========================================="
echo ""

cd /home/user/pxos/pxos-v1.0/microkernel/phase1_poc

# Build bootloader
echo "[1/3] Building 32-bit bootloader..."
nasm -f bin boot_32bit.asm -o build/boot_32bit.bin
echo "✅ Built boot_32bit.bin ($(stat -c%s build/boot_32bit.bin) bytes)"
echo ""

# Build Phase 2 microkernel
echo "[2/3] Building Phase 2 microkernel..."
nasm -f bin microkernel_phase2.asm -o build/microkernel_phase2.bin
echo "✅ Built microkernel_phase2.bin ($(stat -c%s build/microkernel_phase2.bin) bytes)"
echo ""

# Create disk image
echo "[3/3] Creating disk image..."
cat build/boot_32bit.bin build/microkernel_phase2.bin > build/pxos_phase2.img
echo "✅ Created build/pxos_phase2.img ($(stat -c%s build/pxos_phase2.img) bytes)"
echo ""

# Test in QEMU
echo "==========================================="
echo "Testing in QEMU (Phase 2)..."
echo "==========================================="
echo ""
echo "Expected VGA markers:"
echo "  R A D P 3 K M T 6 G"
echo ""
echo "Expected Phase 2 output:"
echo "  'pxOS CPU Microkernel v0.4 (Phase 2)'"
echo "  'Initializing PAT (cache types)... OK'"
echo "  'Scanning PCIe bus 0... OK'"
echo "  'Mapping GPU BARs (UC/WC)... OK'"
echo "  'Initializing hardware mailbox... OK'"
echo "  'Loading GPU program... OK'"
echo "  'Hello from GPU OS!'"
echo "  'System halted.'"
echo ""
echo "Press Ctrl-A then X to exit QEMU"
echo ""
echo "-------------------------------------------"

# Use bochs-display for better GPU emulation
qemu-system-x86_64 \
    -drive file=build/pxos_phase2.img,format=raw \
    -m 512M \
    -nographic \
    -serial mon:stdio \
    -device bochs-display,vgamem=16M

echo ""
echo "==========================================="
echo "Phase 2 Test complete!"
echo "==========================================="
