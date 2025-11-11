#!/bin/bash
# pxOS QEMU Boot Script
# Boots pxOS in QEMU emulator

# Check if binary exists
if [ ! -f "../pxos.bin" ]; then
    echo "Error: pxos.bin not found. Run build_pxos.py first."
    exit 1
fi

echo "Booting pxOS in QEMU..."
echo "Press Ctrl+A then X to exit QEMU"
echo ""

# Boot with minimal memory and serial output
qemu-system-i386 \
    -fda ../pxos.bin \
    -boot a \
    -m 32 \
    -display gtk \
    -monitor stdio
