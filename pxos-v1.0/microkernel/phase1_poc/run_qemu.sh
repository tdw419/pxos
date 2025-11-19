#!/bin/bash

cd "$(dirname "$0")"

if [ ! -f build/pxos.iso ]; then
    echo "Error: build/pxos.iso not found. Run ./test_grub_multiboot.sh first."
    exit 1
fi

echo "=========================================="
echo "Launching pxOS in QEMU"
echo "=========================================="
echo
echo "Expected output:"
echo "  - Boot message"
echo "  - Long mode entry (L marker)"
echo "  - PCIe scan (P marker)"
echo "  - GPU found (G marker)"
echo "  - Hello from GPU OS!"
echo
echo "Press Ctrl+C to exit"
echo
echo "=========================================="
echo

qemu-system-x86_64 \
    -cdrom build/pxos.iso \
    -m 512M \
    -device VGA,vgamem_mb=64 \
    -serial stdio
