#!/bin/bash
# Test bootloader with visible output

echo "Starting QEMU with bootloader..."
echo "You should see boot messages in the QEMU window"
echo ""

# Run QEMU with minimal display (let it try SDL or GTK)
timeout 8 qemu-system-x86_64 \
    -drive format=raw,file=pxos_boot.img \
    -m 512M \
    -device VGA \
    -display gtk 2>&1 | head -20 || true

echo ""
echo "Test complete"
