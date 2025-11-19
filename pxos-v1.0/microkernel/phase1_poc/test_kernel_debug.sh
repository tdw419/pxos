#!/bin/bash
# Debug test for pxOS kernel with fixed BAR0 mapping

echo "========================================"
echo "pxOS Kernel Debug Test"
echo "========================================"
echo ""

# Run QEMU with extensive debugging
timeout 10 qemu-system-x86_64 \
    -cdrom build/pxos.iso \
    -m 512M \
    -serial file:qemu_serial.log \
    -d int,cpu_reset,guest_errors \
    -D qemu_debug.log \
    -no-reboot \
    -no-shutdown \
    2>&1 | head -100

echo ""
echo "========================================"
echo "Serial Output:"
echo "========================================"
if [ -f qemu_serial.log ]; then
    cat qemu_serial.log
    echo ""
else
    echo "(No serial output captured)"
fi

echo ""
echo "========================================"
echo "Debug Log (last 50 lines):"
echo "========================================"
if [ -f qemu_debug.log ]; then
    tail -50 qemu_debug.log
else
    echo "(No debug log)"
fi

echo ""
echo "========================================"
echo "Test Complete"
echo "========================================"
