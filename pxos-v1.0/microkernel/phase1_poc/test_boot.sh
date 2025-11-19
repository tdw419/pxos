#!/bin/bash
echo "Booting pxOS Phase 2 in QEMU..."
echo "Serial output should appear below:"
echo "========================================"
timeout 5 qemu-system-x86_64 \
  -cdrom build/pxos.iso \
  -m 512M \
  -serial file:serial_output.log \
  -display none \
  -d cpu_reset,int 2>&1

echo "========================================"
echo "Serial output log:"
if [ -f serial_output.log ]; then
  cat serial_output.log
  echo ""
  echo "Bytes written: $(wc -c < serial_output.log)"
else
  echo "No serial output file created"
fi
