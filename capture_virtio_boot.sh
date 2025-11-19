#!/bin/bash
# Capture Linux boot with Virtio device to file

timeout 12 qemu-system-x86_64 \
  -kernel vmlinuz-test \
  -initrd corepure64.gz \
  -append 'console=ttyS0' \
  -device virtio-serial-pci \
  -m 512M \
  -display none \
  -serial stdio \
  -no-reboot 2>&1 > virtio_boot_full.log

echo "Boot captured to virtio_boot_full.log"
echo ""
echo "=== VIRTIO DEVICE DETECTION ==="
grep -i "virtio\|1af4" virtio_boot_full.log | head -20
echo ""
echo "=== PCI DEVICES DETECTED ==="
grep "pci 0000:" virtio_boot_full.log | head -10
