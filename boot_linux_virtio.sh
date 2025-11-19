#!/bin/bash
# Boot Linux with Virtio console device

echo "🚀 Booting Linux with Virtio Console..."
echo "========================================"
echo ""

# Check if we have a Linux kernel
if [ ! -f vmlinuz-test ] && [ ! -f /boot/vmlinuz-$(uname -r) ]; then
    echo "❌ No kernel found. Need to download TinyCore Linux or use system kernel"
    echo ""
    echo "Download TinyCore kernel:"
    echo "  wget http://tinycorelinux.net/13.x/x86_64/release/distribution_files/vmlinuz64"
    echo "  wget http://tinycorelinux.net/13.x/x86_64/release/distribution_files/corepure64.gz"
    exit 1
fi

# Use TinyCore if available, otherwise system kernel
if [ -f vmlinuz-test ]; then
    KERNEL="vmlinuz-test"
    INITRD="corepure64.gz"
elif [ -f vmlinuz64 ]; then
    KERNEL="vmlinuz64"
    INITRD="corepure64.gz"
else
    KERNEL="/boot/vmlinuz-$(uname -r)"
    INITRD="/boot/initrd.img-$(uname -r)"
fi

echo "Kernel: $KERNEL"
echo "InitRD: $INITRD"
echo ""
echo "🔍 Looking for Virtio device detection..."
echo ""

# Boot with Virtio serial device
timeout 10 qemu-system-x86_64 \
  -kernel "$KERNEL" \
  -initrd "$INITRD" \
  -append 'console=ttyS0 quiet' \
  -device virtio-serial-pci \
  -device virtconsole,chardev=console0 \
  -chardev stdio,id=console0,signal=off \
  -m 512M \
  -display none \
  -serial stdio \
  -no-reboot 2>&1 | tee boot_output.log

echo ""
echo "========================================"
echo "📊 Checking for Virtio device detection:"
echo ""

grep -i "virtio\|1af4" boot_output.log | head -10

echo ""
echo "✅ Full boot log saved to: boot_output.log"
