#!/bin/bash
# Download TinyCore Linux kernel and initrd for testing

echo "📥 Downloading TinyCore Linux (64-bit)..."
echo "=========================================="
echo ""

# Download kernel
if [ ! -f vmlinuz64 ]; then
    echo "Downloading kernel (vmlinuz64)..."
    wget -q --show-progress http://tinycorelinux.net/13.x/x86_64/release/distribution_files/vmlinuz64
    ln -sf vmlinuz64 vmlinuz-test
    echo "✅ Kernel downloaded"
else
    echo "✅ Kernel already exists"
fi

echo ""

# Download initrd
if [ ! -f corepure64.gz ]; then
    echo "Downloading initrd (corepure64.gz)..."
    wget -q --show-progress http://tinycorelinux.net/13.x/x86_64/release/distribution_files/corepure64.gz
    echo "✅ InitRD downloaded"
else
    echo "✅ InitRD already exists"
fi

echo ""
echo "=========================================="
echo "✅ TinyCore Linux ready!"
echo ""
echo "Files:"
ls -lh vmlinuz64 corepure64.gz vmlinuz-test 2>/dev/null
echo ""
echo "Next step: Run ./boot_linux_virtio.sh"
