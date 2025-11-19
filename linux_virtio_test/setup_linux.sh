#!/bin/bash
# Setup script to download TinyCore Linux for Virtio testing

set -e

echo "========================================="
echo "TinyCore Linux Setup for Virtio Testing"
echo "========================================="
echo ""

# TinyCore Linux 64-bit (CorePure64) - minimal Linux distribution
TINYCORE_VERSION="15.x"
MIRROR="http://tinycorelinux.net/15.x/x86_64/release/distribution_files"

echo "📦 Downloading TinyCore Linux components..."
echo ""

# Download kernel (vmlinuz64)
if [ ! -f vmlinuz-test ]; then
    echo "Downloading kernel..."
    wget -O vmlinuz-test "${MIRROR}/vmlinuz64" || {
        echo "❌ Failed to download kernel"
        echo "Trying alternative: vmlinuz from /boot/"
        # Alternative: try to use host kernel if available
        if [ -f /boot/vmlinuz ]; then
            cp /boot/vmlinuz vmlinuz-test
            echo "✅ Using host kernel"
        else
            echo "⚠️  No kernel available. You may need to manually provide vmlinuz-test"
            exit 1
        fi
    }
    echo "✅ Kernel downloaded: vmlinuz-test"
else
    echo "✅ Kernel already exists: vmlinuz-test"
fi

echo ""

# Download initrd (corepure64.gz)
if [ ! -f corepure64.gz ]; then
    echo "Downloading initrd..."
    wget -O corepure64.gz "${MIRROR}/corepure64.gz" || {
        echo "❌ Failed to download initrd"
        echo "⚠️  You may need to manually provide corepure64.gz"
        exit 1
    }
    echo "✅ Initrd downloaded: corepure64.gz"
else
    echo "✅ Initrd already exists: corepure64.gz"
fi

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "Files ready:"
ls -lh vmlinuz-test corepure64.gz 2>/dev/null || echo "Some files missing!"
echo ""
echo "Next step: Run ./test_linux_virtio.py"
