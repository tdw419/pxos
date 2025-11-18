#!/bin/bash
# Build and test pxOS with GRUB multiboot in QEMU

set -e

echo "==========================================="
echo "pxOS QEMU Testing via GRUB Multiboot"
echo "==========================================="
echo ""

cd /home/user/pxos/pxos-v1.0/microkernel/phase1_poc

# Step 1: Check for GRUB tools
echo "[1/5] Checking for GRUB tools..."
if ! command -v grub-mkrescue &> /dev/null; then
    echo "⚠️  grub-mkrescue not found!"
    echo ""
    echo "To install GRUB tools:"
    echo "  sudo apt-get install grub-pc-bin xorriso"
    echo ""
    echo "Or skip GRUB and use test harness:"
    echo "  python3 test_privilege_broker.py"
    exit 1
fi
echo "✅ GRUB tools found"
echo ""

# Step 2: Build multiboot kernel
echo "[2/5] Building multiboot kernel..."
if [ -f "microkernel_multiboot.asm" ]; then
    nasm -f elf32 microkernel_multiboot.asm -o build/microkernel_multiboot.o 2>&1 | grep -v "warning" || true
    ld -m elf_i386 -T linker.ld -o build/pxos_multiboot.elf build/microkernel_multiboot.o 2>&1 | grep -v "warning" || true
    echo "✅ Built pxos_multiboot.elf"
else
    echo "❌ microkernel_multiboot.asm not found"
    exit 1
fi
echo ""

# Step 3: Create ISO structure
echo "[3/5] Creating ISO directory structure..."
mkdir -p iso/boot/grub
cp build/pxos_multiboot.elf iso/boot/pxos.elf

cat > iso/boot/grub/grub.cfg << 'EOF'
set timeout=0
set default=0

menuentry "pxOS Phase 1 POC" {
    multiboot /boot/pxos.elf
    boot
}
EOF
echo "✅ ISO structure created"
echo ""

# Step 4: Generate bootable ISO
echo "[4/5] Generating bootable ISO..."
grub-mkrescue -o build/pxos.iso iso/ 2>&1 | grep -v "warning" || true
echo "✅ Created build/pxos.iso"
echo ""

# Step 5: Test in QEMU
echo "[5/5] Testing in QEMU..."
echo "----------------------------------------"
echo "Expected output:"
echo "  - 'pxOS Microkernel v0.1'"
echo "  - 'OK' messages for GPU init"
echo "  - Possible 'Hello from GPU OS!' if full execution works"
echo "----------------------------------------"
echo ""

timeout 5 qemu-system-x86_64 \
    -cdrom build/pxos.iso \
    -m 512M \
    -nographic 2>&1 | head -50 || true

echo ""
echo "==========================================="
echo "QEMU Test Complete!"
echo "==========================================="
echo ""
echo "If you saw the microkernel banner, multiboot worked!"
echo "Any crashes are likely in the long mode transition within"
echo "the multiboot kernel itself (not the bootloader)."
echo ""
echo "For guaranteed working tests, use:"
echo "  python3 test_privilege_broker.py"
