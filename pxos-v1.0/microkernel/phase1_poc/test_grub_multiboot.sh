#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "pxOS Phase 2 - GRUB Multiboot Build"
echo "=========================================="
echo

echo "[1/5] Compiling kernel..."
nasm -f elf64 microkernel_multiboot.asm -o build/microkernel.o

echo "[2/5] Compiling BAR0 mapper..."
nasm -f elf64 map_gpu_bar0.asm -o build/map_gpu_bar0.o

echo "[3/5] Linking kernel..."
ld -m elf_x86_64 -T linker.ld build/microkernel.o build/map_gpu_bar0.o -o build/microkernel.bin

echo "[4/5] Preparing ISO structure..."
cp build/microkernel.bin iso/boot/

echo "[5/5] Creating bootable ISO..."
grub-mkrescue -o build/pxos.iso iso/ 2>&1 | grep -v "warning:" || true

echo "[6/6] ISO created: build/pxos.iso"
echo
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo
echo "To test in QEMU, run:"
echo "  qemu-system-x86_64 -cdrom build/pxos.iso -m 512M"
echo
echo "Or run:"
echo "  ./run_qemu.sh"
echo
