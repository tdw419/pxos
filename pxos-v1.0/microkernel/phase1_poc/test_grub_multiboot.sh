#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"
ISO_DIR="$ROOT_DIR/iso"

KERNEL_SRC="$ROOT_DIR/microkernel_multiboot.asm"
KERNEL_OBJ="$BUILD_DIR/microkernel_multiboot.o"
KERNEL_ELF="$BUILD_DIR/pxos.elf"
ISO_FILE="$BUILD_DIR/pxos.iso"

echo "=== pxOS GRUB Multiboot Test ==="
echo "Root: $ROOT_DIR"
echo

mkdir -p "$BUILD_DIR" "$ISO_DIR/boot/grub"

echo "[1/4] Assembling kernel..."
nasm -f elf32 "$KERNEL_SRC" -o "$KERNEL_OBJ"

echo "[2/4] Linking kernel ELF..."
ld -m elf_i386 -T "$ROOT_DIR/linker.ld" -o "$KERNEL_ELF" "$KERNEL_OBJ"

echo "[3/4] Preparing GRUB ISO tree..."
cp "$KERNEL_ELF" "$ISO_DIR/boot/pxos.elf"

cat > "$ISO_DIR/boot/grub/grub.cfg" << 'EOF'
set timeout=0
set default=0

menuentry "pxOS (Multiboot2)" {
    multiboot2 /boot/pxos.elf
    boot
}
EOF

echo "[4/4] Building ISO with grub-mkrescue..."
grub-mkrescue -o "$ISO_FILE" "$ISO_DIR"

echo
echo "ISO created at: $ISO_FILE"
echo
echo "Run in QEMU with:"
echo "  qemu-system-x86_64 -cdrom \"$ISO_FILE\" -m 512M"
echo

# Optional: auto-run if qemu is installed
if command -v qemu-system-x86_64 >/dev/null 2>&1; then
  echo "Launching QEMU now..."
  qemu-system-x86_64 -cdrom "$ISO_FILE" -m 512M
fi
