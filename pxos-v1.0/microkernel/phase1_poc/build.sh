#!/bin/bash
# Build script for pxOS Phase 1 POC

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "pxOS Phase 1 POC - Build System"
echo "========================================"
echo ""

# Check for required tools
echo "Checking build tools..."
command -v nasm >/dev/null 2>&1 || { echo -e "${RED}ERROR: nasm not found${NC}"; exit 1; }
echo -e "${GREEN}✓ nasm found${NC}"

# Create build directory
mkdir -p build

# Build bootloader
echo ""
echo -e "${YELLOW}Building bootloader...${NC}"
nasm -f bin -o build/boot.bin boot.asm
if [ $? -eq 0 ]; then
    SIZE=$(stat -c%s build/boot.bin)
    if [ $SIZE -eq 512 ]; then
        echo -e "${GREEN}✓ Bootloader built: $SIZE bytes${NC}"
    else
        echo -e "${RED}ERROR: Bootloader must be 512 bytes, got $SIZE${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Bootloader build failed${NC}"
    exit 1
fi

# Build microkernel
echo ""
echo -e "${YELLOW}Building microkernel...${NC}"
nasm -f bin -o build/microkernel.bin microkernel.asm
if [ $? -eq 0 ]; then
    SIZE=$(stat -c%s build/microkernel.bin)
    echo -e "${GREEN}✓ Microkernel built: $SIZE bytes${NC}"
    if [ $SIZE -gt 8192 ]; then
        echo -e "${YELLOW}WARNING: Microkernel larger than 8KB ($SIZE bytes)${NC}"
    fi
else
    echo -e "${RED}✗ Microkernel build failed${NC}"
    exit 1
fi

# Create disk image
echo ""
echo -e "${YELLOW}Creating disk image...${NC}"

# Start with bootloader
cat build/boot.bin > build/pxos.img

# Add microkernel at sector 2 (512 bytes offset)
dd if=build/microkernel.bin of=build/pxos.img bs=512 seek=1 conv=notrunc 2>/dev/null

# Pad to 10MB
dd if=/dev/zero of=build/pxos.img bs=1M count=0 seek=10 2>/dev/null

echo -e "${GREEN}✓ Disk image created: build/pxos.img${NC}"

# Print summary
echo ""
echo "========================================"
echo "Build Summary"
echo "========================================"
echo "Bootloader:    $(stat -c%s build/boot.bin) bytes"
echo "Microkernel:   $(stat -c%s build/microkernel.bin) bytes"
echo "Disk image:    $(stat -c%s build/pxos.img) bytes"
echo ""
echo "Total code:    $(( $(stat -c%s build/boot.bin) + $(stat -c%s build/microkernel.bin) )) bytes"
echo ""
echo "========================================"
echo "Test Commands"
echo "========================================"
echo "QEMU (no KVM):"
echo "  qemu-system-x86_64 -drive file=build/pxos.img,format=raw -m 512M"
echo ""
echo "QEMU (with KVM):"
echo "  qemu-system-x86_64 -enable-kvm -drive file=build/pxos.img,format=raw -m 512M"
echo ""
echo -e "${GREEN}Build complete!${NC}"
