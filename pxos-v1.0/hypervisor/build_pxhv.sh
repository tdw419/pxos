#!/bin/bash
# Build script for pxHV (Pixel Hypervisor)

set -e

echo "=== Building pxHV Hypervisor ==="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for required tools
check_tool() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}ERROR: $1 not found. Please install it.${NC}"
        exit 1
    fi
}

echo "Checking build tools..."
check_tool nasm
check_tool dd

# Create output directory
mkdir -p build

# Build boot sector (Stage 1)
echo -e "${YELLOW}Building boot sector...${NC}"
nasm -f bin -o build/pxhv_boot.bin pxhv_boot.asm
if [ $? -eq 0 ]; then
    SIZE=$(stat -c%s build/pxhv_boot.bin)
    echo -e "${GREEN}✓ Boot sector built: $SIZE bytes (must be 512)${NC}"
    if [ $SIZE -ne 512 ]; then
        echo -e "${RED}ERROR: Boot sector must be exactly 512 bytes!${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Boot sector build failed${NC}"
    exit 1
fi

# Build guest code (real mode test)
echo -e "${YELLOW}Building guest code...${NC}"
nasm -f bin -o build/guest_real.bin guest_real.asm
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ guest_real.bin built: $(stat -c%s build/guest_real.bin) bytes${NC}"
else
    echo -e "${RED}✗ guest_real.bin build failed${NC}"
    exit 1
fi

# Build minimal DOS
nasm -f bin -o build/minimal_dos.bin minimal_dos.asm
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ minimal_dos.bin built: $(stat -c%s build/minimal_dos.bin) bytes${NC}"
else
    echo -e "${RED}✗ minimal_dos.bin build failed${NC}"
    exit 1
fi

# Build minimal DOS BIOS version
nasm -f bin -o build/minimal_dos_bios.bin minimal_dos_bios.asm
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ minimal_dos_bios.bin built: $(stat -c%s build/minimal_dos_bios.bin) bytes${NC}"
else
    echo -e "${RED}✗ minimal_dos_bios.bin build failed${NC}"
    exit 1
fi

# Build test bootloader
echo -e "${YELLOW}Building test bootloader...${NC}"
nasm -f bin -o build/test_boot.bin test_boot.asm
if [ $? -eq 0 ]; then
    SIZE=$(stat -c%s build/test_boot.bin)
    echo -e "${GREEN}✓ test_boot.bin built: $SIZE bytes${NC}"
    if [ $SIZE -ne 512 ]; then
        echo -e "${RED}ERROR: Boot sector must be exactly 512 bytes!${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ test_boot.bin build failed${NC}"
    exit 1
fi

# Build Stage 2 (Hypervisor loader)
echo -e "${YELLOW}Building Stage 2 hypervisor loader...${NC}"
nasm -f bin -o build/pxhv_stage2.bin pxhv_stage2.asm
if [ $? -eq 0 ]; then
    SIZE=$(stat -c%s build/pxhv_stage2.bin)
    echo -e "${GREEN}✓ Stage 2 built: $SIZE bytes${NC}"
else
    echo -e "${RED}✗ Stage 2 build failed${NC}"
    exit 1
fi

# Create disk image
echo -e "${YELLOW}Creating disk image...${NC}"
dd if=/dev/zero of=build/pxhv.img bs=1M count=10 2>/dev/null

# Write boot sector
dd if=build/pxhv_boot.bin of=build/pxhv.img conv=notrunc 2>/dev/null

# Write Stage 2 at sector 2 (offset 512 bytes)
dd if=build/pxhv_stage2.bin of=build/pxhv.img bs=512 seek=1 conv=notrunc 2>/dev/null

# Create virtual disk image for guest (at offset 0x100000 = 1MB)
echo -e "${YELLOW}Creating virtual disk for guest...${NC}"
# Write test bootloader as sector 0 of virtual disk
dd if=build/test_boot.bin of=build/pxhv.img bs=512 seek=2048 conv=notrunc 2>/dev/null
# (seek=2048 because 1MB / 512 bytes = 2048 sectors)

# Create a dummy sector 1 for the disk (in case INT 13h tries to read it)
dd if=/dev/zero of=build/pxhv.img bs=512 count=1 seek=2049 conv=notrunc 2>/dev/null

echo -e "${GREEN}✓ Disk image created: build/pxhv.img${NC}"
echo -e "${GREEN}✓ Virtual disk at 1MB with test bootloader${NC}"

# Print summary
echo ""
echo "=== Build Summary ==="
echo "Boot sector:    $(stat -c%s build/pxhv_boot.bin) bytes"
echo "Stage 2:        $(stat -c%s build/pxhv_stage2.bin) bytes"
echo "Disk image:     $(stat -c%s build/pxhv.img) bytes"
echo ""
echo "=== Test Commands ==="
echo "Run in QEMU:    qemu-system-x86_64 -drive file=build/pxhv.img,format=raw -m 512M"
echo "Run with KVM:   qemu-system-x86_64 -enable-kvm -drive file=build/pxhv.img,format=raw -m 512M"
echo "Debug mode:     qemu-system-x86_64 -drive file=build/pxhv.img,format=raw -m 512M -d int,cpu_reset"
echo "Write to USB:   sudo dd if=build/pxhv.img of=/dev/sdX bs=1M status=progress"
echo ""
echo -e "${GREEN}Build complete!${NC}"
