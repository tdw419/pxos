#!/bin/bash
# Build script for pxHV Pixel Driver PoC

set -e

echo "=== Building pxHV Pixel Driver PoC ==="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check for required tools
check_tool() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}ERROR: $1 not found. Please install it.${NC}"
        exit 1
    fi
}

echo "Checking build tools..."
check_tool python3
check_tool nasm
check_tool dd

# Create output directory
mkdir -p build

echo ""
echo -e "${BLUE}=== Stage 1: Compiling Pixel Driver ===${NC}"

# Generate the PXI driver image
../../tools/pxpixel/pxi_system_compiler.py "Hello from GPU!" build/serial_driver.pxi
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Pixel driver compilation failed${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}=== Stage 2: Building Boot Sector ===${NC}"

# Build boot sector
echo -e "${YELLOW}Building pxhv_boot.asm...${NC}"
nasm -f bin -o build/pxhv_boot.bin pxhv_boot.asm
if [ $? -eq 0 ]; then
    SIZE=$(stat -c%s build/pxhv_boot.bin)
    echo -e "${GREEN}✓ Boot sector built: $SIZE bytes (must be 512)${NC}"
else
    echo -e "${RED}✗ Boot sector build failed${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}=== Stage 3: Building Hypervisor ===${NC}"

# Build Stage 2+3 (hypervisor with pixel driver executor)
echo -e "${YELLOW}Building pxhv_stage2_with_stage3.asm...${NC}"
nasm -f bin -o build/pxhv_stage2.bin pxhv_stage2_with_stage3.asm
if [ $? -eq 0 ]; then
    SIZE=$(stat -c%s build/pxhv_stage2.bin)
    echo -e "${GREEN}✓ Hypervisor built: $SIZE bytes${NC}"
else
    echo -e "${RED}✗ Hypervisor build failed${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}=== Stage 4: Creating Disk Image ===${NC}"

# Create disk image
echo -e "${YELLOW}Creating 10MB disk image...${NC}"
dd if=/dev/zero of=build/pxhv.img bs=1M count=10 2>/dev/null
dd if=build/pxhv_boot.bin of=build/pxhv.img conv=notrunc 2>/dev/null
dd if=build/pxhv_stage2.bin of=build/pxhv.img bs=512 seek=1 conv=notrunc 2>/dev/null
echo -e "${GREEN}✓ Disk image created: build/pxhv.img${NC}"

echo ""
echo "=== Build Summary ==="
echo "Pixel Driver:   $(stat -c%s build/serial_driver.pxi) bytes"
echo "Boot sector:    $(stat -c%s build/pxhv_boot.bin) bytes"
echo "Hypervisor:     $(stat -c%s build/pxhv_stage2.bin) bytes"
echo "Disk image:     $(stat -c%s build/pxhv.img) bytes"
echo ""
echo -e "${GREEN}Build complete! Ready to test pixel driver.${NC}"
