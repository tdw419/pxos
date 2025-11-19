#!/bin/bash
# Build script for pxOS Phase 1 POC

set -e

echo "========================================"
echo "pxOS Phase 1 POC - Build System"
echo "========================================"
echo ""

# Check for required tools
echo "Checking build tools..."
command -v nasm >/dev/null 2>&1 || { echo -e "${RED}ERROR: nasm not found${NC}"; exit 1; }
echo -e "${GREEN}✓ nasm found${NC}"
command -v python3 >/dev/null 2>&1 || { echo -e "${RED}ERROR: python3 not found${NC}"; exit 1; }
echo -e "${GREEN}✓ python3 found${NC}"

# Create output directory
mkdir -p build

# Build bootloader
echo ""
echo -e "${YELLOW}Building bootloader...${NC}"
nasm -f bin boot.asm -o build/boot.bin
if [ $? -eq 0 ]; then
    SIZE=$(stat -c%s build/boot.bin)
    echo -e "${GREEN}✓ Bootloader built: $SIZE bytes${NC}"
else
    echo -e "${RED}✗ Bootloader build failed${NC}"
    exit 1
fi

# Build microkernel
echo ""
echo -e "${YELLOW}Building microkernel...${NC}"
nasm -f bin microkernel.asm -o build/microkernel.bin
if [ $? -eq 0 ]; then
    SIZE=$(stat -c%s build/microkernel.bin)
    echo -e "${GREEN}✓ Microkernel built: $SIZE bytes${NC}"
else
    echo -e "${RED}✗ Microkernel build failed${NC}"
    exit 1
fi

# Generate os.pxi (pixel-encoded OS)
echo ""
echo -e "${YELLOW}Generating os.pxi (pixel-encoded OS)...${NC}"
python3 create_os_pxi.py
if [ $? -eq 0 ]; then
    SIZE=$(stat -c%s os_poc.pxi)
    echo -e "${GREEN}✓ os.pxi generated: $SIZE bytes${NC}"
else
    echo -e "${RED}✗ os.pxi generation failed${NC}"
    exit 1
fi

# Create disk image
echo ""
echo -e "${YELLOW}Creating disk image...${NC}"
dd if=/dev/zero of=build/pxos.img bs=1M count=10 2>/dev/null
dd if=build/boot.bin of=build/pxos.img conv=notrunc 2>/dev/null
dd if=build/microkernel.bin of=build/pxos.img bs=512 seek=1 conv=notrunc 2>/dev/null
dd if=os_poc.pxi of=build/pxos.img bs=512 seek=64 conv=notrunc 2>/dev/null
echo -e "${GREEN}✓ Disk image created: build/pxos.img${NC}"

echo ""
echo "========================================"
echo "Build Summary"
echo "========================================"
echo "Bootloader:    $(stat -c%s build/boot.bin) bytes"
echo "Microkernel:   $(stat -c%s build/microkernel.bin) bytes"
echo "GPU program:   $(stat -c%s os_poc.pxi) bytes"
echo "Disk image:    $(stat -c%s build/pxos.img) bytes"
echo ""
echo -e "${GREEN}Build complete!${NC}"
