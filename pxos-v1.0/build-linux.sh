#!/bin/bash
# build-linux.sh - Build bootable Linux image with pxOS bootloader
# Usage: ./build-linux.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  pxOS Linux Bootloader Builder v1.0   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Check for required tools
echo -e "${YELLOW}[1/6]${NC} Checking dependencies..."

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: python3 not found${NC}"
    exit 1
fi

if ! command -v qemu-system-x86_64 &> /dev/null; then
    echo -e "${YELLOW}WARNING: qemu-system-x86_64 not found. You won't be able to test.${NC}"
    echo -e "${YELLOW}Install with: sudo apt install qemu-system-x86${NC}"
fi

echo -e "${GREEN}✓ Dependencies OK${NC}"

# Create build directory
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"

# Step 1: Build bootloader using Python
echo -e "\n${YELLOW}[2/6]${NC} Building pxOS bootloader..."
python3 build_pxos_bootloader.py

if [ ! -f "$BUILD_DIR/pxos-linux.bin" ]; then
    echo -e "${RED}ERROR: Bootloader build failed${NC}"
    exit 1
fi

# Check bootloader size
BOOT_SIZE=$(stat -c%s "$BUILD_DIR/pxos-linux.bin")
if [ "$BOOT_SIZE" -ne 512 ]; then
    echo -e "${RED}ERROR: Bootloader must be exactly 512 bytes (got $BOOT_SIZE)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Bootloader assembled (512 bytes)${NC}"

# Step 2: Get or build Linux kernel
echo -e "\n${YELLOW}[3/6]${NC} Obtaining Linux kernel..."

KERNEL_FILE="$BUILD_DIR/vmlinuz"

if [ -f "$KERNEL_FILE" ]; then
    echo -e "${GREEN}✓ Using existing kernel: $KERNEL_FILE${NC}"
else
    echo -e "${BLUE}Downloading Tiny Core Linux kernel...${NC}"

    # Download a small Linux kernel
    KERNEL_URL="http://tinycorelinux.net/13.x/x86_64/release/distribution_files/vmlinuz64"

    if command -v wget &> /dev/null; then
        wget -q --show-progress "$KERNEL_URL" -O "$KERNEL_FILE"
    elif command -v curl &> /dev/null; then
        curl -# -L "$KERNEL_URL" -o "$KERNEL_FILE"
    else
        echo -e "${RED}ERROR: Neither wget nor curl found${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Kernel downloaded${NC}"
fi

KERNEL_SIZE=$(stat -c%s "$KERNEL_FILE")
echo -e "${BLUE}  Kernel size: $(($KERNEL_SIZE / 1024)) KB${NC}"

# Step 3: Create minimal initrd
echo -e "\n${YELLOW}[4/6]${NC} Creating initrd..."

INITRD_DIR="$BUILD_DIR/initrd"
rm -rf "$INITRD_DIR"
mkdir -p "$INITRD_DIR"/{bin,dev,proc,sys,etc}

# Create init script
cat > "$INITRD_DIR/init" << 'EOF'
#!/bin/sh

echo ""
echo "========================================="
echo "  Welcome to pxOS Linux!"
echo "========================================="
echo ""
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"
echo ""
echo "Mounting filesystems..."
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev

echo ""
echo "System ready!"
echo ""

# Drop to shell
exec /bin/sh
EOF

chmod +x "$INITRD_DIR/init"

# Try to copy busybox if available
if [ -f /bin/busybox ]; then
    cp /bin/busybox "$INITRD_DIR/bin/"
elif [ -f /usr/bin/busybox ]; then
    cp /usr/bin/busybox "$INITRD_DIR/bin/"
else
    echo -e "${YELLOW}  WARNING: busybox not found, initrd will be minimal${NC}"
    # Create a minimal sh script as fallback
    cat > "$INITRD_DIR/bin/sh" << 'EOF'
#!/bin/sh
echo "Minimal shell (no busybox found)"
while true; do
    echo -n "$ "
    read cmd
    case $cmd in
        "help") echo "Available: help, exit, uname" ;;
        "exit") exit 0 ;;
        "uname") uname -a ;;
        *) echo "Unknown command: $cmd" ;;
    esac
done
EOF
    chmod +x "$INITRD_DIR/bin/sh"
fi

# Create initrd image
(cd "$INITRD_DIR" && find . | cpio -o -H newc 2>/dev/null | gzip -9) > "$BUILD_DIR/initrd.gz"

INITRD_SIZE=$(stat -c%s "$BUILD_DIR/initrd.gz")
echo -e "${GREEN}✓ Initrd created ($(($INITRD_SIZE / 1024)) KB)${NC}"

# Step 4: Create disk image
echo -e "\n${YELLOW}[5/6]${NC} Creating disk image..."

DISK_IMAGE="$BUILD_DIR/pxos-linux.img"
DISK_SIZE_MB=64

# Create empty disk image
dd if=/dev/zero of="$DISK_IMAGE" bs=1M count=$DISK_SIZE_MB status=none

echo -e "${GREEN}✓ Created ${DISK_SIZE_MB}MB disk image${NC}"

# Step 5: Install components
echo -e "\n${YELLOW}[6/6]${NC} Installing components to disk..."

# Install bootloader at sector 0
dd if="$BUILD_DIR/pxos-linux.bin" of="$DISK_IMAGE" conv=notrunc status=none
echo -e "${GREEN}✓ Bootloader installed (sector 0)${NC}"

# Install kernel starting at sector 1 (after boot sector)
dd if="$KERNEL_FILE" of="$DISK_IMAGE" bs=512 seek=1 conv=notrunc status=none
echo -e "${GREEN}✓ Kernel installed (starting sector 1)${NC}"

# Install initrd at sector 10000 (arbitrary location after kernel)
dd if="$BUILD_DIR/initrd.gz" of="$DISK_IMAGE" bs=512 seek=10000 conv=notrunc status=none
echo -e "${GREEN}✓ Initrd installed (sector 10000)${NC}"

# Build complete
echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Build Complete!                ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Disk image:${NC} $DISK_IMAGE"
echo -e "${BLUE}Size:${NC} ${DISK_SIZE_MB}MB"
echo ""
echo -e "${YELLOW}To test in QEMU:${NC}"
echo -e "  ${GREEN}qemu-system-x86_64 -drive file=$DISK_IMAGE,format=raw -m 512M -serial stdio${NC}"
echo ""
echo -e "${YELLOW}To boot on real hardware:${NC}"
echo -e "  ${RED}WARNING: This will overwrite the target device!${NC}"
echo -e "  ${GREEN}sudo dd if=$DISK_IMAGE of=/dev/sdX bs=4M${NC}"
echo ""
