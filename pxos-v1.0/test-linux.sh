#!/bin/bash
# test-linux.sh - Test pxOS Linux boot in QEMU
# Usage: ./test-linux.sh [options]

set -e

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Default values
DISK_IMAGE="build/pxos-linux.img"
MEMORY="512M"
SERIAL=1
NOGRAPHIC=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--memory)
            MEMORY="$2"
            shift 2
            ;;
        -g|--graphical)
            NOGRAPHIC=0
            shift
            ;;
        -n|--nographic)
            NOGRAPHIC=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -m, --memory SIZE    Set RAM size (default: 512M)"
            echo "  -g, --graphical      Use graphical display"
            echo "  -n, --nographic      Use serial console (default)"
            echo "  -h, --help           Show this help"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if disk image exists
if [ ! -f "$DISK_IMAGE" ]; then
    echo -e "${RED}ERROR: Disk image not found: $DISK_IMAGE${NC}"
    echo -e "${YELLOW}Run ./build-linux.sh first${NC}"
    exit 1
fi

# Check if QEMU is installed
if ! command -v qemu-system-x86_64 &> /dev/null; then
    echo -e "${RED}ERROR: qemu-system-x86_64 not found${NC}"
    echo -e "${YELLOW}Install with: sudo apt install qemu-system-x86${NC}"
    exit 1
fi

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     pxOS Linux - QEMU Test Boot       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Disk image:${NC} $DISK_IMAGE"
echo -e "${YELLOW}Memory:${NC} $MEMORY"
echo -e "${YELLOW}Mode:${NC} $([ $NOGRAPHIC -eq 1 ] && echo 'Serial console' || echo 'Graphical')"
echo ""
echo -e "${GREEN}Starting QEMU...${NC}"
echo -e "${YELLOW}(Press Ctrl+A then X to exit QEMU)${NC}"
echo ""
sleep 1

# Build QEMU command
QEMU_CMD="qemu-system-x86_64"
QEMU_CMD="$QEMU_CMD -drive file=$DISK_IMAGE,format=raw"
QEMU_CMD="$QEMU_CMD -m $MEMORY"

if [ $NOGRAPHIC -eq 1 ]; then
    QEMU_CMD="$QEMU_CMD -nographic -serial stdio"
else
    QEMU_CMD="$QEMU_CMD -serial stdio"
fi

# Additional useful options
QEMU_CMD="$QEMU_CMD -no-reboot"  # Don't reboot on kernel panic

# Run QEMU
eval $QEMU_CMD

echo ""
echo -e "${GREEN}QEMU exited${NC}"
