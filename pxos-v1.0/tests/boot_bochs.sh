#!/bin/bash
# pxOS Bochs Boot Script
# Boots pxOS in Bochs emulator (if installed)

# Check if binary exists
if [ ! -f "../pxos.bin" ]; then
    echo "Error: pxos.bin not found. Run build_pxos.py first."
    exit 1
fi

# Check if bochs is installed
if ! command -v bochs &> /dev/null; then
    echo "Error: bochs not found. Install with: sudo apt install bochs bochs-x"
    exit 1
fi

# Create temporary bochsrc
cat > /tmp/bochsrc.pxos << EOF
megs: 32
romimage: file=/usr/share/bochs/BIOS-bochs-latest
vgaromimage: file=/usr/share/bochs/VGABIOS-lgpl-latest
floppya: 1_44=../pxos.bin, status=inserted
boot: floppy
log: /tmp/bochsout.txt
panic: action=ask
error: action=report
info: action=report
debug: action=ignore
EOF

echo "Booting pxOS in Bochs..."
bochs -f /tmp/bochsrc.pxos -q
