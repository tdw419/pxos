#!/bin/bash
# pxOS Automated Input Test
# Tests basic keyboard input and echo functionality

if ! command -v expect &> /dev/null; then
    echo "Error: 'expect' not found. Install with: sudo apt install expect"
    exit 1
fi

if [ ! -f "../pxos.bin" ]; then
    echo "Error: pxos.bin not found. Run build_pxos.py first."
    exit 1
fi

echo "Running automated input test..."

expect << 'EOF'
set timeout 10

# Start QEMU with serial output
spawn qemu-system-i386 -fda ../pxos.bin -nographic -serial mon:stdio

# Wait for boot
sleep 2

# Send test input
send "hello\r"
sleep 1

# Send more input
send "testing pxOS\r"
sleep 1

# Exit QEMU
send "\x01"
send "x"

expect eof
EOF

echo ""
echo "Test complete!"
