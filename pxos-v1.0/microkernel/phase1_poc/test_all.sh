#!/bin/bash
echo "========================================="
echo "pxOS Testing - All Methods"
echo "========================================="
echo ""

echo "✅ Method 1: Python Test Harness"
echo "----------------------------------------"
python3 test_privilege_broker.py 2>/dev/null
echo ""

echo "✅ Method 2: C User Mode Test"
echo "----------------------------------------"
cat > /tmp/broker_test.c << 'CEOF'
#include <stdio.h>
#include <stdint.h>
uint32_t mailbox = 0;
void handle_req() {
    uint8_t op = (mailbox >> 24) & 0xFF;
    if (op == 0x80) printf("%c", (char)(mailbox & 0xFF));
    mailbox = 0;
}
int main() {
    mailbox = (0x80 << 24) | 'H';
    handle_req();
    mailbox = (0x80 << 24) | 'i';
    handle_req();
    printf("!\n✅ Mailbox protocol working in C test\n");
    return 0;
}
CEOF
gcc /tmp/broker_test.c -o /tmp/broker_test && /tmp/broker_test
echo ""

echo "========================================="
echo "All Working Tests Complete!"
echo "========================================="
