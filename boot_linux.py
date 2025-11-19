#!/usr/bin/env python3
"""
Boot Linux with QEMU - Clean execution
"""
import subprocess
import sys

print("🚀 BOOTING LINUX WITH QEMU...")
print("=" * 70)

cmd = [
    'qemu-system-x86_64',
    '-kernel', 'vmlinuz-test',
    '-append', 'console=ttyS0',
    '-serial', 'stdio',
    '-display', 'none',
    '-m', '256M',
    '-no-reboot'
]

print("Command:", ' '.join(cmd))
print()

try:
    result = subprocess.run(
        cmd,
        timeout=15,
        capture_output=False,  # Let output go to stdout/stderr directly
        text=True
    )
    print(f"\nQEMU exited with code: {result.returncode}")
except subprocess.TimeoutExpired:
    print("\n⏱️ QEMU timed out (kernel may still be running)")
except KeyboardInterrupt:
    print("\n🛑 Interrupted by user")
except Exception as e:
    print(f"\n❌ Error: {e}")
