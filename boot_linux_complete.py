#!/usr/bin/env python3
"""
COMPLETE LINUX BOOT - With kernel AND rootfs
"""
import subprocess
import sys

print("🚀 BOOTING COMPLETE LINUX SYSTEM")
print("=" * 70)
print("Kernel: vmlinuz-test (TinyCore 6.1.2)")
print("Rootfs: corepure64.gz (TinyCore x86_64)")
print("=" * 70)
print()

cmd = [
    'qemu-system-x86_64',
    '-kernel', 'vmlinuz-test',
    '-initrd', 'corepure64.gz',
    '-append', 'console=ttyS0',
    '-serial', 'stdio',
    '-display', 'none',
    '-m', '512M',
    '-device', 'VGA,vgamem_mb=64'  # Our GPU device!
]

print("Command:", ' '.join(cmd))
print()
print("⏳ Booting Linux (this may take 10-15 seconds)...")
print("=" * 70)
print()

try:
    # Run without timeout to let it fully boot
    subprocess.run(cmd, timeout=60)
except subprocess.TimeoutExpired:
    print("\n⏱️ Timed out after 60s")
except KeyboardInterrupt:
    print("\n🛑 Interrupted")
