#!/usr/bin/env python3
"""
Boot Linux fully and capture all output
"""
import subprocess

print("🚀 COMPLETE LINUX BOOT TEST")
print("=" * 70)
print()

cmd = [
    'qemu-system-x86_64',
    '-kernel', 'vmlinuz-test',
    '-append', 'console=ttyS0 panic=1',  # panic=1 to reboot on kernel panic
    '-serial', 'stdio',
    '-display', 'none',
    '-m', '256M',
    '-no-reboot'
]

print("Starting Linux...")
print()

try:
    result = subprocess.run(
        cmd,
        timeout=30,
        capture_output=True,
        text=True
    )

    # Print output
    print(result.stdout)
    if result.stderr:
        print("\n=== STDERR ===")
        print(result.stderr)

    print(f"\n=== EXIT CODE: {result.returncode} ===")

    # Analyze output
    if "Kernel panic" in result.stdout:
        print("\n❌ KERNEL PANIC - Boot failed")
    elif "login:" in result.stdout.lower():
        print("\n✅ LINUX BOOTED TO LOGIN PROMPT!")
    elif "init" in result.stdout:
        print("\n✅ LINUX INIT STARTED!")
    else:
        print("\n📊 Boot partially successful, check output above")

except subprocess.TimeoutExpired as e:
    print("Captured output:")
    print(e.stdout if e.stdout else "No stdout")
    print("\n⏱️ Timed out - kernel may be waiting for input or hung")
except Exception as e:
    print(f"\n❌ Error: {e}")
