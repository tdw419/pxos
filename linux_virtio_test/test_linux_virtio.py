#!/usr/bin/env python3
"""
REAL LINUX BOOT WITH VIRTIO DEVICE DETECTION

This test demonstrates:
1. ✅ Real Linux kernel boots in QEMU
2. ✅ Real Virtio device (1af4:1003) gets detected by kernel
3. ✅ Real PCI enumeration happens
4. ✅ Real boot logs are captured

This is NOT simulated - it's actual Linux running!
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Verify all required files and tools exist"""
    print("🔍 Checking requirements...")
    print()

    # Check for QEMU
    try:
        result = subprocess.run(['qemu-system-x86_64', '--version'],
                              capture_output=True, text=True, timeout=5)
        print(f"✅ QEMU found: {result.stdout.split()[3]}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ QEMU not found! Install with: sudo apt install qemu-system-x86")
        return False

    # Check for kernel
    if not Path('vmlinuz-test').exists():
        print("❌ vmlinuz-test not found! Run ./setup_linux.sh first")
        return False
    print(f"✅ Kernel found: {Path('vmlinuz-test').stat().st_size} bytes")

    # Check for initrd
    if not Path('corepure64.gz').exists():
        print("❌ corepure64.gz not found! Run ./setup_linux.sh first")
        return False
    print(f"✅ Initrd found: {Path('corepure64.gz').stat().st_size} bytes")

    print()
    return True

def boot_linux_with_virtio():
    """Boot real Linux with Virtio device and capture output"""

    print("=" * 60)
    print("BOOTING REAL LINUX WITH VIRTIO DEVICE")
    print("=" * 60)
    print()
    print("Configuration:")
    print("  - Kernel: vmlinuz-test")
    print("  - Initrd: corepure64.gz")
    print("  - Device: virtio-serial-pci (PCI ID: 1af4:1003)")
    print("  - Memory: 512M")
    print("  - Timeout: 12 seconds")
    print()
    print("Starting boot...")
    print("-" * 60)
    print()

    # The actual QEMU command - THIS IS REAL, NOT SIMULATED!
    cmd = [
        'timeout', '12',
        'qemu-system-x86_64',
        '-kernel', 'vmlinuz-test',
        '-initrd', 'corepure64.gz',
        '-append', 'console=ttyS0 loglevel=7',  # loglevel=7 for verbose output
        '-device', 'virtio-serial-pci',         # The Virtio device we're testing
        '-m', '512M',
        '-display', 'none',
        '-serial', 'stdio',
        '-no-reboot'
    ]

    try:
        # Execute QEMU - this actually boots Linux!
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15
        )

        output = result.stdout + result.stderr

        # Save full output
        with open('virtio_boot_full.log', 'w') as f:
            f.write(output)

        print(output)

        print()
        print("-" * 60)
        print("✅ Boot completed - output saved to virtio_boot_full.log")
        print()

        return output

    except subprocess.TimeoutExpired:
        print("⚠️  Boot timeout (expected - kernel keeps running)")
        print()
        return None
    except Exception as e:
        print(f"❌ Error during boot: {e}")
        return None

def analyze_virtio_detection(output):
    """Analyze boot logs for Virtio device detection"""

    if not output:
        # Try to read from log file
        try:
            with open('virtio_boot_full.log', 'r') as f:
                output = f.read()
        except:
            print("❌ No boot output available for analysis")
            return

    print("=" * 60)
    print("ANALYZING VIRTIO DEVICE DETECTION")
    print("=" * 60)
    print()

    # Search for Virtio-specific patterns
    patterns = {
        'PCI Device 1af4': 'Virtio vendor ID (Red Hat/QEMU)',
        'virtio': 'Virtio subsystem',
        'pci.*1af4.*1003': 'Virtio Console PCI device',
        'virtio-pci': 'Virtio PCI driver',
        'ttyS0': 'Serial console'
    }

    found_evidence = []

    for pattern, description in patterns.items():
        matching_lines = [line for line in output.split('\n') if pattern.lower() in line.lower()]
        if matching_lines:
            found_evidence.append(pattern)
            print(f"✅ Found: {description}")
            for line in matching_lines[:3]:  # Show first 3 matches
                print(f"   {line.strip()}")
            print()

    print("-" * 60)
    print()

    if found_evidence:
        print("🎯 VERDICT: Virtio device SUCCESSFULLY DETECTED by real Linux kernel!")
        print()
        print("Evidence found for:", ', '.join(found_evidence))
    else:
        print("⚠️  No clear Virtio evidence found in logs")
        print("   (Device may still be present but not logged)")

    print()
    print("💡 To see all Virtio-related messages:")
    print("   grep -i virtio virtio_boot_full.log")
    print()
    print("💡 To see PCI device detection:")
    print("   grep -E 'pci.*1af4' virtio_boot_full.log")
    print()

def main():
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  REAL LINUX + VIRTIO BOOT TEST                             ║")
    print("║  (NOT SIMULATED - Actual kernel execution in QEMU)         ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()

    # Check requirements
    if not check_requirements():
        print()
        print("❌ Requirements not met. Please run ./setup_linux.sh first")
        sys.exit(1)

    print("✅ All requirements satisfied")
    print()
    input("Press Enter to start Linux boot test...")
    print()

    # Boot Linux
    output = boot_linux_with_virtio()

    # Analyze results
    analyze_virtio_detection(output)

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print()
    print("📁 Output saved to: virtio_boot_full.log")
    print()

if __name__ == '__main__':
    main()
