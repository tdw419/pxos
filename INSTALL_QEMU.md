# Installing QEMU

QEMU is needed to boot Linux and test Virtio devices.

## Installation by Platform

### **Ubuntu / Debian:**
```bash
sudo apt update
sudo apt install qemu-system-x86
```

### **Fedora / RHEL / CentOS:**
```bash
sudo dnf install qemu-system-x86
```

### **Arch Linux:**
```bash
sudo pacman -S qemu-system-x86
```

### **macOS (with Homebrew):**
```bash
brew install qemu
```

## Verify Installation

```bash
qemu-system-x86_64 --version
```

Should show something like:
```
QEMU emulator version 6.2.0
```

## After Installation

Once QEMU is installed, run:
```bash
./boot_linux_virtio.sh
```

This will boot Linux and show Virtio device detection!
