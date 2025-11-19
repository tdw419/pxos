#!/bin/bash
# Setup script for pxOS GPU Hypervisor on host system

set -e

echo "================================================================"
echo "pxOS GPU Hypervisor - Host Setup"
echo "================================================================"
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (or with sudo)"
    exit 1
fi

# Detect installation directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/opt/pxos/pxos-heterogeneous"

echo "Installation directory: $INSTALL_DIR"
echo

# Step 1: Install dependencies
echo "Step 1: Installing dependencies..."
echo

if command -v apt-get &> /dev/null; then
    # Debian/Ubuntu
    apt-get update
    apt-get install -y \
        python3 \
        python3-pip \
        qemu-system-x86 \
        qemu-kvm \
        libvirt-daemon-system \
        libvirt-clients \
        bridge-utils
elif command -v yum &> /dev/null; then
    # RHEL/CentOS
    yum install -y \
        python3 \
        python3-pip \
        qemu-kvm \
        libvirt \
        libvirt-client
elif command -v dnf &> /dev/null; then
    # Fedora
    dnf install -y \
        python3 \
        python3-pip \
        qemu-kvm \
        libvirt
else
    echo "WARNING: Could not detect package manager"
    echo "Please install manually:"
    echo "  - Python 3"
    echo "  - QEMU/KVM"
    echo "  - libvirt"
fi

echo "✓ Dependencies installed"
echo

# Step 2: Copy files to installation directory
echo "Step 2: Installing hypervisor files..."
echo

mkdir -p "$INSTALL_DIR"
cp "$SCRIPT_DIR"/*.py "$INSTALL_DIR/"
cp "$SCRIPT_DIR"/*.sh "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR"/*.sh

echo "✓ Files installed to $INSTALL_DIR"
echo

# Step 3: Install systemd service
echo "Step 3: Installing systemd service..."
echo

cp "$SCRIPT_DIR/pxos-gpu-hypervisor.service" /etc/systemd/system/
systemctl daemon-reload

echo "✓ Systemd service installed"
echo

# Step 4: Create runtime directory
echo "Step 4: Creating runtime directories..."
echo

mkdir -p /var/run/pxos
mkdir -p /var/log/pxos
chmod 755 /var/run/pxos
chmod 755 /var/log/pxos

echo "✓ Runtime directories created"
echo

# Step 5: Check GPU
echo "Step 5: Detecting GPUs..."
echo

if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
elif command -v rocm-smi &> /dev/null; then
    echo "AMD GPU detected:"
    rocm-smi --showproductname
elif lspci | grep -i vga | grep -i intel &> /dev/null; then
    echo "Intel GPU detected:"
    lspci | grep -i vga | grep -i intel
else
    echo "WARNING: No GPU detected!"
    echo "The hypervisor will run but won't have GPU acceleration"
fi

echo
echo "================================================================"
echo "Installation Complete!"
echo "================================================================"
echo
echo "Next steps:"
echo
echo "1. Start the hypervisor:"
echo "   systemctl start pxos-gpu-hypervisor"
echo
echo "2. Enable at boot (optional):"
echo "   systemctl enable pxos-gpu-hypervisor"
echo
echo "3. Check status:"
echo "   systemctl status pxos-gpu-hypervisor"
echo
echo "4. View logs:"
echo "   journalctl -u pxos-gpu-hypervisor -f"
echo
echo "5. Start a VM with GPU:"
echo "   $INSTALL_DIR/start_vm_with_gpu.sh my-vm 4096 2 guest.qcow2"
echo
echo "================================================================"
