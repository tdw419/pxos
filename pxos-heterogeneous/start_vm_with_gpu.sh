#!/bin/bash
# Start a VM with pxOS GPU Hypervisor support

set -e

# Configuration
VM_NAME="${1:-guest-vm-1}"
VM_MEMORY="${2:-4096}"    # MB
VM_CPUS="${3:-2}"
VM_IMAGE="${4:-guest.qcow2}"
GPU_MEMORY_QUOTA="${5:-8000}"  # MB

PXOS_GPU_SOCKET="/var/run/pxos-gpu.sock"

echo "================================================================"
echo "Starting VM with pxOS GPU Hypervisor"
echo "================================================================"
echo "VM Name:           $VM_NAME"
echo "Memory:            ${VM_MEMORY}MB"
echo "CPUs:              $VM_CPUS"
echo "Disk Image:        $VM_IMAGE"
echo "GPU Memory Quota:  ${GPU_MEMORY_QUOTA}MB"
echo "================================================================"
echo

# Check if hypervisor is running
if [ ! -S "$PXOS_GPU_SOCKET" ]; then
    echo "ERROR: pxOS GPU Hypervisor not running!"
    echo "Please start it first:"
    echo "  sudo systemctl start pxos-gpu-hypervisor"
    echo "Or run:"
    echo "  sudo python3 gpu_hypervisor.py --daemon"
    exit 1
fi

echo "✓ GPU Hypervisor detected"

# Check if VM image exists
if [ ! -f "$VM_IMAGE" ]; then
    echo "ERROR: VM image not found: $VM_IMAGE"
    echo
    echo "Create a VM image first:"
    echo "  qemu-img create -f qcow2 $VM_IMAGE 20G"
    echo "Then install OS:"
    echo "  qemu-system-x86_64 -m 4096 -cdrom ubuntu.iso -drive file=$VM_IMAGE"
    exit 1
fi

echo "✓ VM image found: $VM_IMAGE"

# Check KVM support
if [ ! -e /dev/kvm ]; then
    echo "WARNING: KVM not available, will use slower TCG emulation"
    KVM_OPTS=""
else
    echo "✓ KVM acceleration available"
    KVM_OPTS="-enable-kvm"
fi

echo
echo "Starting VM..."
echo

# Start QEMU with GPU hypervisor support
exec qemu-system-x86_64 \
    $KVM_OPTS \
    -name "$VM_NAME" \
    -m "$VM_MEMORY" \
    -smp "$VM_CPUS" \
    -drive file="$VM_IMAGE",format=qcow2,if=virtio \
    -device virtio-net-pci,netdev=net0 \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -device virtio-gpu-pxos,max_outputs=1,memory_quota="$GPU_MEMORY_QUOTA" \
    -chardev socket,id=gpu0,path="$PXOS_GPU_SOCKET" \
    -device virtio-serial \
    -device virtserialport,chardev=gpu0,name=pxos.gpu.0 \
    -serial mon:stdio \
    -display gtk,gl=on \
    "$@"

# Note: The last "$@" allows passing additional QEMU options
# Example: ./start_vm_with_gpu.sh my-vm 8192 4 guest.qcow2 10000 -snapshot
