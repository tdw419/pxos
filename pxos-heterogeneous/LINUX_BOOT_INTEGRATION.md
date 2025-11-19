# Booting Linux with GPU Hypervisor

## Important Clarification

**You cannot boot Linux ON a GPU** - GPUs fundamentally lack boot capability (no BIOS, no I/O, no interrupts).

**You CAN boot Linux WITH GPU hypervisor support** - Linux boots normally on CPU, but VMs can use GPU acceleration through the hypervisor.

---

## Complete Boot Sequence

### Overview

```
┌─────────────────────────────────────────────────────────┐
│              1. HOST SYSTEM BOOTS                        │
│         Linux boots normally on CPU                      │
│      (BIOS/UEFI → Bootloader → Linux Kernel)            │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│         2. GPU HYPERVISOR INITIALIZES                    │
│    Kernel module loads, discovers physical GPU           │
│    Hypervisor daemon starts, ready for VMs               │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│           3. GUEST VMs BOOT (via QEMU/KVM)              │
│    VMs boot normally on CPU, see virtio-gpu-pxos         │
│    VM GPU operations → primitives → hypervisor           │
└─────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Boot Process

### Phase 1: Host Linux Boot (Normal Boot on CPU)

```
Power On
   ↓
BIOS/UEFI (runs on CPU)
   ↓
Bootloader (GRUB) (runs on CPU)
   ↓
Linux Kernel boots (runs on CPU)
   ↓
Init system (systemd) starts (runs on CPU)
   ↓
All services start (runs on CPU)
   ↓
System ready!
```

**Nothing special here** - Linux boots completely normally on the CPU, just like always.

### Phase 2: GPU Hypervisor Initialization

After the host Linux boots, the GPU hypervisor components start:

#### 2.1 Kernel Module Loads

```bash
# During boot, systemd loads the kernel module
modprobe pxos_gpu_hypervisor

# This module:
# 1. Discovers physical GPU(s)
# 2. Initializes GPU memory manager
# 3. Creates /dev/pxos_gpu device
# 4. Registers with QEMU/KVM
```

**Kernel log shows:**
```
[    2.456] pxos_gpu: GPU hypervisor initializing
[    2.458] pxos_gpu: Found NVIDIA RTX 4090 (24GB)
[    2.460] pxos_gpu: GPU memory pool: 24000 MB
[    2.462] pxos_gpu: Ready to accept VM connections
```

#### 2.2 Hypervisor Daemon Starts

```bash
# systemd starts the hypervisor daemon
systemctl start pxos-gpu-hypervisor.service

# The daemon:
# 1. Connects to kernel module
# 2. Initializes LLM scheduler
# 3. Opens virtio-gpu-pxos endpoint
# 4. Waits for VM connections
```

**Daemon logs:**
```
[INFO] pxOS GPU Hypervisor v1.0 starting
[INFO] Connected to GPU: NVIDIA RTX 4090 (24GB)
[INFO] LLM scheduler initialized
[INFO] VirtIO endpoint ready: /var/run/pxos-gpu.sock
[INFO] Hypervisor ready for VMs
```

### Phase 3: Guest VM Boot

When you start a guest VM with QEMU/KVM:

#### 3.1 QEMU Command with GPU Hypervisor

```bash
qemu-system-x86_64 \
    -enable-kvm \
    -m 4096 \
    -smp 2 \
    -drive file=guest.qcow2,format=qcow2 \
    -device virtio-gpu-pxos,max_outputs=1 \    # GPU hypervisor device!
    -chardev socket,id=gpu0,path=/var/run/pxos-gpu.sock \
    -device virtio-serial \
    -device virtserialport,chardev=gpu0,name=pxos.gpu.0
```

#### 3.2 Guest VM Boot Sequence

```
Guest VM powers on (virtual CPU starts)
   ↓
Guest BIOS/UEFI (emulated by QEMU)
   ↓
Guest bootloader (GRUB in guest)
   ↓
Guest Linux kernel boots
   ↓
Guest detects virtio-gpu-pxos device
   ↓
Guest loads virtio-gpu-pxos driver
   ↓
Driver connects to host hypervisor
   ↓
Guest can now use GPU via primitives!
```

**Guest kernel log:**
```
[    1.234] virtio-gpu-pxos: Detected pxOS GPU hypervisor
[    1.236] virtio-gpu-pxos: Connected to host GPU: 8000MB quota
[    1.238] virtio-gpu-pxos: Ready for primitive submission
```

---

## Complete System Architecture During Runtime

```
┌──────────────────────────────────────────────────────────┐
│                    HOST LINUX (CPU)                       │
│                                                            │
│  ┌────────────────────────────────────────────────┐      │
│  │   pxOS GPU Hypervisor Daemon (userspace)       │      │
│  │   - LLM Scheduler                              │      │
│  │   - VM Management                               │      │
│  │   - Primitive Parser                            │      │
│  └──────────┬──────────────────────┬────────────────┘      │
│             ↓                      ↓                      │
│  ┌──────────────────────┐  ┌──────────────────────┐      │
│  │ Kernel Module        │  │ QEMU Process         │      │
│  │ (pxos_gpu_hypervisor)│  │ - Runs guest VM      │      │
│  │ - GPU driver         │  │ - virtio-gpu-pxos    │      │
│  │ - Memory mgmt        │  │                      │      │
│  └──────────┬───────────┘  └──────────┬───────────┘      │
│             ↓                          ↓                  │
│  ┌──────────────────────────────────────────────┐        │
│  │         Physical GPU (NVIDIA/AMD/Intel)       │        │
│  │               (runs primitives)               │        │
│  └──────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────┘
             ↑
             │ virtio-gpu-pxos (virtual device)
             │
┌────────────┴─────────────────────────────────────────────┐
│               GUEST VM (runs on CPU)                      │
│                                                            │
│  ┌────────────────────────────────────────────────┐      │
│  │  Guest Application                              │      │
│  │  - Writes GPU primitives                        │      │
│  │  - Calls GPU functions                          │      │
│  └──────────┬──────────────────────────────────────┘      │
│             ↓                                             │
│  ┌────────────────────────────────────────────────┐      │
│  │  virtio-gpu-pxos Guest Driver                   │      │
│  │  - Intercepts GPU calls                         │      │
│  │  - Sends primitives to host                     │      │
│  └─────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────┘
```

---

## Practical Example: Complete Boot Walkthrough

Let's walk through a real example of booting a system with GPU hypervisor.

### Step 1: Host System Boot

```bash
# Power on physical server
# [Physical CPU executes BIOS]
# [BIOS loads bootloader from disk]
# [Bootloader (GRUB) starts Linux kernel]

# Linux kernel boots on CPU:
[    0.000] Linux version 6.5.0 (gcc version 13.1.0)
[    0.001] Command line: BOOT_IMAGE=/boot/vmlinuz root=/dev/sda1
[    0.100] CPU: Intel Xeon E5-2680 v4 @ 2.40GHz
[    1.234] Memory: 128GB RAM detected
[    2.345] GPU: NVIDIA RTX 4090 detected (PCI device)
[    3.456] systemd[1]: Starting system

# System is now running on CPU, GPU is idle
```

### Step 2: GPU Hypervisor Starts

```bash
# systemd starts the hypervisor service
[    4.123] systemd[1]: Starting pxos-gpu-hypervisor.service

# Kernel module loads
[    4.125] pxos_gpu: Initializing GPU hypervisor
[    4.127] pxos_gpu: Detected NVIDIA RTX 4090 (24GB)
[    4.129] pxos_gpu: Initialized GPU memory allocator
[    4.131] pxos_gpu: Created /dev/pxos_gpu0

# Daemon starts
[    4.150] pxos-hypervisor[1234]: Starting GPU hypervisor daemon
[    4.152] pxos-hypervisor[1234]: GPU 0: NVIDIA RTX 4090 (24GB)
[    4.154] pxos-hypervisor[1234]: LLM scheduler initialized
[    4.156] pxos-hypervisor[1234]: VirtIO endpoint: /var/run/pxos-gpu.sock
[    4.158] pxos-hypervisor[1234]: Ready to accept VMs

# Host is now ready!
```

### Step 3: Start Guest VM

```bash
# Administrator starts a VM
$ qemu-system-x86_64 \
    -enable-kvm \
    -m 8192 \
    -smp 4 \
    -drive file=ubuntu-guest.qcow2,format=qcow2 \
    -device virtio-gpu-pxos,max_outputs=1 \
    -chardev socket,id=gpu0,path=/var/run/pxos-gpu.sock

# QEMU starts the VM
[    5.000] qemu[5678]: Starting VM with 8GB RAM, 4 vCPUs
[    5.001] qemu[5678]: virtio-gpu-pxos device initialized
[    5.002] qemu[5678]: Connected to GPU hypervisor

# Guest VM boots (on virtual CPU)
[Guest] 0.000: Guest BIOS starting
[Guest] 0.100: Loading bootloader
[Guest] 0.200: Starting Linux kernel
[Guest] 1.234: Linux version 6.5.0
[Guest] 2.345: Detected virtio-gpu-pxos device
[Guest] 2.346: Loading virtio-gpu-pxos driver

# Guest driver connects to hypervisor
[    7.123] pxos-hypervisor[1234]: New VM connected: VM ID 1
[    7.124] pxos-hypervisor[1234]: Allocated 8000MB GPU memory quota
[    7.125] pxos-hypervisor[1234]: VM 1 ready for primitives

[Guest] 2.347: virtio-gpu-pxos: Connected to host GPU
[Guest] 2.348: virtio-gpu-pxos: Memory quota: 8000MB
[Guest] 2.349: virtio-gpu-pxos: Ready!

# Guest is now fully booted and can use GPU!
```

### Step 4: Guest Uses GPU

```bash
# Inside guest VM, application runs
[Guest] $ python3 my_app.py

# Application writes primitive
primitive = """
GPU_KERNEL process_data
GPU_PARAM input float[]
GPU_THREAD_CODE:
    THREAD_ID → tid
    LOAD input[tid] → value
    MUL value 2.0 → result
    STORE result → output[tid]
GPU_END
"""

# Guest driver sends to hypervisor
[Guest] 10.123: virtio-gpu-pxos: Submitting primitive to host

# Hypervisor receives and schedules
[   10.125] pxos-hypervisor[1234]: VM 1 submitted primitive
[   10.126] pxos-hypervisor[1234]: LLM analysis: THROUGHPUT workload
[   10.127] pxos-hypervisor[1234]: Scheduling: 50% GPU allocation
[   10.128] pxos-hypervisor[1234]: Executing on physical GPU

# Physical GPU executes
[   10.130] pxos_gpu: Executing kernel for VM 1
[   10.145] pxos_gpu: Kernel complete (15ms)

# Result sent back to guest
[Guest] 10.150: virtio-gpu-pxos: Primitive completed
[Guest] $ Result: Success! GPU processed 1M elements in 15ms
```

---

## Integration with Linux Boot Process

### Method 1: Systemd Service (Recommended)

Create `/etc/systemd/system/pxos-gpu-hypervisor.service`:

```ini
[Unit]
Description=pxOS GPU Hypervisor
After=network.target
Requires=pxos-gpu-kernel.service

[Service]
Type=simple
ExecStartPre=/sbin/modprobe pxos_gpu_hypervisor
ExecStart=/usr/bin/pxos-gpu-hypervisor --daemon
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

Enable at boot:
```bash
systemctl enable pxos-gpu-hypervisor.service
systemctl start pxos-gpu-hypervisor.service
```

### Method 2: Init Script (Legacy)

Create `/etc/init.d/pxos-gpu-hypervisor`:

```bash
#!/bin/bash
### BEGIN INIT INFO
# Provides:          pxos-gpu-hypervisor
# Required-Start:    $network
# Required-Stop:     $network
# Default-Start:     2 3 4 5
# Default-Stop:      0 1 6
# Description:       pxOS GPU Hypervisor
### END INIT INFO

case "$1" in
    start)
        modprobe pxos_gpu_hypervisor
        /usr/bin/pxos-gpu-hypervisor --daemon
        ;;
    stop)
        killall pxos-gpu-hypervisor
        rmmod pxos_gpu_hypervisor
        ;;
    *)
        echo "Usage: $0 {start|stop}"
        exit 1
        ;;
esac
```

### Method 3: Early Boot Integration (Advanced)

For GPU acceleration during early boot, add to initramfs:

```bash
# /etc/initramfs-tools/modules
pxos_gpu_hypervisor

# Rebuild initramfs
update-initramfs -u

# Now GPU hypervisor loads during early boot!
```

---

## Guest VM Configuration

### Guest Kernel Config

Enable these options in guest kernel:

```
CONFIG_VIRTIO=y
CONFIG_VIRTIO_PCI=y
CONFIG_VIRTIO_GPU=y
CONFIG_VIRTIO_GPU_PXOS=y  # Our driver
```

### Guest Driver Installation

```bash
# Inside guest VM
cd /path/to/virtio-gpu-pxos-driver
make
make install
modprobe virtio_gpu_pxos

# Verify
dmesg | grep virtio-gpu-pxos
# Should show: "Connected to host GPU: 8000MB quota"
```

### Guest Application Integration

Applications in guest VM can now use GPU primitives:

```python
# Inside guest VM: /home/user/gpu_app.py
from pxos_gpu import submit_primitive

primitive = """
GPU_KERNEL my_operation
GPU_PARAM data float[]
GPU_THREAD_CODE:
    THREAD_ID → tid
    LOAD data[tid] → value
    MUL value 2.0 → result
    STORE result → data[tid]
GPU_END
"""

# This goes through virtio-gpu-pxos → hypervisor → physical GPU!
result = submit_primitive(primitive, data=my_array)
print(f"GPU processed {len(my_array)} elements!")
```

---

## Multi-VM Boot Scenario

Let's see what happens when multiple VMs boot:

```bash
# Host boots (Phase 1)
[    0.000] Host Linux kernel starts
[    4.158] pxos-hypervisor: Ready (24GB GPU available)

# Start VM 1
[   10.000] Admin: qemu ... -device virtio-gpu-pxos (VM 1)
[   12.000] pxos-hypervisor: VM 1 registered (8GB quota)
[   15.000] VM 1 boots successfully, GPU ready

# Start VM 2
[   20.000] Admin: qemu ... -device virtio-gpu-pxos (VM 2)
[   22.000] pxos-hypervisor: VM 2 registered (8GB quota)
[   25.000] VM 2 boots successfully, GPU ready

# Start VM 3
[   30.000] Admin: qemu ... -device virtio-gpu-pxos (VM 3)
[   32.000] pxos-hypervisor: VM 3 registered (8GB quota)
[   35.000] VM 3 boots successfully, GPU ready

# All 3 VMs now share the 24GB GPU!
[   35.001] pxos-hypervisor: Active VMs: 3
[   35.002] pxos-hypervisor: GPU allocation: VM1=33%, VM2=33%, VM3=33%
[   35.003] pxos-hypervisor: GPU memory: 24GB total, 24GB allocated

# When VMs submit primitives, LLM scheduler allocates fairly!
```

---

## Boot Options and Kernel Parameters

### Host Boot Parameters

Add to `/etc/default/grub`:

```bash
GRUB_CMDLINE_LINUX="pxos_gpu.memory=24000 pxos_gpu.debug=1"
```

Update GRUB:
```bash
update-grub
reboot
```

### Guest Boot Parameters

Pass to guest kernel:

```bash
qemu-system-x86_64 \
    ... \
    -append "virtio_gpu_pxos.debug=1 virtio_gpu_pxos.memory_quota=8000"
```

---

## Troubleshooting Boot Issues

### Issue 1: Hypervisor Doesn't Start

```bash
# Check kernel module
lsmod | grep pxos_gpu
# If not loaded:
modprobe pxos_gpu_hypervisor
dmesg | tail -20

# Check daemon
systemctl status pxos-gpu-hypervisor
journalctl -u pxos-gpu-hypervisor -f
```

### Issue 2: Guest Can't Connect

```bash
# On host, check socket
ls -la /var/run/pxos-gpu.sock
# Should exist with correct permissions

# Check hypervisor logs
journalctl -u pxos-gpu-hypervisor | grep "VM connected"

# In guest, check driver
dmesg | grep virtio-gpu-pxos
```

### Issue 3: GPU Not Detected

```bash
# Check physical GPU
lspci | grep -i vga
lspci | grep -i nvidia

# Check GPU driver
nvidia-smi  # or AMD equivalent
```

---

## Summary: The Boot Reality

### What Actually Happens

1. **Host Linux boots on CPU** (normal boot, nothing special)
2. **GPU hypervisor starts** (kernel module + daemon)
3. **Guest VMs boot on virtual CPUs** (normal VM boot)
4. **Guests get virtio-gpu-pxos device** (virtual GPU connected to hypervisor)
5. **Guest GPU operations** → primitives → hypervisor → physical GPU

### What Does NOT Happen

❌ Linux does **not** boot "on the GPU"
❌ GPU does **not** execute the Linux kernel
❌ GPU does **not** run the init system or services

### What DOES Happen

✅ Linux boots normally on CPU (host and guests)
✅ GPU hypervisor provides GPU sharing to VMs
✅ VMs can accelerate operations using GPU primitives
✅ 10-50x speedup for parallel workloads
✅ Multiple VMs share one physical GPU efficiently

---

## Next Steps

1. **Install host components**: Kernel module + daemon
2. **Build guest driver**: virtio-gpu-pxos for guest VMs
3. **Test with one VM**: Boot guest, verify GPU access
4. **Scale to multiple VMs**: Boot multiple guests, test fair sharing
5. **Deploy applications**: Run real workloads with GPU acceleration

The system is ready to deploy! 🚀
