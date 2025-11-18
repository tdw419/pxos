# Pixel-Native Device Drivers

## Vision

**Replace CPU-based device drivers with GPU-native pixel-encoded operations.**

Traditional drivers execute on the CPU sequentially. Pixel-native drivers encode hardware operations as pixels and execute on the GPU in parallel, enabling:

- **10-100x lower latency** (GPU shader vs CPU syscall)
- **Massive parallelism** (256+ operations vs 1 operation)
- **Zero-copy architecture** (direct VRAM→hardware DMA)
- **Visual debugging** (see I/O patterns as images)

## Architecture

```
Traditional Path:          Pixel-Native Path:
─────────────────          ──────────────────
Application                Application
    ↓ syscall                  ↓ GPU call
Kernel                     GPU Shader
    ↓ function call            ↓ texture load
Driver (RAM)               Pixel Driver (VRAM)
    ↓ MMIO write               ↓ direct MMIO
Hardware Register          Hardware Register
```

The GPU becomes the system's primary I/O controller, processing thousands of hardware operations in parallel.

## Proof-of-Concept: Serial Driver

This directory contains a proof-of-concept that implements a simple serial (UART) driver using this architecture.

### 1. Pixel Encoding (`create_driver.py`)

Each RGBA pixel of a PNG image (`serial_driver.pxi`) encodes one UART operation:
- **R (8 bits)**: Character/data byte to write.
- **G (8 bits)**: Baud rate divisor (e.g., 0x01 for 115200).
- **B (8 bits)**: Line control flags (e.g., 0x03 for 8N1).
- **A (8 bits)**: Operation type (e.g., 0x01 for write).

A string like "Hello" becomes a 5x1 pixel image.

### 2. GPU Shader (`serial_driver.wgsl`)

A WebGPU compute shader (`.wgsl`) is written to execute these pixel-encoded operations.
- It is dispatched with a workgroup size of 256, meaning it processes **256 serial operations simultaneously**.
- Each thread reads one pixel, decodes the RGBA channels, and writes the character to a simulated UART MMIO buffer.

### 3. Test Harness (`test_serial.py`)

A Python script uses `wgpu-py` to:
1.  Invoke `create_driver.py` to generate a `serial_driver.pxi` from a message.
2.  Load the PXI image as a GPU texture.
3.  Load and execute the `serial_driver.wgsl` compute shader.
4.  Read back the output from the simulated UART buffer and verify it.
5.  It also includes a pure CPU reference implementation to benchmark against.

## Performance Goals

- **Latency**: A single GPU dispatch should be significantly faster than thousands of individual CPU-based `out` instructions.
- **Throughput**: The system should be able to process an entire 256-character buffer in a single parallel operation.
- **CPU Offload**: The CPU should be entirely free while the GPU handles the I/O.

## Integration with pxHV Hypervisor

The next step is to integrate this model with the hypervisor:
1.  The hypervisor will load the `serial_driver.pxi` file from its virtual disk into a memory region accessible by the GPU.
2.  When a guest VM performs a `PIO` or `MMIO` operation, the VM exit handler will not emulate the instruction on the CPU.
3.  Instead, it will **batch the I/O request** by creating or updating a PXI image in memory.
4.  It will then **dispatch the GPU compute shader** to process the entire batch of I/O operations in parallel.
5.  Once the GPU signals completion, the hypervisor will resume the guest.

This hybrid model combines the proven foundation of the x86 hypervisor with the revolutionary performance of GPU-native drivers.
