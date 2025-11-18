# Pixel-Native Serial Driver - Quick Start

## Installation

```bash
# Install dependencies
pip install numpy pillow wgpu

# Or if wgpu not available, CPU-only mode works with just:
pip install numpy pillow
```

## Usage

### Create a pixel-encoded driver

```bash
# Create driver with default message
./create_driver.py

# Create with custom message
./create_driver.py -m "Hello, pxOS!" -o my_driver.pxi

# Visualize the driver
./create_driver.py -m "Test" -v
```

### Test the driver

```bash
# Test with inline driver (no file)
./test_serial.py -m "Hello from GPU!"

# Test with PXI file
./create_driver.py -m "GPU Rocks!" -o test.pxi
./test_serial.py -i test.pxi

# CPU-only mode (no WebGPU required)
./test_serial.py --cpu-only
```

## Expected Output

```
Creating inline driver for: 'Hello from GPU!'
Driver size: 256x1 pixels

============================================================
CPU Reference Implementation
============================================================
Output: 'Hello from GPU!'
Execution time: 1,234 ns (1.23 µs)

============================================================
GPU WebGPU Implementation
============================================================
Output: 'Hello from GPU!'
Execution time: 567 ns (0.57 µs)
Statistics:
  Operations executed: 15
  Characters written: 15
  Errors: 0

============================================================
Performance Comparison
============================================================
CPU time:  1,234 ns (1.23 µs)
GPU time:  567 ns (0.57 µs)
Speedup:   2.18x

✅ Output matches! GPU implementation is correct.
```

## How It Works

1. **create_driver.py** encodes your message as RGBA pixels:
   - R = ASCII character
   - G = Baud rate (0x01 = 115200)
   - B = Line control (0x03 = 8N1)
   - A = Operation type (0x01 = write)

2. **serial_driver.wgsl** executes operations in parallel on GPU:
   - Loads pixels from texture
   - Decodes RGBA → serial operations
   - Writes to simulated UART buffer

3. **test_serial.py** runs both CPU and GPU implementations:
   - CPU: Sequential processing (reference)
   - GPU: Parallel processing via WebGPU
   - Compares output and performance

## Architecture

```
Message String
    ↓
create_driver.py (encode as pixels)
    ↓
serial_driver.pxi (PNG image)
    ↓
test_serial.py (load to GPU texture)
    ↓
serial_driver.wgsl (GPU shader)
    ↓
Output String + Performance Stats
```

## Next Steps

If this proof-of-concept works:

1. **Benchmark on real hardware** - Measure actual GPU vs CPU latency
2. **Test with longer messages** - Batch operations (256+ characters)
3. **Add real UART access** - Replace simulated buffer with MMIO
4. **Integrate with hypervisor** - Load drivers via INT 13h
5. **Expand to other drivers** - Network, disk, graphics

## Troubleshooting

**WebGPU not available:**
```
pip install wgpu
# Or run with --cpu-only
```

**Import errors:**
```
pip install numpy pillow
```

**Wrong directory:**
```
cd pixel_drivers/serial_driver_poc/
./test_serial.py
```

## Files

- `create_driver.py` - Generate pixel-encoded drivers
- `serial_driver.wgsl` - GPU shader (WGSL)
- `test_serial.py` - Test harness and benchmark
- `QUICKSTART.md` - This file

## Performance Goals

| Metric | CPU | GPU | Goal |
|--------|-----|-----|------|
| Latency | 10 µs | <1 µs | 10x faster |
| Throughput | 115 KB/s | >10 MB/s | 87x faster |
| Parallelism | 1 char | 256 chars | 256x parallel |

---

**Status**: Proof-of-concept ready for testing
**Next**: Run on real hardware and measure performance
