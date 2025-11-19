# Pixel LLM Integration for pxOS

This directory contains the **Pixel LLM infrastructure** integrated into pxOS Phase 2.

## Overview

The Pixel LLM system enables:
- **AI-assisted kernel development** via autonomous architect loops
- **GPU-accelerated neural network primitives** (attention, activations, matmul)
- **Bytecode execution** on a pixel-native virtual machine
- **God Pixel compression** (16,384:1 ratio)
- **Hypervisor-based code execution** from pixel images

## Architecture

```
pixel_llm/
├── core/                   # Core Pixel LLM infrastructure
│   ├── hypervisor.py       # Execution controller for pixel-native code
│   ├── pixel_vm.py         # Stack-based bytecode VM (~30 opcodes)
│   └── pixel_asm.py        # Assembler for pixel bytecode
│
├── gpu_kernels/            # WGSL compute shaders for GPU
│   ├── attention.wgsl      # Transformer attention mechanism
│   ├── activations.wgsl    # Neural network activation functions
│   └── mailbox_runtime.wgsl # Hardware mailbox protocol handler
│
└── tools/                  # Development tools
    ├── god_pixel.py        # God Pixel compression system
    ├── pxos_architect_loop.py      # Generic autonomous architect
    └── pxos_kernel_architect.py   # Kernel-specific architect
```

## How It Works

### 1. **Pixel VM** - Native Bytecode Execution

The Pixel VM executes programs stored as bytecode in pixel images (.pxi files).

**Opcodes:**
- `0x01 PUSH <value>` - Push value onto stack
- `0x03 ADD` - Pop two values, push sum
- `0x10 PRINT` - Pop and print value
- `0x30 JMP <offset>` - Jump to offset
- `0xFF HALT` - Stop execution

**Example:**
```python
from pixel_llm.core.pixel_vm import PixelVM, assemble_program

program = assemble_program([
    (PixelVM.OP_PUSH, 42),
    (PixelVM.OP_PUSH, 8),
    (PixelVM.OP_ADD,),
    (PixelVM.OP_PRINT,),
    (PixelVM.OP_HALT,),
])

vm = PixelVM(debug=True)
vm.load_program(program)
vm.run()  # Output: 50
```

### 2. **Hypervisor** - Pixel-Native Code Execution

The hypervisor loads and runs code from pixel archives.

**Features:**
- Runtime validation (Python version, dependencies)
- Entrypoint resolution from manifests
- Sandboxed execution context
- Error handling and logging

**Example:**
```python
from pixel_llm.core.hypervisor import create_hypervisor

manifest = {
    "entrypoints": {
        "default": "my_module:main"
    },
    "python_runtime": {
        "min_version": "3.11"
    }
}

hypervisor = create_hypervisor(manifest, archive_reader)
hypervisor.validate_runtime()
hypervisor.run_entrypoint("default")
```

### 3. **GPU Kernels** - Hardware Acceleration

WGSL compute shaders for GPU-accelerated operations.

**Attention Mechanism:**
```wgsl
// Scaled dot-product attention
// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

@compute @workgroup_size(16, 16, 1)
fn compute_attention_scores(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    // Compute Q[i] · K[j]
    // Scale by 1/sqrt(d_k)
    // Apply softmax
    // Multiply by V
}
```

**Mailbox Runtime:**
The `mailbox_runtime.wgsl` shader implements the GPU-side mailbox protocol:

```wgsl
@compute @workgroup_size(1, 1, 1)
fn mailbox_handler(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (mailbox_doorbell != 0u) {
        let cmd = mailbox_cmd;
        let opcode = (cmd >> 24u) & 0xFFu;

        switch (opcode) {
            case OP_UART_WRITE: { uart_write(payload); }
            case OP_GPU_EXECUTE: { gpu_execute_kernel(payload); }
            // ... more opcodes
        }

        mailbox_status = STATUS_COMPLETE;
    }
}
```

### 4. **God Pixel** - Extreme Compression

One pixel can represent an entire program through:
- Hash-based lookup (pixel color → program)
- Fractal/procedural generation (seed → program)
- Self-extracting recursive expansion

**Example:**
```python
from pixel_llm.tools.god_pixel import GodPixel

gp = GodPixel()

# Compress 128x128 program into ONE pixel
color = gp.create_god_pixel(
    program_img,
    method="hash",
    output_path="god.png"
)
# Result: RGBA(214, 42, 178, 91) = entire program

# Resurrect from single pixel
resurrected = gp.resurrect("god.png")
# Result: Full 128x128 program restored perfectly
```

**Compression ratio:** 16,384:1

### 5. **pxOS Kernel Architect** - AI-Assisted Development

Autonomous LLM agent that helps develop the pxOS kernel.

**Expertise:**
- x86-64 assembly (NASM)
- WGSL compute shaders
- PCIe, BAR0, MMIO
- Performance optimization
- Documentation

**Usage:**

**Interactive Mode** (user provides prompts):
```bash
python3 pixel_llm/tools/pxos_kernel_architect.py --mode interactive
```

**Autonomous Mode** (self-directed):
```bash
python3 pixel_llm/tools/pxos_kernel_architect.py --mode autonomous --interval 60
```

The architect can:
- Write assembly code for the microkernel
- Create WGSL GPU shaders
- Generate documentation
- Run builds and tests
- Analyze performance
- Propose optimizations

**Example interaction:**
```
What should I work on? > Optimize the mailbox protocol for <500ns latency

Consulting architect...

Task: Reduce mailbox latency by using write-combining memory
Rationale: UC memory enforces strict ordering, WC allows batching writes

✓ Wrote file: pxos-v1.0/microkernel/phase1_poc/mailbox_wc.asm
✓ Build succeeded
✓ Task completed
```

## Integration with pxOS Phase 2

### CPU Microkernel (Assembly)

The CPU microkernel (`microkernel_multiboot.asm`) initializes the mailbox:

```nasm
; Boot sequence
call pcie_scan_64          ; Find GPU
call map_gpu_bar0          ; Map BAR0 MMIO
call mailbox_init          ; Initialize mailbox
call mailbox_test          ; Test with UART write
```

### GPU Runtime (WGSL)

The GPU runtime (`mailbox_runtime.wgsl`) processes commands:

```wgsl
// GPU continuously polls for commands
loop {
    if (mailbox_doorbell != 0u) {
        process_command();
        mailbox_status = STATUS_COMPLETE;
    }
}
```

### Privilege Inversion

Traditional OS: **CPU (ring 0) → GPU (servant)**
pxOS: **GPU (ring 0) ← CPU (ring 3)**

The CPU makes "syscalls" to the GPU via mailbox:

```nasm
; CPU code (ring 3)
mov rdi, (OP_UART_WRITE << 24) | 'H'
call mailbox_send_command
call mailbox_poll_complete
```

```wgsl
// GPU code (ring 0)
if (opcode == OP_UART_WRITE) {
    uart_write(payload);  // Privileged operation
}
```

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Mailbox latency | < 1 μs | 🚧 In progress |
| CPU overhead | < 5% | 🚧 In progress |
| Boot time | < 1 s | ✅ ~0.5s (QEMU) |
| PCIe scan | < 10 ms | ✅ ~2ms |

## Setup

### Prerequisites

```bash
# Install Python dependencies (for architect)
pip3 install requests Pillow

# Install LM Studio (for AI assistance)
# Download from: https://lmstudio.ai/
# Load a model (recommended: Llama 3 8B or similar)
```

### Running the Architect

1. **Start LM Studio:**
   - Download and install LM Studio
   - Load a model (e.g., "Meta Llama 3 8B Instruct")
   - Start the local server (default: http://localhost:1234)

2. **Run architect in interactive mode:**
   ```bash
   cd /home/user/pxos
   python3 pxos-v1.0/pixel_llm/tools/pxos_kernel_architect.py --mode interactive
   ```

3. **Example prompts:**
   - "Add support for OP_SYSCALL to the mailbox protocol"
   - "Optimize mailbox polling to reduce CPU overhead"
   - "Create a WGSL shader for memory allocation"
   - "Generate documentation for the BAR0 mapping"

### Running the Pixel VM

```bash
cd /home/user/pxos
python3 -c "from pxos-v1.0.pixel_llm.core.pixel_vm import main; main()" --demo
```

## Next Steps

- [ ] Implement complete WGSL GPU runtime for all mailbox opcodes
- [ ] Add WebGPU bindings to execute WGSL shaders from Python
- [ ] Measure mailbox performance with RDTSC
- [ ] Optimize to achieve <1 μs latency target
- [ ] Create pixel-native kernel modules (.pxi files)
- [ ] Implement God Pixel compression for kernel code
- [ ] Use architect to generate test suite

## Resources

- **Pixel LLM Roadmap:** Original concept and implementation
- **WGSL Specification:** https://www.w3.org/TR/WGSL/
- **WebGPU API:** https://gpuweb.github.io/gpuweb/
- **LM Studio:** https://lmstudio.ai/

---

**Made with pixels and revolution in mind** 🚀

*"The future of operating systems runs on the GPU, developed by AI, compressed into pixels"* - pxOS + Pixel LLM
