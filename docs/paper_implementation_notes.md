# Implementation Plan Based on Academic Paper

## Summary

The academic paper "Architectural Patterns and Implementation Strategies for High-Performance
GPU-Resident Virtual Machines" provides detailed validation and implementation guidance for
our Pixel OS architecture.

## Key Validations

1. **Architecture is Sound**
   - Paper describes full Linux running on GPU compute shaders
   - Same fundamental approach: VM bytecode → WGSL execution
   - Proven in production (Linux in VRChat project)

2. **Memory Model Confirmed**
   - Storage buffers are correct choice for RAM (128MB-2GB)
   - Alternative: Textures (2048x2048 RGBA32 = 64MB)
   - Register file must use private/workgroup memory

3. **Performance Characteristics**
   - Hundreds of kHz to low MHz achievable
   - Sufficient to boot Linux in minutes
   - Warp divergence is main bottleneck

## Critical Additions Needed

### 1. Bit Operations (Paper Section 2.1.2)

**Why:** Byte-addressable memory requires bit manipulation.

Add to `runtime/shader_vm.py`:
```python
class Opcode(IntEnum):
    # ... existing opcodes ...

    # Bit operations (NEW - from paper)
    BIT_SHL = 120    # Bit shift left
    BIT_SHR = 121    # Bit shift right
    BIT_AND = 122    # Bitwise AND
    BIT_OR = 123     # Bitwise OR
    BIT_XOR = 124    # Bitwise XOR
    BIT_NOT = 125    # Bitwise NOT

    # Memory operations with byte addressing (NEW)
    LOAD_BYTE = 130   # Load 8-bit value
    STORE_BYTE = 131  # Store 8-bit value
    LOAD_HALF = 132   # Load 16-bit value
    STORE_HALF = 133  # Store 16-bit value
    LOAD_WORD = 134   # Load 32-bit value (alias for LOAD)
    STORE_WORD = 135  # Store 32-bit value (alias for STORE)
```

### 2. WGSL Implementation (Paper Section 2.1.2)

Add to `runtime/shader_vm.wgsl`:
```wgsl
// Byte-addressable memory operations
fn load_byte(address: u32) -> u32 {
    let word_index = address >> 2u;
    let byte_offset = (address & 3u) * 8u;
    let word = bytecode[word_index];
    return (word >> byte_offset) & 0xFFu;
}

fn store_byte(address: u32, value: u32) {
    let word_index = address >> 2u;
    let shift = (address & 3u) * 8u;
    let mask = ~(0xFFu << shift);

    // Read-Modify-Write
    let old_word = bytecode[word_index];
    let new_word = (old_word & mask) | ((value & 0xFFu) << shift);
    bytecode[word_index] = new_word;
}

// Add to main switch statement:
case OP_BIT_SHL: {
    if (sp >= 2u) {
        let shift_amount = u32(stack[sp - 1u]);
        let value = u32(stack[sp - 2u]);
        stack[sp - 2u] = f32(value << shift_amount);
        sp -= 1u;
    }
}
case OP_BIT_SHR: {
    if (sp >= 2u) {
        let shift_amount = u32(stack[sp - 1u]);
        let value = u32(stack[sp - 2u]);
        stack[sp - 2u] = f32(value >> shift_amount);
        sp -= 1u;
    }
}
case OP_BIT_AND: {
    if (sp >= 2u) {
        let b = u32(stack[sp - 1u]);
        let a = u32(stack[sp - 2u]);
        stack[sp - 2u] = f32(a & b);
        sp -= 1u;
    }
}
case OP_LOAD_BYTE: {
    if (sp > 0u) {
        let address = u32(stack[sp - 1u]);
        stack[sp - 1u] = f32(load_byte(address));
    }
}
case OP_STORE_BYTE: {
    if (sp >= 2u) {
        let address = u32(stack[sp - 2u]);
        let value = u32(stack[sp - 1u]);
        store_byte(address, value);
        sp -= 2u;
    }
}
```

### 3. Time-Sliced Execution (Paper Section 3.4)

**Critical:** GPU watchdog timer kills long-running shaders!

Add to `runtime/webgpu_runtime.py`:
```python
class ShaderVMRuntime:
    def __init__(self, cycles_per_frame: int = 10000):
        self.cycles_per_frame = cycles_per_frame

    def execute_time_slice(self):
        """Execute fixed number of cycles per frame"""
        # Update uniform with cycle budget
        uniform_data = struct.pack(
            'ffffffff',
            time.time() - self.start_time,
            float(self.width),
            float(self.height),
            float(self.cycles_per_frame),  # Cycle budget
            0.0, 0.0, 0.0, 0.0
        )

        # Dispatch compute shader
        # Shader will execute exactly cycles_per_frame instructions
        # then save state and exit
```

### 4. Ring Buffer I/O (Paper Section 5.1)

**For keyboard input to terminal:**

Add to `runtime/async_io.py` (NEW FILE):
```python
class RingBufferIO:
    """Async I/O using ring buffers (Paper Section 5.1)"""

    def __init__(self, device, buffer_size: int = 256):
        self.buffer_size = buffer_size

        # Create ring buffer structure
        buffer_data = struct.pack(
            'II' + f'{buffer_size}I',
            0,  # write_head (atomic)
            0,  # read_head (atomic)
            *([0] * buffer_size)  # events
        )

        self.ring_buffer = device.create_buffer(
            size=len(buffer_data),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            mapped_at_creation=True
        )

    def push_keyboard_event(self, keycode: int):
        """CPU side: Push keyboard event"""
        # Read current write_head
        # Write event to events[write_head % buffer_size]
        # Increment write_head atomically

    def shader_code(self):
        """WGSL code for reading from ring buffer"""
        return """
        fn poll_keyboard() -> u32 {
            let write_head = atomicLoad(&ring.write_head);
            let read_head = atomicLoad(&ring.read_head);

            if (read_head != write_head) {
                let event = ring.events[read_head % 256u];
                atomicAdd(&ring.read_head, 1u);
                return event;
            }
            return 0u;
        }
        """
```

### 5. Font Rendering with Bit Operations

**Now we can implement actual text rendering:**

Add to `runtime/font_renderer.py` (NEW FILE):
```python
class BitmapFontRenderer:
    """Render 8x16 bitmap font using bit operations"""

    def compile_char_render(self, char_code: int) -> ShaderVM:
        """Compile character rendering to use BIT operations"""
        vm = ShaderVM()

        # Given: pixel position (x, y) within character cell
        # Load font bitmap row for this character
        vm.emit(Opcode.PUSH, char_code)      # Character code
        vm.emit(Opcode.PUSH, 16)             # 16 rows per char
        vm.emit(Opcode.MUL)                  # char * 16
        vm.emit(Opcode.PUSH, pixel_y)        # Add row offset
        vm.emit(Opcode.ADD)                  # font_index

        vm.emit(Opcode.LOAD)                 # Load bitmap row (8 bits)

        # Extract bit at pixel_x position
        vm.emit(Opcode.PUSH, 7)
        vm.emit(Opcode.PUSH, pixel_x)
        vm.emit(Opcode.SUB)                  # 7 - pixel_x
        vm.emit(Opcode.BIT_SHR)              # Shift to get bit
        vm.emit(Opcode.PUSH, 1)
        vm.emit(Opcode.BIT_AND)              # Mask to get single bit

        # If bit == 1: foreground color, else: background
        return vm
```

## Implementation Priority

### Week 1: Bit Operations
- [ ] Add opcodes to shader_vm.py
- [ ] Implement in shader_vm.wgsl
- [ ] Write tests for bit operations
- [ ] Update documentation

### Week 2: Time-Sliced Execution
- [ ] Modify webgpu_runtime.py for time slicing
- [ ] Add cycle budget to uniforms
- [ ] Test with long-running programs
- [ ] Measure performance

### Week 3: Ring Buffer I/O
- [ ] Implement RingBufferIO class
- [ ] Add keyboard event handling
- [ ] Integrate with Shader Terminal
- [ ] Test input latency

### Week 4: Font Rendering
- [ ] Create font bitmap data
- [ ] Implement character rendering with bit ops
- [ ] Full terminal text display
- [ ] DEMO: Working Shader Terminal!

## Performance Expectations (from Paper)

| Metric | Paper Results | Our Target |
|--------|---------------|------------|
| Clock Speed | 100s kHz - low MHz | 500 kHz - 1 MHz |
| Memory | 64MB (texture) | 128MB (buffer) |
| Boot Time | Minutes (Linux) | Instant (terminal) |
| Latency | 1-2 frames | 1-2 frames |

## Key Insights from Paper

1. **Warp Divergence is the Enemy**
   - Use hierarchical decode
   - Group similar instructions
   - Minimize branching in hot loops

2. **Memory is Not Byte-Addressable**
   - Must emulate with RMW operations
   - Use atomics for thread safety
   - Alignment is critical

3. **Can't Run Forever**
   - TDR timeout kills long shaders
   - Must use time slicing
   - Save/restore state every frame

4. **It Actually Works!**
   - Linux booted in VRChat
   - Hundreds of kHz achievable
   - Proven architecture

## References

Paper: "Architectural Patterns and Implementation Strategies for
High-Performance GPU-Resident Virtual Machines: The Pixel OS Paradigm in WebGPU"

Key Projects Mentioned:
- Linux in a Pixel Shader (pimaker)
- WASM-4 fantasy console
- RV32IMA emulator

## Next Action

Start with bit operations this week. This unblocks:
1. Font rendering (need bit shifts for bitmap fonts)
2. Byte-addressable memory (need for terminal buffer)
3. Character I/O (need for UART emulation)

Let's implement these extensions to shader_vm.py and shader_vm.wgsl!
