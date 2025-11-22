# Native Pixel OS Architecture
## Running Pixels Directly on Hardware

**Vision**: Build a launcher that loads pixel programs directly into CPU/GPU and executes them natively using pixels as the fundamental computational unit.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Pixel OS Launcher                        │
│  (Native C/C++ binary - boots and loads pixel programs)    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Pixel Runtime Engine                       │
│  - Memory Manager (GPU framebuffer as RAM)                 │
│  - Process Scheduler (GPU compute shaders)                 │
│  - Syscall Handler (CPU ↔ GPU communication)              │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
    ┌──────────────┐              ┌──────────────┐
    │  CPU Path    │              │  GPU Path    │
    │  - Syscalls  │              │  - Pixel Ops │
    │  - I/O       │              │  - Parallel  │
    │  - Scheduling│              │  - Shaders   │
    └──────────────┘              └──────────────┘
                            │
                            ▼
                ┌──────────────────────┐
                │  Hardware Execution  │
                │  - Framebuffer VRAM  │
                │  - GPU Compute Units │
                │  - Display Output    │
                └──────────────────────┘
```

---

## How It Works: GPU as Pixel Processor

### Core Concept
The GPU framebuffer becomes **pixel memory**. Each pixel is a memory cell. GPU compute shaders execute pixel operations in parallel.

### Memory Model
```
GPU Framebuffer (e.g., 1920x1080 = 2,073,600 pixels)
┌────────────────────────────────────────┐
│ Kernel Space (top 10%)                │  Read-only from user processes
├────────────────────────────────────────┤
│ Process 1 Memory Region               │  16x16 pixel pages
├────────────────────────────────────────┤
│ Process 2 Memory Region               │
├────────────────────────────────────────┤
│ Process N Memory Region               │
├────────────────────────────────────────┤
│ Free Memory Pool                      │
└────────────────────────────────────────┘
```

Each pixel (R, G, B) stores:
- **R channel**: Data/opcode
- **G channel**: Data/operand
- **B channel**: Data/flags
- **Alpha**: Reserved for metadata

### Execution Model

**GPU Compute Shader** (executes in parallel):
```glsl
#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba8, binding = 0) uniform image2D memory;

// Process scheduler runs every pixel in parallel
void main() {
    ivec2 pixel_coord = ivec2(gl_GlobalInvocationID.xy);
    vec4 pixel_value = imageLoad(memory, pixel_coord);

    // Decode instruction from RGB
    uint opcode = uint(pixel_value.r * 255.0);
    uint operand = uint(pixel_value.g * 255.0);
    uint flags = uint(pixel_value.b * 255.0);

    // Execute pixel instruction
    if (opcode == OPCODE_ADD) {
        // Parallel addition across all pixels
        vec4 result = pixel_value + imageLoad(memory, pixel_coord + ivec2(1, 0));
        imageStore(memory, pixel_coord, result);
    }
    // ... more opcodes
}
```

---

## Implementation Path

### Phase 1: Python Prototype (Weeks 1-4)
**Goal**: Prove the concept works in software

- Implement core components in Python
- Visualize using pygame/OpenGL
- Test scheduling, memory management, processes
- **Output**: `pixel_os_prototype.py` - working simulation

**Why**: Rapid iteration, easy debugging, validates design

### Phase 2: GPU Compute Backend (Weeks 5-8)
**Goal**: Move execution to GPU

#### 2.1 OpenGL/Vulkan Compute Shaders
- Translate Python logic to GLSL/SPIR-V shaders
- Implement pixel operations as shaders
- CPU-GPU communication via framebuffer mapping

#### 2.2 Frameworks to Consider
| Framework | Pros | Cons |
|-----------|------|------|
| **OpenGL Compute** | Widely supported, simpler | Less control |
| **Vulkan Compute** | Maximum performance, fine control | Complex API |
| **CUDA** | Best for NVIDIA, easy to use | NVIDIA-only |
| **WebGPU** | Cross-platform, future-proof | Newer, less mature |
| **OpenCL** | Cross-vendor | Declining support |

**Recommendation**: Start with **OpenGL Compute** (simplicity), migrate to **Vulkan** (performance).

#### 2.3 Memory Management
```c++
// Map GPU framebuffer to CPU-accessible memory
GLuint framebuffer_texture;
glGenTextures(1, &framebuffer_texture);
glBindTexture(GL_TEXTURE_2D, framebuffer_texture);
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1920, 1080, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

// Read/write pixel memory
glBindImageTexture(0, framebuffer_texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8);
```

### Phase 3: C++ Runtime Engine (Weeks 9-12)
**Goal**: Build native runtime in C++

#### 3.1 Core Components
```cpp
// kernel/pixel_runtime.hpp
class PixelRuntime {
public:
    void initialize();
    void load_program(const char* pixel_binary);
    void execute();

private:
    GPUMemoryManager memory_;      // Manages framebuffer
    PixelScheduler scheduler_;     // Dispatches compute shaders
    SyscallHandler syscalls_;      // CPU-side syscalls
    DisplayDriver display_;        // Output to screen
};
```

#### 3.2 Syscall Bridge
When pixel program needs I/O:
1. GPU shader writes syscall request to special pixel region
2. CPU monitors this region each frame
3. CPU executes syscall (read file, network, etc.)
4. CPU writes result back to pixel memory
5. GPU shader resumes

```cpp
// Syscall pixel format (special color range)
struct SyscallPixel {
    uint8_t syscall_number;  // R: which syscall (read/write/fork)
    uint8_t arg1;            // G: argument 1
    uint8_t arg2;            // B: argument 2
    uint8_t status;          // A: pending/complete
};
```

### Phase 4: Standalone Launcher (Weeks 13-16)
**Goal**: Create bootable launcher

#### 4.1 Launcher Binary
```cpp
// launcher/main.cpp
int main(int argc, char** argv) {
    // Initialize GPU
    if (!init_gpu_context()) {
        fprintf(stderr, "Failed to initialize GPU\n");
        return 1;
    }

    // Load pixel OS kernel
    PixelRuntime runtime;
    runtime.initialize();

    // Load init program (PID 1)
    runtime.load_program("init.pxl");

    // Enter execution loop
    runtime.execute();  // Never returns

    return 0;
}
```

#### 4.2 Pixel Program Format (.pxl)
```
Magic Number: 0x50584C00 ("PXL\0")
Header:
  - Width: 256 pixels
  - Height: 256 pixels
  - Entry point: (0, 0)
  - Version: 1
Data:
  - Raw pixel data (RGBA)
  - Embedded in PNG or custom binary format
```

#### 4.3 Build System
```bash
# Compile launcher
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make

# Output: pixel_os_launcher (native binary)
./pixel_os_launcher init.pxl
```

### Phase 5: Bootloader Integration (Weeks 17-20)
**Goal**: Boot directly from hardware (optional)

#### Option A: Extend pxOS Bootloader
Modify `pxos-v1.0` to:
1. Switch to protected mode (32-bit)
2. Initialize GPU via VESA/UEFI GOP
3. Load pixel OS launcher from disk
4. Jump to launcher

#### Option B: UEFI Application
Create UEFI app that:
1. Uses UEFI GOP (Graphics Output Protocol) for framebuffer
2. Loads pixel runtime
3. Boots into pixel OS

#### Option C: Linux/Windows Launcher
Run as userspace application:
- Easier development
- Still uses GPU compute
- Can access filesystems, network

**Recommendation**: Start with **Option C** (userspace), then **Option B** (UEFI) for standalone.

---

## Technical Details

### GPU Shader Instruction Set

```glsl
// Pixel opcodes (R channel values)
#define OPCODE_NOP    0x00
#define OPCODE_ADD    0x01  // Add two pixels
#define OPCODE_SUB    0x02  // Subtract
#define OPCODE_MUL    0x03  // Multiply
#define OPCODE_DIV    0x04  // Divide
#define OPCODE_LOAD   0x10  // Load from memory
#define OPCODE_STORE  0x11  // Store to memory
#define OPCODE_JMP    0x20  // Jump to pixel coordinate
#define OPCODE_JZ     0x21  // Jump if zero
#define OPCODE_CALL   0x22  // Function call
#define OPCODE_RET    0x23  // Return
#define OPCODE_SYSCALL 0xFF // System call (handled by CPU)
```

### Process Execution on GPU

Each process gets a **pixel region** and a **compute shader dispatch**:

```cpp
// Dispatch compute shader for process execution
void PixelScheduler::execute_process(PixelProcess* proc) {
    // Set shader uniforms
    glUniform2i(process_region_offset, proc->region.x, proc->region.y);
    glUniform2i(process_region_size, proc->region.width, proc->region.height);
    glUniform2i(program_counter, proc->pc.x, proc->pc.y);

    // Dispatch compute shader (16x16 work groups)
    glDispatchCompute(proc->region.width / 16, proc->region.height / 16, 1);

    // Memory barrier (ensure writes complete)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}
```

### Context Switching

GPU doesn't natively support context switching, so:
1. **Save state**: Copy process pixel region to system memory
2. **Load state**: Copy next process from system memory to GPU
3. **Resume**: Dispatch compute shader for new process

**Optimization**: Keep multiple processes in GPU memory, use shader uniforms to select active process.

### File System Persistence

**Filesystem lives in GPU memory during runtime**, persisted to disk on shutdown:

```cpp
void PixelFileSystem::save_to_disk(const char* path) {
    // Read framebuffer to CPU memory
    std::vector<uint8_t> pixel_data(1920 * 1080 * 4);
    glReadPixels(0, 0, 1920, 1080, GL_RGBA, GL_UNSIGNED_BYTE, pixel_data.data());

    // Save as PNG (lossless)
    stbi_write_png(path, 1920, 1080, 4, pixel_data.data(), 1920 * 4);
}

void PixelFileSystem::load_from_disk(const char* path) {
    // Load PNG to CPU memory
    int width, height, channels;
    uint8_t* pixel_data = stbi_load(path, &width, &height, &channels, 4);

    // Upload to GPU framebuffer
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixel_data);

    stbi_image_free(pixel_data);
}
```

---

## Development Stack

### Phase 1: Python Prototype
- **Language**: Python 3.10+
- **Graphics**: pygame + PyOpenGL
- **Math**: NumPy
- **Testing**: pytest

### Phase 2-4: Native Runtime
- **Language**: C++17
- **Graphics API**: OpenGL 4.3+ (compute shaders) or Vulkan 1.2+
- **Windowing**: GLFW or SDL2
- **Image I/O**: stb_image / stb_image_write
- **Build**: CMake 3.20+
- **Testing**: Google Test

### Phase 5: Bootloader (Optional)
- **Bootloader**: Extend pxOS (x86 assembly) or UEFI (C)
- **GPU Init**: VESA BIOS Extensions or UEFI GOP
- **Toolchain**: GCC/Clang with freestanding target

---

## Revised Roadmap

### Stage 1: Prototype (4 weeks)
| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1-2 | Python pixel grid + basic ops | Interactive pixel viewer |
| 3 | Memory manager + processes | Allocate/free pixel regions |
| 4 | Simple scheduler + instruction exec | Run 3 pixel programs concurrently |

**Output**: `pixel_os_prototype.py` proving concept works

### Stage 2: GPU Backend (4 weeks)
| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 5 | OpenGL compute shader setup | Framebuffer as memory |
| 6 | Port pixel operations to shaders | ADD/SUB/MUL on GPU |
| 7 | GPU scheduler | Multi-process execution |
| 8 | CPU-GPU syscall bridge | File I/O from pixel programs |

**Output**: Python runtime calling GPU shaders

### Stage 3: Native Runtime (4 weeks)
| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 9 | C++ runtime skeleton | Launcher loads .pxl files |
| 10 | Native memory manager | GPU memory allocation |
| 11 | Native scheduler + syscalls | Full process lifecycle |
| 12 | Filesystem + persistence | Save/load to PNG |

**Output**: `pixel_os_launcher` native binary

### Stage 4: Polish & Extend (4 weeks)
| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 13 | Shell + user programs | Interactive pixel shell |
| 14 | Development tools | Assembler, debugger |
| 15 | Performance optimization | 60 FPS with 100 processes |
| 16 | Documentation | User guide, API reference |

**Output**: Complete Pixel OS distribution

### Stage 5: Bootloader (4 weeks, optional)
| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 17-18 | UEFI application | Boot from USB |
| 19-20 | Hardware drivers | Real keyboard/mouse |

**Output**: Bootable Pixel OS ISO

---

## Key Design Decisions

### Why GPU Compute Shaders?
- **Parallelism**: Execute millions of pixels simultaneously
- **Performance**: 10-100x faster than CPU for pixel operations
- **Natural fit**: Pixels are already GPU's domain
- **Hardware access**: Direct framebuffer manipulation

### CPU vs GPU Division of Labor
| Component | Execution | Reason |
|-----------|-----------|--------|
| Pixel operations (ADD/SUB/etc.) | GPU | Massive parallelism |
| Process scheduling | GPU | Select active process |
| Memory allocation | CPU | Complex logic, rare operation |
| System calls (I/O) | CPU | OS interface, blocking ops |
| Filesystem | GPU (data) + CPU (logic) | Hybrid approach |
| Display output | GPU | Direct framebuffer |

### Memory Layout Strategy
```
1920x1080 framebuffer = 2,073,600 pixels = ~8 MB

Option 1: One large framebuffer (chosen)
- Single 1920x1080 texture
- Processes at different XY regions
- Simple, fast

Option 2: Multiple textures
- Separate texture per process
- More flexible, complex
```

### Persistence Format
- **Runtime**: GPU framebuffer (RGBA8)
- **Disk**: PNG (lossless, compressed)
- **Boot**: Load PNG → Upload to GPU

---

## Building & Running

### Prerequisites
```bash
# Ubuntu/Debian
sudo apt install build-essential cmake libglfw3-dev libglew-dev

# macOS
brew install cmake glfw glew

# Windows
# Install Visual Studio 2022 with C++ tools
# Download GLFW and GLEW binaries
```

### Build Native Launcher
```bash
cd pixel-os
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Output: bin/pixel_os_launcher
```

### Run Pixel OS
```bash
# Create a simple pixel program
echo "Creating hello world pixel program..."
python3 tools/pixel_assembler.py examples/hello.pxl.asm -o hello.pxl

# Launch pixel OS
./bin/pixel_os_launcher hello.pxl

# Should open window and execute pixel program
```

---

## Performance Targets

| Metric | Target | Strategy |
|--------|--------|----------|
| Boot time | < 100ms | Preload kernel to GPU |
| Context switch | < 1ms | GPU state already in VRAM |
| Process count | 1000+ | Limited by VRAM (8 MB = ~1000 processes) |
| Instructions/sec | 1B+ | GPU parallel execution |
| Frame rate | 60 FPS | VSync, efficient shaders |
| Memory bandwidth | GPU VRAM speed | 200+ GB/s on modern GPUs |

---

## Example: Hello World on Native Pixel OS

### Pixel Assembly (.pxl.asm)
```asm
; hello.pxl.asm
section .data
    msg: db "Hello, Pixel OS!", 0

section .text
    global _start

_start:
    ; System call: write(stdout, msg, len)
    LOAD R1, msg        ; R1 = pointer to message
    LOAD R2, 16         ; R2 = length
    SYSCALL SYS_WRITE   ; Invoke syscall

    ; System call: exit(0)
    LOAD R1, 0          ; Exit code 0
    SYSCALL SYS_EXIT    ; Terminate
```

### Compiled to Pixel Binary
```
Pixel (0, 0): [0x10, 0x00, 0x00]  ; LOAD opcode
Pixel (0, 1): [0x48, 0x65, 0x6C]  ; "Hel"
Pixel (0, 2): [0x6C, 0x6F, 0x2C]  ; "lo,"
...
Pixel (5, 0): [0xFF, 0x02, 0x00]  ; SYSCALL WRITE
```

### Executed on GPU
1. Launcher loads `hello.pxl` to GPU framebuffer
2. Compute shader starts at pixel (0, 0)
3. Shader decodes LOAD instructions
4. Shader encounters SYSCALL → writes to syscall region
5. CPU detects syscall, prints "Hello, Pixel OS!"
6. Shader encounters EXIT → terminates

---

## FAQ

**Q: Why not just use a VM/emulator?**
A: We want native GPU execution for performance and the unique experience of "pixels thinking."

**Q: Can I boot this on real hardware?**
A: Yes, via UEFI app (Stage 5). Requires GPU with compute shader support.

**Q: What about old GPUs without compute shaders?**
A: Fallback to CPU implementation (slower but works).

**Q: How do I debug pixel programs?**
A: Visual debugger shows current instruction pixel highlighted, step-through execution.

**Q: Can this run Linux binaries?**
A: No, but you could write a syscall translation layer (advanced).

**Q: What about security?**
A: Process isolation via separate pixel regions, GPU enforces boundaries.

---

## Next Steps

1. **Validate approach**: Build Phase 1 prototype in Python
2. **GPU proof-of-concept**: Single compute shader executing pixel addition
3. **Design instruction set**: Finalize opcode mapping
4. **Begin C++ runtime**: Core launcher skeleton

**Ready to start?** Let's build the Python prototype first (4 weeks), then migrate to native GPU execution.

Would you like me to begin with **Phase 1.1: PixelGrid in Python** or jump straight to **GPU compute shader proof-of-concept**?
