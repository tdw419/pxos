"""
WebGPU Runtime - Executes shader VM bytecode on the GPU

This module handles:
- WebGPU initialization
- Shader compilation and pipeline creation
- Buffer management for bytecode and uniforms
- Render loop with hot-reloading support
"""

import time
import struct
from pathlib import Path
from typing import Optional, Tuple

try:
    import wgpu
    from wgpu.gui.auto import WgpuCanvas, run
    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    print("⚠️  wgpu-py not available. Install with: pip install wgpu")

from shader_vm import ShaderVM, EffectCompiler


class ShaderVMRuntime:
    """
    WebGPU runtime for executing shader VM bytecode
    """

    def __init__(self, width: int = 800, height: int = 600):
        if not WGPU_AVAILABLE:
            raise RuntimeError("wgpu-py is required for WebGPU runtime")

        self.width = width
        self.height = height
        self.start_time = time.time()

        # WebGPU resources
        self.device: Optional[wgpu.GPUDevice] = None
        self.canvas: Optional[WgpuCanvas] = None
        self.pipeline: Optional[wgpu.GPUComputePipeline] = None
        self.bind_group: Optional[wgpu.GPUBindGroup] = None

        # Buffers
        self.uniform_buffer: Optional[wgpu.GPUBuffer] = None
        self.bytecode_buffer: Optional[wgpu.GPUBuffer] = None
        self.output_texture: Optional[wgpu.GPUTexture] = None

        # Current bytecode
        self.current_bytecode: Optional[bytes] = None
        self.current_effect: str = "plasma"

    def initialize(self):
        """Initialize WebGPU device and resources"""
        # Create canvas
        self.canvas = WgpuCanvas(
            size=(self.width, self.height),
            title="pxOS Shader VM Runtime",
        )

        # Get adapter and device
        adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
        self.device = adapter.request_device()

        # Load shader code
        shader_path = Path(__file__).parent / "shader_vm.wgsl"
        shader_code = shader_path.read_text()

        # Create shader module
        shader_module = self.device.create_shader_module(code=shader_code)

        # Create output texture
        self.output_texture = self.device.create_texture(
            size=(self.width, self.height, 1),
            format="rgba8unorm",
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC,
        )

        # Create uniform buffer (time, resolution, mouse, padding)
        self.uniform_buffer = self.device.create_buffer(
            size=32,  # 8 floats * 4 bytes
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # Create bytecode buffer (initially empty, will be updated)
        self.bytecode_buffer = self.device.create_buffer(
            size=4096 * 4,  # 4096 uint32s
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        # Create bind group layout
        bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": wgpu.BufferBindingType.read_only_storage,
                    },
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "storage_texture": {
                        "access": wgpu.StorageTextureAccess.write_only,
                        "format": "rgba8unorm",
                        "view_dimension": wgpu.TextureViewDimension.d2,
                    },
                },
            ]
        )

        # Create bind group
        self.bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self.uniform_buffer}},
                {"binding": 1, "resource": {"buffer": self.bytecode_buffer}},
                {
                    "binding": 2,
                    "resource": self.output_texture.create_view(),
                },
            ],
        )

        # Create pipeline
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )

        self.pipeline = self.device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader_module, "entry_point": "main"},
        )

        # Set up canvas draw handler
        self.canvas.request_draw(self.draw_frame)

        print("✅ WebGPU runtime initialized")
        print(f"   Device: {adapter.request_adapter_info()['description']}")
        print(f"   Resolution: {self.width}x{self.height}")

    def load_bytecode(self, vm: ShaderVM):
        """Load compiled bytecode into GPU buffer"""
        bytecode_array = vm.compile_to_uint32_array()

        # Pad to buffer size if needed
        buffer_size = 4096
        if len(bytecode_array) > buffer_size:
            raise ValueError(f"Bytecode too large: {len(bytecode_array)} > {buffer_size}")

        padded = bytecode_array + [0] * (buffer_size - len(bytecode_array))

        # Convert to bytes
        bytecode_bytes = struct.pack(f'{len(padded)}I', *padded)

        # Upload to GPU
        self.device.queue.write_buffer(
            self.bytecode_buffer,
            0,
            bytecode_bytes,
        )

        self.current_bytecode = bytecode_bytes

        print(f"📦 Loaded bytecode: {len(bytecode_array)} instructions ({len(bytecode_array)*4} bytes)")

    def update_uniforms(self, mouse_pos: Tuple[float, float] = (0.0, 0.0)):
        """Update uniform buffer with current time and mouse position"""
        current_time = time.time() - self.start_time

        # Pack uniforms: time, resolution.x, resolution.y, mouse.x, mouse.y, padding...
        uniform_data = struct.pack(
            'ffffffff',
            current_time,
            float(self.width),
            float(self.height),
            0.0,  # padding
            mouse_pos[0],
            mouse_pos[1],
            0.0,  # padding
            0.0,  # padding
        )

        self.device.queue.write_buffer(
            self.uniform_buffer,
            0,
            uniform_data,
        )

    def draw_frame(self):
        """Render a single frame"""
        # Update uniforms
        self.update_uniforms()

        # Create command encoder
        command_encoder = self.device.create_command_encoder()

        # Dispatch compute shader
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, self.bind_group)

        # Dispatch workgroups (8x8 workgroup size)
        workgroups_x = (self.width + 7) // 8
        workgroups_y = (self.height + 7) // 8
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1)
        compute_pass.end()

        # Copy texture to canvas
        texture_view = self.output_texture.create_view()
        canvas_texture = self.canvas.get_context().get_current_texture()

        command_encoder.copy_texture_to_texture(
            {"texture": self.output_texture},
            {"texture": canvas_texture},
            (self.width, self.height, 1),
        )

        # Submit commands
        self.device.queue.submit([command_encoder.finish()])

        # Request next frame
        self.canvas.request_draw(self.draw_frame)

    def run(self):
        """Start the render loop"""
        if not self.current_bytecode:
            print("⚠️  No bytecode loaded. Use load_bytecode() first.")
            return

        print("\n🚀 Starting shader VM runtime...")
        print("   Press Ctrl+C to stop")
        run()

    def hot_reload(self, vm: ShaderVM):
        """
        Hot-reload new bytecode without restarting the runtime

        This is the magic of the VM architecture - we can change
        the program while it's running!
        """
        self.load_bytecode(vm)
        print("🔥 Hot-reloaded bytecode")


def demo_runtime():
    """Demo the shader VM runtime with multiple effects"""
    print("🎨 pxOS Shader VM Runtime Demo")
    print("=" * 60)

    # Create runtime
    runtime = ShaderVMRuntime(width=800, height=600)
    runtime.initialize()

    # Compile effects
    compiler = EffectCompiler()

    # Start with plasma effect
    plasma_vm = compiler.compile_plasma()
    print("\n📋 Plasma Effect Disassembly:")
    print(plasma_vm.disassemble())

    runtime.load_bytecode(plasma_vm)

    # TODO: Add keyboard handlers to switch effects
    # For now, just run plasma
    runtime.run()


def test_all_effects():
    """Test compilation of all effects without rendering"""
    print("🧪 Testing Effect Compilation")
    print("=" * 60)

    compiler = EffectCompiler()

    effects = {
        "Plasma": compiler.compile_plasma(),
        "Gradient": compiler.compile_gradient(),
        "Pulsing Circle": compiler.compile_pulsing_circle(),
        "Rainbow Spiral": compiler.compile_rainbow_spiral(),
    }

    for name, vm in effects.items():
        print(f"\n{name}:")
        bytecode = vm.compile_to_uint32_array()
        print(f"  ✓ {len(vm.instructions)} instructions")
        print(f"  ✓ {len(bytecode)} uint32s ({len(bytecode)*4} bytes)")

        # Show first few instructions
        lines = vm.disassemble().split('\n')[:6]
        for line in lines:
            print(f"    {line}")

    print("\n✅ All effects compiled successfully!")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        test_all_effects()
    else:
        if WGPU_AVAILABLE:
            demo_runtime()
        else:
            print("\n⚠️  wgpu-py not available")
            print("   Install with: pip install wgpu glfw")
            print("   Running compilation test instead...\n")
            test_all_effects()
