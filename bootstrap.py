#!/usr/bin/env python3
"""
bootstrap.py - The pxOS CPU
This file is FROZEN. It loads and executes an OS entirely from VRAM.
"""
import sys
import argparse
import time
import numpy as np
from PIL import Image

try:
    import wgpu
    from rendercanvas.auto import RenderCanvas, loop
except (ImportError, RuntimeError):
    RenderCanvas, loop = None, None

VRAM_WIDTH, VRAM_HEIGHT = 800, 600

# Hardcoded constants for the CPU architecture
OP_HALT = 0x00
OP_PSET = 0x01

BLIT_SHADER_CODE = """
@group(0) @binding(0) var screen_sampler: sampler;
@group(0) @binding(1) var screen_texture: texture_2d<f32>;
struct VertexOutput { @builtin(position) pos: vec4<f32>, @location(0) tex_coord: vec2<f32> };
@vertex fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((idx << 1u) & 2u) - 1.0;
    let y = f32(idx & 2u) - 1.0;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.tex_coord = vec2<f32>(x * 0.5 + 0.5, -y * 0.5 + 0.5);
    return out;
}
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(screen_texture, screen_sampler, in.tex_coord);
}
"""

class pxOS_CPU:
    def __init__(self, headless=False, vram_file=None):
        self.vram = np.zeros((VRAM_HEIGHT, VRAM_WIDTH, 4), dtype=np.uint8)
        self.pc = 0  # Program Counter, points to a pixel index in VRAM
        self.running = True
        self.headless = headless

        self.canvas = None
        if not self.headless:
            if RenderCanvas is None: raise RuntimeError("GUI mode needs rendercanvas.")
            self.canvas = RenderCanvas(size=(VRAM_WIDTH, VRAM_HEIGHT), title="pxOS CPU")

        self.setup_gpu()

        if vram_file:
            self.load_vram_from_file(vram_file)

    def setup_gpu(self):
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = adapter.request_device_sync()
        self.gpu_texture = self.device.create_texture(
            size=(VRAM_WIDTH, VRAM_HEIGHT, 1),
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.COPY_SRC,
            format=wgpu.TextureFormat.rgba8unorm
        )
        if not self.headless: self.setup_gui_pipeline()

    def setup_gui_pipeline(self):
        shader = self.device.create_shader_module(code=BLIT_SHADER_CODE)
        context = self.canvas.get_context("wgpu")
        tex_format = context.get_preferred_format(self.device.adapter)
        context.configure(device=self.device, format=tex_format)

        bind_group_layout = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {}},
        ])
        pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])

        self.render_pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={"module": shader, "entry_point": "vs_main"},
            fragment={"module": shader, "entry_point": "fs_main", "targets": [{"format": tex_format}]},
        )
        sampler = self.device.create_sampler()
        self.bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {"binding": 0, "resource": sampler},
                {"binding": 1, "resource": self.gpu_texture.create_view()},
            ],
        )

    def load_vram_from_file(self, filename):
        try:
            with open(filename, 'rb') as f:
                data = f.read()
            loaded_vram = np.frombuffer(data, dtype=np.uint8).reshape(self.vram.shape).copy()
            self.vram = loaded_vram
            print(f"Loaded VRAM from {filename}")
        except Exception as e:
            print(f"Failed to load VRAM from {filename}: {e}")
            self.running = False

    def read_vram_pixel(self, index):
        y = index // VRAM_WIDTH
        x = index % VRAM_WIDTH
        return self.vram[y, x]

    def write_vram_pixel(self, index, value):
        y = index // VRAM_WIDTH
        x = index % VRAM_WIDTH
        self.vram[y, x] = value

    def fetch_decode_execute(self):
        if not self.running or self.pc >= VRAM_WIDTH * VRAM_HEIGHT:
            self.running = False
            return

        instruction_pixel = self.read_vram_pixel(self.pc)
        opcode = instruction_pixel[0] # R channel is opcode

        if opcode == OP_HALT:
            print(f"PC={self.pc}: HALT")
            self.running = False
            self.pc += 1
        elif opcode == OP_PSET:
            # PSET reads the next 2 pixels for args
            arg1_pixel = self.read_vram_pixel(self.pc + 1)
            arg2_pixel = self.read_vram_pixel(self.pc + 2)

            x = (arg1_pixel[0] << 8) | arg1_pixel[1]
            y = (arg1_pixel[2] << 8) | arg1_pixel[3]
            color = arg2_pixel # RGBA

            if 0 <= y < VRAM_HEIGHT and 0 <= x < VRAM_WIDTH:
                self.vram[y, x] = color

            print(f"PC={self.pc}: PSET ({x}, {y}) to color {color.tolist()}")
            self.pc += 3
        else:
            print(f"PC={self.pc}: UNKNOWN_OPCODE {opcode}")
            self.pc += 1

    def run(self, max_cycles=None):
        print("="*50)
        print("pxOS CPU Execution Start")
        print("="*50)

        cycle_count = 0
        while self.running:
            self.fetch_decode_execute()
            if not self.headless:
                self.canvas.request_draw()

            if max_cycles and cycle_count >= max_cycles:
                print("Max cycles reached. Halting.")
                self.running = False
            cycle_count += 1
            time.sleep(0.01) # Slow down for visualization

        print("pxOS CPU Halted.")

    def render_and_save(self, output_filename):
        self.device.queue.write_texture(
            {"texture": self.gpu_texture}, self.vram,
            {"bytes_per_row": VRAM_WIDTH * 4, "rows_per_image": VRAM_HEIGHT},
            (VRAM_WIDTH, VRAM_HEIGHT, 1)
        )
        bytes_per_row = VRAM_WIDTH * 4
        if bytes_per_row % 256 != 0: bytes_per_row += 256 - (bytes_per_row % 256)

        buffer = self.device.create_buffer(size=bytes_per_row * VRAM_HEIGHT, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        encoder = self.device.create_command_encoder()
        encoder.copy_texture_to_buffer(
            {"texture": self.gpu_texture}, {"buffer": buffer, "bytes_per_row": bytes_per_row}, (VRAM_WIDTH, VRAM_HEIGHT, 1)
        )
        self.device.queue.submit([encoder.finish()])

        buffer.map_sync(wgpu.MapMode.READ)
        data = buffer.read_mapped()
        buffer.unmap()

        Image.frombytes("RGBA", (VRAM_WIDTH, VRAM_HEIGHT), data, "raw", "RGBA", bytes_per_row).save(output_filename)
        print(f"Rendered VRAM state to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pxOS Bootstrap CPU")
    parser.add_argument('vram_file', nargs='?', default=None, help="Path to a .vram file to load at boot.")
    parser.add_argument('--output', type=str, default=None, help="Enable headless mode and save the final VRAM state to this file.")
    parser.add_argument('--cycles', type=int, default=100, help="Max number of CPU cycles to run in headless mode.")
    args = parser.parse_args()

    cpu = pxOS_CPU(headless=args.output is not None, vram_file=args.vram_file)

    if not args.output:
        # GUI mode
        def gui_run_loop():
            if cpu.running:
                cpu.fetch_decode_execute()
                loop.call_later(0.1, gui_run_loop)
            else:
                print("CPU Halted. Close window to exit.")

        # This part is tricky because the render loop is separate.
        # A proper GUI would integrate the CPU loop into the render loop.
        # For now, let's just run for a bit then show.
        def render_frame_gui():
             cpu.device.queue.write_texture({"texture": cpu.gpu_texture}, cpu.vram, {"bytes_per_row": VRAM_WIDTH * 4}, (VRAM_WIDTH, VRAM_HEIGHT, 1))
             current_texture_view = cpu.canvas.get_current_texture()
             command_encoder = cpu.device.create_command_encoder()
             render_pass = command_encoder.begin_render_pass(color_attachments=[{"view": current_texture_view, "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}])
             render_pass.set_pipeline(cpu.render_pipeline)
             render_pass.set_bind_group(0, cpu.bind_group, [], 0, 999999)
             render_pass.draw(3, 1, 0, 0)
             render_pass.end()
             cpu.device.queue.submit([command_encoder.finish()])

        cpu.canvas.request_draw(render_frame_gui)
        loop.call_soon(gui_run_loop)
        loop.run()
    else:
        # Headless mode
        cpu.run(max_cycles=args.cycles)
        cpu.render_and_save(args.output)
