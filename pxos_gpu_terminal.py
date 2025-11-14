import os
import numpy as np
import wgpu
from wgpu.gui.auto import WgpuCanvas, run
from pxos_text import BitmapFont, TextRenderer
from PIL import Image

class Layer:
    def __init__(self, name, width, height, z_index=0):
        self.name = name
        self.buffer = np.zeros((height, width, 4), dtype=np.uint8)
        self.visible = True
        self.z_index = z_index

class Console:
    def __init__(self, screen_width, screen_height, font_height, x=8, y=None, rows=10, bg=(0,0,0,200), fg=(255,255,255,255)):
        self.x = x
        self.y = y if y is not None else screen_height - (rows * (font_height + 1) + 8)
        self.rows = rows
        self.bg = bg
        self.fg = fg
        self.lines = []
        self.width = screen_width - 2 * x
        self.height = rows * (font_height + 1) + 4

    def write(self, text: str):
        for line in text.split("\n"):
            self.lines.append(line)
        while len(self.lines) > self.rows:
            self.lines.pop(0)

    def render(self, buf, text_renderer):
        h, w, _ = buf.shape
        x0, y0 = self.x, self.y
        x1, y1 = min(w, x0 + self.width), min(h, y0 + self.height)

        buf[y0:y1, x0:x1] = self.bg

        line_y = y0 + 2
        for line in self.lines[-self.rows:]:
            text_renderer.draw_text(buf, x0 + 2, line_y, line, self.fg)
            line_y += text_renderer.font.height + 1

class PxOSTerminalGPU:
    def __init__(self, width=800, height=600, offscreen=False, imperfect=True):
        self.width = width
        self.height = height
        self.imperfect = imperfect
        self.is_offscreen = offscreen or 'DISPLAY' not in os.environ

        if self.is_offscreen:
            self.canvas = None
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            self.device = adapter.request_device_sync()
            self.target_format = wgpu.TextureFormat.rgba8unorm
        else:
            self.canvas = WgpuCanvas(title="pxOS GPU Terminal")
            adapter = wgpu.gpu.request_adapter_sync(canvas=self.canvas, power_preference="high-performance")
            self.device = adapter.request_device_sync()
            self.target_format = self.canvas.get_preferred_format(adapter)

        self.layers = {}
        self.active_layer_name = None
        self.new_layer("default")
        self.use_layer("default")

        self.font = BitmapFont()
        self.text_renderer = TextRenderer(self.font)

        console_layer = Layer("console", self.width, self.height)
        console_layer.z_index = 100
        self.layers["console"] = console_layer
        self.console = Console(self.width, self.height, self.font.height)

        self.vram_tex = self.device.create_texture(
            size=(self.width, self.height, 1),
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension="2d",
            format="rgba8unorm",
            mip_level_count=1,
            sample_count=1,
        )

        with open("frozen_display.wgsl", "r") as f:
            shader_code = f.read()
        shader = self.device.create_shader_module(code=shader_code)

        self.sampler = self.device.create_sampler()

        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {}},
                {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {}},
            ]
        )

        self.pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[self.bind_group_layout])

        self.render_pipeline = self.device.create_render_pipeline(
            layout=self.pipeline_layout,
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [],
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.target_format}],
            },
            primitive={"topology": "triangle-strip"},
        )

        self.bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {"binding": 0, "resource": self.sampler},
                {"binding": 1, "resource": self.vram_tex.create_view()},
            ],
        )

    def new_layer(self, name, z_index=0):
        if name in self.layers:
            if self.imperfect:
                self.print_line(f"[warn] Layer '{name}' already exists.")
                return
            else:
                raise ValueError(f"Layer with name '{name}' already exists.")
        self.layers[name] = Layer(name, self.width, self.height, z_index)

    def use_layer(self, name):
        if name not in self.layers:
            if self.imperfect:
                self.print_line(f"[warn] Layer '{name}' does not exist.")
                return
            else:
                raise ValueError(f"Layer with name '{name}' does not exist.")
        self.active_layer_name = name

    def get_active_layer(self):
        if not self.active_layer_name or self.active_layer_name not in self.layers:
            if self.imperfect:
                self.print_line("[warn] No active layer selected, using 'default'.")
                self.use_layer("default")
            else:
                raise Exception("No active layer selected.")
        return self.layers[self.active_layer_name]

    def clear(self, r, g, b, a=255):
        layer = self.get_active_layer()
        layer.buffer[:, :, 0] = r
        layer.buffer[:, :, 1] = g
        layer.buffer[:, :, 2] = b
        layer.buffer[:, :, 3] = a

    def pixel(self, x, y, r, g, b, a=255):
        layer = self.get_active_layer()
        if 0 <= x < self.width and 0 <= y < self.height:
            layer.buffer[y, x] = (r, g, b, a)

    def hline(self, x, y, length, r, g, b, a=255):
        layer = self.get_active_layer()
        if 0 <= y < self.height:
            x_start = max(0, x)
            x_end = min(self.width, x + length)
            if x_start < x_end:
                layer.buffer[y, x_start:x_end] = (r, g, b, a)

    def rect(self, x, y, w, h, r, g, b, a=255):
        layer = self.get_active_layer()
        x_start, y_start = max(0, x), max(0, y)
        x_end, y_end = min(self.width, x + w), min(self.height, y + h)
        if x_start < x_end and y_start < y_end:
            layer.buffer[y_start:y_end, x_start:x_end] = (r, g, b, a)

    def text(self, x, y, text, r, g, b, a=255):
        layer = self.get_active_layer()
        self.text_renderer.draw_text(layer.buffer, x, y, text, (r, g, b, a))

    def upload_buffer(self, buffer):
        unpadded_bytes_per_row = self.width * 4
        padded_bytes_per_row = (unpadded_bytes_per_row + 255) & ~255

        if padded_bytes_per_row == unpadded_bytes_per_row:
            data = buffer.tobytes()
        else:
            padded_buffer = np.zeros((self.height, padded_bytes_per_row), dtype=np.uint8)
            unpadded_view = buffer.view(np.uint8).reshape(self.height, unpadded_bytes_per_row)
            padded_buffer[:, :unpadded_bytes_per_row] = unpadded_view
            data = padded_buffer.tobytes()

        self.device.queue.write_texture(
            {"texture": self.vram_tex, "mip_level": 0, "origin": (0, 0, 0)},
            data,
            {"bytes_per_row": padded_bytes_per_row, "rows_per_image": self.height},
            (self.width, self.height, 1),
        )

    def draw_frame(self):
        try:
            self._render_console()
            composite_buffer = self._compose_layers()
            self.upload_buffer(composite_buffer)
            self._present_frame(composite_buffer)
        except Exception as e:
            if self.imperfect:
                self.print_line(f"[frame error] {e}")
            else:
                raise

    def _present_frame(self, composite):
        try:
            if not self.is_offscreen:
                texture_view = self.canvas.get_current_texture().create_view()
                command_encoder = self.device.create_command_encoder()
                render_pass = command_encoder.begin_render_pass(
                    color_attachments=[{"view": texture_view, "load_op": "clear", "store_op": "store"}]
                )
                render_pass.set_pipeline(self.render_pipeline)
                render_pass.set_bind_group(0, self.bind_group, [], 0, 99)
                render_pass.draw(4, 1, 0, 0)
                render_pass.end()
                self.device.queue.submit([command_encoder.finish()])
        except Exception as e:
            if self.imperfect:
                self.print_line(f"[gpu warn] {e}")
            else:
                raise

    def _compose_layers(self):
        composite_buffer = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        visible_layers = [layer for layer in self.layers.values() if layer.visible]
        sorted_layers = sorted(visible_layers, key=lambda l: l.z_index)

        for layer in sorted_layers:
            fg = layer.buffer.astype(np.float32)
            bg = composite_buffer.astype(np.float32)
            alpha = fg[:, :, 3:4] / 255.0
            bg[:, :, :3] = fg[:, :, :3] * alpha + bg[:, :, :3] * (1.0 - alpha)
            bg[:, :, 3:4] = (alpha + (bg[:,:,3:4]/255.0) * (1.0 - alpha)) * 255.0
            composite_buffer = bg.astype(np.uint8)
        return composite_buffer

    def save_frame(self, path: str):
        self._render_console()
        composite = self._compose_layers()
        img = Image.fromarray(composite, mode="RGBA")
        img.save(path)

    def print_line(self, text: str):
        self.console.write(text)

    def _render_console(self):
        buf = self.layers["console"].buffer
        self.console.render(buf, self.text_renderer)

if __name__ == "__main__":
    terminal = PxOSTerminalGPU()
    terminal.new_layer("background", z_index=-1)
    terminal.use_layer("background")
    terminal.clear(50, 50, 50)

    terminal.new_layer("foreground")
    terminal.use_layer("foreground")
    terminal.rect(100, 100, 200, 150, 255, 0, 0, 128)
    terminal.rect(200, 200, 200, 150, 0, 255, 0, 128)
    terminal.print_line("Test console message.")

    def animation_frame():
        terminal.draw_frame()
        terminal.canvas.request_draw(animation_frame)

    animation_frame()
    run()
