"""
pxOS GPU Terminal - Frozen Shader Architecture

Version: 0.1

Architecture:
- Frozen WGSL shader (never changes)
- CPU-side VRAM buffer (numpy array)
- Layer-based compositing
- All logic in Python (debuggable, LLM-friendly)
- GPU is just a "display wire"
"""

import numpy as np
import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import pathlib
from PIL import Image
from typing import Dict, Optional

# Display dimensions
WIDTH, HEIGHT = 800, 600


class Layer:
    """
    A single compositable layer.

    Each layer has its own RGBA buffer, z-index for ordering,
    opacity for blending, and visibility flag.
    """
    def __init__(self, name: str, width: int, height: int, z_index: int = 0, opacity: int = 255):
        self.name = name
        self.width = width
        self.height = height
        self.buffer = np.zeros((height, width, 4), dtype=np.uint8)
        self.z_index = z_index
        self.opacity = opacity  # 0-255
        self.visible = True

    def __repr__(self):
        return f"Layer({self.name}, z={self.z_index}, visible={self.visible}, opacity={self.opacity})"


class PxOSTerminalGPU:
    """
    GPU-accelerated terminal using the Frozen Shader Bus pattern.

    Key concepts:
    - Frozen WGSL shader that never changes
    - Multiple compositable layers
    - CPU-side numpy VRAM buffers
    - All rendering logic in Python
    - Simple drawing API (CLEAR, PIXEL, RECT, HLINE, VLINE)
    """

    def __init__(self, width: int = WIDTH, height: int = HEIGHT):
        self.width = width
        self.height = height

        # Layer management
        self.layers: Dict[str, Layer] = {}
        self.current_layer_name = "default"
        self._create_default_layer()

        # WebGPU initialization
        self.canvas = WgpuCanvas(title="pxOS GPU Terminal", size=(width, height))
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = self.adapter.request_device_sync()

        # Create VRAM texture (what the shader samples)
        self.vram_tex = self.device.create_texture(
            size=(width, height, 1),
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.RENDER_ATTACHMENT,
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba8unorm,
            mip_level_count=1,
            sample_count=1,
        )

        # Create sampler
        self.sampler = self.device.create_sampler(
            mag_filter=wgpu.FilterMode.nearest,
            min_filter=wgpu.FilterMode.nearest,
        )

        # Load frozen shader
        shader_path = pathlib.Path(__file__).parent / "shaders" / "frozen_display.wgsl"
        with open(shader_path, "r") as f:
            shader_code = f.read()

        shader_module = self.device.create_shader_module(code=shader_code)

        # Create bind group layout
        bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                    },
                },
            ]
        )

        # Create bind group
        self.bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {"binding": 0, "resource": self.sampler},
                {"binding": 1, "resource": self.vram_tex.create_view()},
            ],
        )

        # Create pipeline layout
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )

        # Create render pipeline
        self.pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [
                    {
                        "array_stride": 4 * 4,  # 4 floats per vertex (x,y, u,v)
                        "step_mode": wgpu.VertexStepMode.vertex,
                        "attributes": [
                            {"format": wgpu.VertexFormat.float32x2, "offset": 0, "shader_location": 0},  # position
                            {"format": wgpu.VertexFormat.float32x2, "offset": 8, "shader_location": 1},  # uv
                        ],
                    }
                ],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [{"format": self.canvas.get_preferred_format(self.adapter)}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )

        # Full-screen quad vertices (x, y, u, v)
        vertices = np.array([
            # Triangle 1
            -1.0, -1.0, 0.0, 1.0,  # bottom-left
             1.0, -1.0, 1.0, 1.0,  # bottom-right
            -1.0,  1.0, 0.0, 0.0,  # top-left
            # Triangle 2
            -1.0,  1.0, 0.0, 0.0,  # top-left
             1.0, -1.0, 1.0, 1.0,  # bottom-right
             1.0,  1.0, 1.0, 0.0,  # top-right
        ], dtype=np.float32)

        self.vertex_buffer = self.device.create_buffer_with_data(
            data=vertices,
            usage=wgpu.BufferUsage.VERTEX,
        )

        # Register draw callback
        self.canvas.request_draw(self._draw_frame)

        print(f"[GPU Terminal] Initialized {width}x{height}")
        print(f"[GPU Terminal] Frozen shader loaded: {shader_path}")
        print(f"[GPU Terminal] Current layer: {self.current_layer_name}")

    def _create_default_layer(self):
        """Create the default background layer"""
        default = Layer("default", self.width, self.height, z_index=0)
        self.layers["default"] = default
        self.current_layer_name = "default"

    @property
    def current_layer(self) -> Layer:
        """Get the currently active layer"""
        return self.layers[self.current_layer_name]

    def _compose_layers(self) -> np.ndarray:
        """
        Compose all visible layers into a single RGBA buffer.

        Layers are sorted by z-index (low to high) and alpha-blended.
        This is where the "Photoshop-style" compositing happens.
        """
        # Start with transparent black
        composite = np.zeros((self.height, self.width, 4), dtype=np.float32)

        # Sort layers by z-index
        sorted_layers = sorted(
            [l for l in self.layers.values() if l.visible],
            key=lambda l: l.z_index
        )

        # Composite each layer
        for layer in sorted_layers:
            # Normalize to 0-1 range
            layer_f = layer.buffer.astype(np.float32) / 255.0
            opacity_f = layer.opacity / 255.0

            # Extract alpha channel
            alpha = layer_f[:, :, 3:4] * opacity_f

            # Alpha blend: result = src * alpha + dst * (1 - alpha)
            composite[:, :, :3] = layer_f[:, :, :3] * alpha + composite[:, :, :3] * (1 - alpha)
            composite[:, :, 3:4] = alpha + composite[:, :, 3:4] * (1 - alpha)

        # Convert back to uint8
        return (composite * 255).astype(np.uint8)

    def _upload_vram(self):
        """Upload composed VRAM to GPU texture"""
        composite = self._compose_layers()
        data = composite.tobytes()

        self.device.queue.write_texture(
            {
                "texture": self.vram_tex,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            data,
            {
                "bytes_per_row": self.width * 4,
                "rows_per_image": self.height,
            },
            (self.width, self.height, 1),
        )

    def _draw_frame(self):
        """Main render loop - compose layers and draw to screen"""
        # Upload composed VRAM to GPU
        self._upload_vram()

        # Get current texture
        current_texture = self.canvas.get_context().get_current_texture()
        command_encoder = self.device.create_command_encoder()

        # Render pass
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": current_texture.create_view(),
                    "resolve_target": None,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0, 0, 0, 1),
                }
            ],
        )

        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group, [], 0, 99)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(6, 1, 0, 0)
        render_pass.end()

        self.device.queue.submit([command_encoder.finish()])
        self.canvas.request_draw(self._draw_frame)

    # ========== Layer Management API ==========

    def layer_new(self, name: str, z_index: int = 0, opacity: int = 255):
        """Create a new layer"""
        if name in self.layers:
            print(f"[LAYER] Layer '{name}' already exists")
            return

        layer = Layer(name, self.width, self.height, z_index, opacity)
        self.layers[name] = layer
        print(f"[LAYER] Created layer '{name}' at z={z_index}")

    def layer_use(self, name: str):
        """Switch to a different layer for drawing"""
        if name not in self.layers:
            print(f"[LAYER] Layer '{name}' does not exist")
            return

        self.current_layer_name = name
        print(f"[LAYER] Now using layer '{name}'")

    def layer_delete(self, name: str):
        """Delete a layer"""
        if name == "default":
            print(f"[LAYER] Cannot delete default layer")
            return

        if name not in self.layers:
            print(f"[LAYER] Layer '{name}' does not exist")
            return

        if self.current_layer_name == name:
            self.current_layer_name = "default"

        del self.layers[name]
        print(f"[LAYER] Deleted layer '{name}'")

    def layer_list(self):
        """List all layers"""
        sorted_layers = sorted(self.layers.values(), key=lambda l: l.z_index)
        for layer in sorted_layers:
            current = "*" if layer.name == self.current_layer_name else " "
            print(f"{current} {layer}")

    # ========== Drawing API (operates on current layer) ==========

    def cmd_clear(self, r: int, g: int, b: int, a: int = 255):
        """
        CLEAR r g b [a]

        Fill the current layer with a solid color.
        """
        buf = self.current_layer.buffer
        buf[:, :, 0] = r
        buf[:, :, 1] = g
        buf[:, :, 2] = b
        buf[:, :, 3] = a
        print(f"[CMD] CLEAR ({r}, {g}, {b}, {a}) on layer '{self.current_layer_name}'")

    def cmd_pixel(self, x: int, y: int, r: int, g: int, b: int, a: int = 255):
        """
        PIXEL x y r g b [a]

        Set a single pixel in the current layer.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            buf = self.current_layer.buffer
            buf[y, x, 0] = r
            buf[y, x, 1] = g
            buf[y, x, 2] = b
            buf[y, x, 3] = a
        else:
            print(f"[CMD] PIXEL out of bounds: ({x}, {y})")

    def cmd_rect(self, x: int, y: int, w: int, h: int, r: int, g: int, b: int, a: int = 255):
        """
        RECT x y w h r g b [a]

        Draw a filled rectangle in the current layer.
        """
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + w)
        y1 = min(self.height, y + h)

        if x0 < x1 and y0 < y1:
            buf = self.current_layer.buffer
            buf[y0:y1, x0:x1, 0] = r
            buf[y0:y1, x0:x1, 1] = g
            buf[y0:y1, x0:x1, 2] = b
            buf[y0:y1, x0:x1, 3] = a
        else:
            print(f"[CMD] RECT out of bounds or invalid size")

    def cmd_hline(self, x: int, y: int, length: int, r: int, g: int, b: int, a: int = 255):
        """
        HLINE x y length r g b [a]

        Draw a horizontal line in the current layer.
        """
        if not (0 <= y < self.height):
            print(f"[CMD] HLINE out of bounds: y={y}")
            return

        x_end = min(self.width, max(0, x + length))
        x_start = max(0, x)

        if x_start >= x_end:
            print(f"[CMD] HLINE invalid range")
            return

        buf = self.current_layer.buffer
        buf[y, x_start:x_end, 0] = r
        buf[y, x_start:x_end, 1] = g
        buf[y, x_start:x_end, 2] = b
        buf[y, x_start:x_end, 3] = a

    def cmd_vline(self, x: int, y: int, length: int, r: int, g: int, b: int, a: int = 255):
        """
        VLINE x y length r g b [a]

        Draw a vertical line in the current layer.
        """
        if not (0 <= x < self.width):
            print(f"[CMD] VLINE out of bounds: x={x}")
            return

        y_end = min(self.height, max(0, y + length))
        y_start = max(0, y)

        if y_start >= y_end:
            print(f"[CMD] VLINE invalid range")
            return

        buf = self.current_layer.buffer
        buf[y_start:y_end, x, 0] = r
        buf[y_start:y_end, x, 1] = g
        buf[y_start:y_end, x, 2] = b
        buf[y_start:y_end, x, 3] = a

    # ========== Utility API ==========

    def save_frame(self, path: str):
        """
        SAVE path

        Save the current composed frame to a PNG file.
        """
        composite = self._compose_layers()
        img = Image.fromarray(composite, mode="RGBA")
        img.save(path)
        print(f"[SAVE] Saved frame to {path}")

    def info(self):
        """Print terminal info"""
        print(f"[INFO] Canvas: {self.width}x{self.height}")
        print(f"[INFO] Current layer: {self.current_layer_name}")
        print(f"[INFO] Total layers: {len(self.layers)}")

    def run(self):
        """Start the event loop"""
        print("[GPU Terminal] Starting event loop...")
        run()


# Simple demo if run directly
if __name__ == "__main__":
    terminal = PxOSTerminalGPU()

    # Draw a simple demo scene
    terminal.cmd_clear(0, 0, 64)  # Dark blue background
    terminal.cmd_rect(100, 100, 200, 150, 255, 0, 0, 200)  # Red rectangle
    terminal.cmd_hline(0, 300, 800, 255, 255, 255)  # White line

    terminal.run()
