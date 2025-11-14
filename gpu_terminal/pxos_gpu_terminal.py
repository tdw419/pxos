"""
pxOS GPU Terminal - Frozen Shader Architecture
Version: 0.1

Architecture:
- Frozen WGSL shader (never changes)
- CPU-side VRAM buffer (numpy array)
- All logic in Python (debuggable, LLM-friendly)
- GPU is just a "display wire"
"""

import numpy as np
import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import pathlib

# Display dimensions
WIDTH, HEIGHT = 800, 600


class PxOSTerminalGPU:
    """
    The GPU terminal for pxOS.

    LLM/users interact with this via terminal commands.
    All rendering logic is CPU-side for easy debugging.
    """

    def __init__(self):
        print("[pxOS GPU Terminal] Initializing...")

        # Create window
        self.canvas = WgpuCanvas(
            title="pxOS GPU Terminal v0.1",
            size=(WIDTH, HEIGHT)
        )

        # Request GPU adapter and device
        self.adapter = wgpu.gpu.request_adapter(
            canvas=self.canvas,
            power_preference="high-performance"
        )
        self.device = self.adapter.request_device()

        # CPU-side VRAM buffer (this is what LLM manipulates)
        # Shape: (height, width, 4) for RGBA
        # dtype: uint8 for 0-255 values
        self.vram = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)

        print(f"[pxOS] VRAM buffer created: {self.vram.shape}, {self.vram.dtype}")

        # Initialize rendering pipeline
        self._init_gpu_resources()

        # Set up canvas draw callback
        self.canvas.request_draw(self.draw_frame)

        print("[pxOS] GPU Terminal ready!")

    def _init_gpu_resources(self):
        """Initialize all GPU resources (texture, sampler, pipeline, etc.)"""

        # Create VRAM texture on GPU
        self.vram_tex = self.device.create_texture(
            size=(WIDTH, HEIGHT, 1),
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension="2d",
            format="rgba8unorm",
            mip_level_count=1,
            sample_count=1,
        )

        self.vram_tex_view = self.vram_tex.create_view()

        # Create sampler (nearest neighbor for pixel-perfect display)
        self.sampler = self.device.create_sampler(
            mag_filter="nearest",
            min_filter="nearest",
        )

        # Load frozen shader
        shader_path = pathlib.Path(__file__).parent / "shaders" / "frozen_display.wgsl"
        shader_code = shader_path.read_text()
        shader_module = self.device.create_shader_module(code=shader_code)

        print("[pxOS] Frozen shader loaded: frozen_display.wgsl v0.1")

        # Create bind group layout
        bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": "filtering"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": "float",
                        "view_dimension": "2d",
                    },
                },
            ]
        )

        # Create bind group
        self.bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {"binding": 0, "resource": self.sampler},
                {"binding": 1, "resource": self.vram_tex_view},
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
                        "array_stride": 4 * 4,  # 4 floats (x, y, u, v)
                        "attributes": [
                            {"format": "float32x2", "offset": 0, "shader_location": 0},
                            {"format": "float32x2", "offset": 8, "shader_location": 1},
                        ],
                    }
                ],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [{"format": "bgra8unorm"}],
            },
            primitive={"topology": "triangle-list"},
        )

        # Create full-screen quad vertex buffer
        # Two triangles covering the screen in NDC (-1 to 1)
        vertices = np.array(
            [
                # Triangle 1
                [-1.0, -1.0, 0.0, 1.0],  # bottom-left
                [ 1.0, -1.0, 1.0, 1.0],  # bottom-right
                [-1.0,  1.0, 0.0, 0.0],  # top-left
                # Triangle 2
                [-1.0,  1.0, 0.0, 0.0],  # top-left
                [ 1.0, -1.0, 1.0, 1.0],  # bottom-right
                [ 1.0,  1.0, 1.0, 0.0],  # top-right
            ],
            dtype=np.float32,
        )

        self.vertex_buffer = self.device.create_buffer_with_data(
            data=vertices,
            usage=wgpu.BufferUsage.VERTEX,
        )

        print("[pxOS] GPU resources initialized")

    def upload_vram(self):
        """Upload CPU VRAM buffer to GPU texture"""
        data = self.vram.tobytes()
        self.device.queue.write_texture(
            {
                "texture": self.vram_tex,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            data,
            {
                "bytes_per_row": WIDTH * 4,
                "rows_per_image": HEIGHT,
            },
            (WIDTH, HEIGHT, 1),
        )

    def draw_frame(self):
        """Main rendering loop - called each frame"""

        # Upload VRAM to GPU
        self.upload_vram()

        # Get current swap chain texture
        current_texture = self.canvas.get_current_texture()

        # Create command encoder
        command_encoder = self.device.create_command_encoder()

        # Begin render pass
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": current_texture.create_view(),
                    "resolve_target": None,
                    "clear_value": (0, 0, 0, 1),
                    "load_op": "clear",
                    "store_op": "store",
                }
            ],
        )

        # Draw full-screen quad with frozen shader
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(6, 1, 0, 0)
        render_pass.end()

        # Submit commands
        self.device.queue.submit([command_encoder.finish()])

        # Request next frame
        self.canvas.request_draw(self.draw_frame)

    # ========================================================================
    # Terminal Commands (LLM-facing API)
    # ========================================================================

    def cmd_clear(self, r: int, g: int, b: int, a: int = 255):
        """
        CLEAR r g b [a]

        Clear the entire VRAM buffer to a solid color.

        Args:
            r, g, b, a: Color components (0-255)

        Example:
            terminal.cmd_clear(0, 0, 0, 255)  # Clear to black
            terminal.cmd_clear(255, 0, 0)      # Clear to red
        """
        self.vram[:, :] = [r, g, b, a]
        print(f"[CMD] CLEAR {r} {g} {b} {a}")

    def cmd_pixel(self, x: int, y: int, r: int, g: int, b: int, a: int = 255):
        """
        PIXEL x y r g b [a]

        Set a single pixel in VRAM.

        Args:
            x, y: Pixel coordinates (0,0 is top-left)
            r, g, b, a: Color components (0-255)

        Example:
            terminal.cmd_pixel(100, 100, 255, 255, 255)  # White pixel at (100, 100)
        """
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            self.vram[y, x] = [r, g, b, a]
            print(f"[CMD] PIXEL {x} {y} {r} {g} {b} {a}")
        else:
            print(f"[CMD] PIXEL out of bounds: ({x}, {y})")

    def cmd_rect(self, x: int, y: int, w: int, h: int, r: int, g: int, b: int, a: int = 255):
        """
        RECT x y w h r g b [a]

        Draw a filled rectangle in VRAM.

        Args:
            x, y: Top-left corner
            w, h: Width and height
            r, g, b, a: Color components (0-255)

        Example:
            terminal.cmd_rect(50, 50, 100, 100, 0, 255, 0)  # Green 100x100 square
        """
        # Clamp to screen bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(WIDTH, x + w)
        y2 = min(HEIGHT, y + h)

        if x2 > x1 and y2 > y1:
            self.vram[y1:y2, x1:x2] = [r, g, b, a]
            print(f"[CMD] RECT {x} {y} {w} {h} {r} {g} {b} {a}")
        else:
            print(f"[CMD] RECT out of bounds or invalid size")

    def run(self):
        """Start the event loop"""
        print("[pxOS] Starting event loop...")
        run()


def main():
    """Demo: Create terminal and run some test commands"""

    # Create terminal
    terminal = PxOSTerminalGPU()

    # Demo commands
    print("\n[pxOS] Running demo commands...")

    # Clear to dark blue
    terminal.cmd_clear(0, 0, 64)

    # Draw some test pixels
    terminal.cmd_pixel(400, 300, 255, 255, 255)  # White center pixel

    # Draw test rectangles
    terminal.cmd_rect(50, 50, 100, 100, 255, 0, 0)      # Red square
    terminal.cmd_rect(650, 50, 100, 100, 0, 255, 0)     # Green square
    terminal.cmd_rect(50, 450, 100, 100, 0, 0, 255)     # Blue square
    terminal.cmd_rect(650, 450, 100, 100, 255, 255, 0)  # Yellow square

    # Draw a centered white rectangle
    terminal.cmd_rect(300, 250, 200, 100, 255, 255, 255)

    print("\n[pxOS] Demo complete. Close window to exit.")

    # Start event loop
    terminal.run()


if __name__ == "__main__":
    main()
