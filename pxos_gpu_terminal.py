#!/usr/bin/env python3
"""
PxOS GPU Terminal - Layered rendering with frozen WGSL shader
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
import wgpu
from wgpu.utils.device import get_default_device
from PIL import Image

from pxos_text import BitmapFont, TextRenderer


@dataclass
class Layer:
    """A single render layer with z-order"""
    name: str
    z: int
    buffer: np.ndarray  # RGBA uint8 array


class PxOSTerminalGPU:
    """GPU-accelerated terminal with multi-layer compositing"""

    def __init__(self, width: int = 800, height: int = 600, use_cpu: bool = False):
        self.width = width
        self.height = height
        self.use_cpu = use_cpu

        # Layer management
        self.layers: Dict[str, Layer] = {}
        self.current_layer_name = "background"

        # Text rendering
        self.font = BitmapFont()
        self.text_renderer = TextRenderer(self.font)

        # Create default background layer
        self._create_default_layer()

        # Initialize GPU (if available)
        if not use_cpu:
            try:
                self._init_gpu()
            except Exception as e:
                print(f"GPU initialization failed: {e}")
                print("Falling back to CPU rendering")
                self.use_cpu = True

    def _create_default_layer(self):
        """Create the default background layer"""
        self.add_layer("background", z=0)

    @property
    def current_layer(self) -> Layer:
        """Get the currently active layer"""
        return self.layers[self.current_layer_name]

    def add_layer(self, name: str, z: int = 0):
        """Add a new layer"""
        buffer = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self.layers[name] = Layer(name=name, z=z, buffer=buffer)

    def set_layer(self, name: str):
        """Switch to a different layer"""
        if name not in self.layers:
            self.add_layer(name)
        self.current_layer_name = name

    def clear_layer(self, r: int = 0, g: int = 0, b: int = 0, a: int = 0):
        """Clear current layer to a color"""
        layer = self.current_layer
        layer.buffer[:, :] = [r, g, b, a]

    def _init_gpu(self):
        """Initialize WGPU device and create pipeline"""
        self.device = get_default_device()

        # Frozen shader code
        shader_code = """
@group(0) @binding(0)
var<storage, read> layers: array<u32>;

@group(0) @binding(1)
var<storage, read> metadata: array<u32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );

    var uv = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
    );

    var output: VertexOutput;
    output.position = vec4<f32>(pos[vertex_index], 0.0, 1.0);
    output.uv = uv[vertex_index];
    return output;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let width = metadata[0];
    let height = metadata[1];
    let num_layers = metadata[2];

    let x = u32(in.uv.x * f32(width));
    let y = u32(in.uv.y * f32(height));
    let pixel_index = y * width + x;

    var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    for (var layer_idx = 0u; layer_idx < num_layers; layer_idx++) {
        let offset = layer_idx * width * height * 4u;
        let base = pixel_index * 4u + offset;

        let r = f32(layers[base + 0u]) / 255.0;
        let g = f32(layers[base + 1u]) / 255.0;
        let b = f32(layers[base + 2u]) / 255.0;
        let a = f32(layers[base + 3u]) / 255.0;

        // Alpha blending
        let src = vec4<f32>(r, g, b, a);
        color = src * a + color * (1.0 - a);
    }

    return vec4<f32>(color.rgb, 1.0);
}
"""

        # Create shader module
        self.shader_module = self.device.create_shader_module(code=shader_code)

        # Create bind group layout
        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {
                        "type": wgpu.BufferBindingType.read_only_storage,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {
                        "type": wgpu.BufferBindingType.read_only_storage,
                    },
                },
            ]
        )

        # Create pipeline layout
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout]
        )

        # Create render pipeline
        self.pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": self.shader_module,
                "entry_point": "vs_main",
            },
            fragment={
                "module": self.shader_module,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": wgpu.TextureFormat.rgba8unorm,
                    }
                ],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
            },
        )

    def pixel(self, x: int, y: int, r: int, g: int, b: int, a: int = 255):
        """Draw a single pixel"""
        if 0 <= x < self.width and 0 <= y < self.height:
            layer = self.current_layer
            layer.buffer[y, x] = [r, g, b, a]

    def hline(self, x: int, y: int, w: int, r: int, g: int, b: int, a: int = 255):
        """Draw a horizontal line"""
        if 0 <= y < self.height:
            x_start = max(0, x)
            x_end = min(self.width, x + w)
            layer = self.current_layer
            layer.buffer[y, x_start:x_end] = [r, g, b, a]

    def rect(self, x: int, y: int, w: int, h: int, r: int, g: int, b: int, a: int = 255):
        """Draw a filled rectangle"""
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(self.width, x + w)
        y_end = min(self.height, y + h)

        if x_start < x_end and y_start < y_end:
            layer = self.current_layer
            layer.buffer[y_start:y_end, x_start:x_end] = [r, g, b, a]

    def text(self, x: int, y: int, text: str, r: int, g: int, b: int, a: int = 255):
        """Draw text using bitmap font"""
        buf = self.current_layer.buffer
        self.text_renderer.draw_text(buf, x, y, text, (r, g, b, a))

    def _draw_frame_cpu(self) -> np.ndarray:
        """CPU-based layer compositing (fallback when GPU not available)"""
        # Sort layers by z-order
        sorted_layers = sorted(self.layers.values(), key=lambda l: l.z)

        # Initialize output with transparent black
        result = np.zeros((self.height, self.width, 4), dtype=np.float32)

        # Composite layers using alpha blending
        for layer in sorted_layers:
            # Convert to float for blending
            src = layer.buffer.astype(np.float32) / 255.0

            # Extract alpha channel
            src_alpha = src[:, :, 3:4]

            # Alpha blend: result = src * alpha + result * (1 - alpha)
            result = src * src_alpha + result * (1.0 - src_alpha)

        # Convert back to uint8
        result = (result * 255).astype(np.uint8)

        return result

    def _draw_frame_gpu(self) -> np.ndarray:
        """GPU-based layer compositing"""
        # Sort layers by z-order
        sorted_layers = sorted(self.layers.values(), key=lambda l: l.z)

        # Prepare layer data for GPU
        layer_data = []
        for layer in sorted_layers:
            layer_data.append(layer.buffer.flatten())

        # Concatenate all layer data
        all_layers = np.concatenate(layer_data).astype(np.uint32)

        # Metadata: width, height, num_layers
        metadata = np.array([self.width, self.height, len(sorted_layers)], dtype=np.uint32)

        # Create GPU buffers
        layer_buffer = self.device.create_buffer_with_data(
            data=all_layers,
            usage=wgpu.BufferUsage.STORAGE,
        )

        metadata_buffer = self.device.create_buffer_with_data(
            data=metadata,
            usage=wgpu.BufferUsage.STORAGE,
        )

        # Create bind group
        bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": layer_buffer,
                        "offset": 0,
                        "size": layer_buffer.size,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": metadata_buffer,
                        "offset": 0,
                        "size": metadata_buffer.size,
                    },
                },
            ],
        )

        # Create output texture
        texture = self.device.create_texture(
            size=(self.width, self.height, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )

        # Render
        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": texture.create_view(),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0, 0, 0, 1),
                }
            ],
        )

        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, bind_group)
        render_pass.draw(6, 1, 0, 0)
        render_pass.end()

        # Read back result
        bytes_per_pixel = 4
        bytes_per_row = self.width * bytes_per_pixel
        buffer_size = bytes_per_row * self.height

        output_buffer = self.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

        command_encoder.copy_texture_to_buffer(
            {
                "texture": texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            {
                "buffer": output_buffer,
                "offset": 0,
                "bytes_per_row": bytes_per_row,
                "rows_per_image": self.height,
            },
            (self.width, self.height, 1),
        )

        self.device.queue.submit([command_encoder.finish()])

        # Map and read buffer
        output_buffer.map(wgpu.MapMode.READ)
        data = output_buffer.read_mapped()
        output_buffer.unmap()

        # Convert to numpy array
        result = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 4))

        return result

    def draw_frame(self) -> np.ndarray:
        """Composite all layers and return the final frame"""
        if self.use_cpu:
            return self._draw_frame_cpu()
        else:
            return self._draw_frame_gpu()

    def save_frame(self, filename: str):
        """Render and save frame to file"""
        frame = self.draw_frame()
        img = Image.fromarray(frame, mode='RGBA')
        img.save(filename)
        print(f"Saved frame to {filename}")
