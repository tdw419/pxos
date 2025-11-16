"""
Visual renderer for pixel filesystem map
"""

from pathlib import Path
from typing import List, Tuple
from .storage import MetadataStore


class PixelMapVisualizer:
    """Generate visual representations of the pixel map"""

    def __init__(self, metadata_store: MetadataStore):
        self.metadata_store = metadata_store

    def render_html(
        self,
        output_path: str,
        chunk_x: int = 0,
        chunk_y: int = 0,
        pixel_size: int = 10,
        show_labels: bool = True
    ):
        """
        Render pixel map as interactive HTML.

        Args:
            output_path: Where to save HTML file
            chunk_x: Chunk X coordinate to render
            chunk_y: Chunk Y coordinate to render
            pixel_size: Size of each pixel in pixels
            show_labels: Whether to show file names on hover
        """
        # Get all pixels in chunk
        pixels = self.metadata_store.get_pixels_in_chunk(chunk_x, chunk_y)

        if not pixels:
            print(f"No pixels found in chunk ({chunk_x}, {chunk_y})")
            return

        # Calculate bounds
        min_x = min(p["position"]["x"] for p in pixels)
        max_x = max(p["position"]["x"] for p in pixels)
        min_y = min(p["position"]["y"] for p in pixels)
        max_y = max(p["position"]["y"] for p in pixels)

        width = (max_x - min_x + 1) * pixel_size + 100
        height = (max_y - min_y + 1) * pixel_size + 100

        # Generate HTML
        html = self._generate_html(
            pixels,
            width,
            height,
            pixel_size,
            min_x,
            min_y,
            show_labels,
            chunk_x,
            chunk_y
        )

        # Write to file
        Path(output_path).write_text(html)
        print(f"✓ Visualization saved to: {output_path}")
        print(f"  Canvas size: {width}x{height}")
        print(f"  Pixels rendered: {len(pixels)}")

    def _generate_html(
        self,
        pixels: List[dict],
        width: int,
        height: int,
        pixel_size: int,
        min_x: int,
        min_y: int,
        show_labels: bool,
        chunk_x: int,
        chunk_y: int
    ) -> str:
        """Generate HTML content"""
        pixel_elements = []

        for px in pixels:
            x = (px["position"]["x"] - min_x) * pixel_size + 50
            y = (px["position"]["y"] - min_y) * pixel_size + 50

            r = px["visual"]["r"]
            g = px["visual"]["g"]
            b = px["visual"]["b"]
            a = px["visual"]["a"] / 255.0

            size = pixel_size
            if px["type"] == "directory":
                size = pixel_size * 2  # Directories are larger

            # Create pixel element
            style = f"left:{x}px; top:{y}px; width:{size}px; height:{size}px; background-color:rgba({r},{g},{b},{a});"
            if px["type"] == "directory":
                style += " border: 2px solid white;"

            title = f"{px['metadata']['name']} ({px['type']}) - {px['metadata']['size']:,} bytes"

            pixel_elements.append(
                f'<div class="pixel" style="{style}" title="{title}"></div>'
            )

        stats = self.metadata_store.get_stats()

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>pxOS Pixel Map - Chunk ({chunk_x}, {chunk_y})</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: #000;
            font-family: 'Courier New', monospace;
            color: #0f0;
        }}

        h1 {{
            text-align: center;
            margin-bottom: 10px;
        }}

        .info {{
            text-align: center;
            margin-bottom: 20px;
            font-size: 12px;
        }}

        .canvas {{
            position: relative;
            width: {width}px;
            height: {height}px;
            background: #111;
            margin: 0 auto;
            border: 1px solid #333;
            image-rendering: pixelated;
        }}

        .pixel {{
            position: absolute;
            cursor: pointer;
            transition: transform 0.1s;
        }}

        .pixel:hover {{
            transform: scale(1.5);
            z-index: 1000;
            box-shadow: 0 0 10px rgba(255,255,255,0.5);
        }}

        .legend {{
            margin: 20px auto;
            width: {width}px;
            display: flex;
            justify-content: space-around;
            font-size: 12px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
        }}

        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 5px;
            border: 1px solid #fff;
        }}

        .stats {{
            text-align: center;
            margin-top: 20px;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <h1>pxOS Pixel Filesystem Map</h1>
    <div class="info">
        Chunk ({chunk_x}, {chunk_y}) | {len(pixels)} pixels | Pixel size: {pixel_size}px
    </div>

    <div class="canvas">
        {"".join(pixel_elements)}
    </div>

    <div class="legend">
        <div class="legend-item">
            <div class="legend-color" style="background-color: rgb(255,50,50);"></div>
            <span>Executable/Binary</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: rgb(200,50,50);"></div>
            <span>Media</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: rgb(150,50,50);"></div>
            <span>Text</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: rgb(100,50,50);"></div>
            <span>Archive</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: rgb(100,100,200);"></div>
            <span>Directory</span>
        </div>
    </div>

    <div class="stats">
        <strong>Storage Stats:</strong><br>
        Total Pixels: {stats['total_pixels']:,} |
        Unique Files: {stats['unique_files']:,} |
        Total Size: {stats['total_size']:,} bytes |
        Directories: {stats['directories']:,} |
        Files: {stats['files']:,}
    </div>

    <script>
        // Click to get pixel info
        document.querySelectorAll('.pixel').forEach(el => {{
            el.addEventListener('click', function() {{
                alert(this.getAttribute('title'));
            }});
        }});

        // Keyboard navigation
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') {{
                window.close();
            }}
        }});
    </script>
</body>
</html>"""

        return html

    def render_ascii(
        self,
        chunk_x: int = 0,
        chunk_y: int = 0,
        width: int = 80,
        height: int = 24
    ) -> str:
        """
        Render pixel map as ASCII art for terminal display.

        Args:
            chunk_x: Chunk X coordinate
            chunk_y: Chunk Y coordinate
            width: Terminal width in characters
            height: Terminal height in lines

        Returns:
            ASCII art string
        """
        pixels = self.metadata_store.get_pixels_in_chunk(chunk_x, chunk_y)

        if not pixels:
            return f"No pixels found in chunk ({chunk_x}, {chunk_y})"

        # Calculate bounds
        min_x = min(p["position"]["x"] for p in pixels)
        max_x = max(p["position"]["x"] for p in pixels)
        min_y = min(p["position"]["y"] for p in pixels)
        max_y = max(p["position"]["y"] for p in pixels)

        # Create canvas
        canvas = [[' ' for _ in range(width)] for _ in range(height)]

        # Map pixels to canvas
        for px in pixels:
            # Normalize coordinates to canvas
            x = int((px["position"]["x"] - min_x) / (max_x - min_x + 1) * (width - 1))
            y = int((px["position"]["y"] - min_y) / (max_y - min_y + 1) * (height - 1))

            if 0 <= x < width and 0 <= y < height:
                # Choose character based on type
                if px["type"] == "directory":
                    canvas[y][x] = '█'
                elif px["type"] == "executable":
                    canvas[y][x] = '●'
                else:
                    canvas[y][x] = '·'

        # Convert to string
        result = [
            f"pxOS Pixel Map - Chunk ({chunk_x}, {chunk_y})",
            "=" * width,
            *[''.join(row) for row in canvas],
            "=" * width,
            f"Legend: █=Directory ●=Executable ·=File | Total: {len(pixels)} pixels"
        ]

        return '\n'.join(result)
