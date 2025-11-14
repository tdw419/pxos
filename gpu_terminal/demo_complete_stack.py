"""
Complete Stack Demo - Frozen Shader Architecture

This demonstrates the full stack working together:
1. Frozen shader (WGSL - never changes)
2. GPU terminal (Python API)
3. Drawing primitives (numpy operations)
4. Visual output

No GPU complexity, just simple drawing commands.
"""

from pxos_gpu_terminal import PxOSTerminalGPU


def draw_demo_scene(terminal):
    """
    Draw a complete demo scene using all primitives.

    This shows what an LLM or pxVM could generate.
    """

    # Sky gradient background
    print("[DEMO] Drawing sky gradient...")
    for y in range(0, 400):
        # Gradient from dark blue (top) to light blue (horizon)
        ratio = y / 400
        r = int(135 * ratio)
        g = int(206 * ratio)
        b = int(235 * (0.5 + 0.5 * ratio))
        terminal.cmd_hline(0, y, 800, r, g, b)

    # Ground
    print("[DEMO] Drawing ground...")
    terminal.cmd_rect(0, 400, 800, 200, 34, 139, 34)

    # Sun
    print("[DEMO] Drawing sun...")
    sun_x, sun_y = 650, 100
    sun_radius = 30
    for r in range(sun_radius, 0, -1):
        # Circular approximation using rectangles
        y_offset = int((sun_radius**2 - r**2)**0.5)
        terminal.cmd_hline(sun_x - r, sun_y - y_offset, 2*r, 255, 255, 0)
        terminal.cmd_hline(sun_x - r, sun_y + y_offset, 2*r, 255, 255, 0)

    # House
    print("[DEMO] Drawing house...")
    house_x, house_y = 250, 250
    house_w, house_h = 200, 150

    # House body (tan)
    terminal.cmd_rect(house_x, house_y, house_w, house_h, 210, 180, 140)

    # Roof (red triangle - approximated)
    roof_layers = 50
    for i in range(roof_layers):
        y = house_y - roof_layers + i
        width = int(house_w + (roof_layers - i) * 2)
        x = house_x + house_w//2 - width//2
        terminal.cmd_hline(x, y, width, 178, 34, 34)

    # Door (brown)
    door_x = house_x + 70
    door_y = house_y + 50
    terminal.cmd_rect(door_x, door_y, 60, 100, 101, 67, 33)

    # Door frame
    terminal.cmd_hline(door_x, door_y, 60, 0, 0, 0)
    terminal.cmd_hline(door_x, door_y + 100, 60, 0, 0, 0)
    terminal.cmd_vline(door_x, door_y, 100, 0, 0, 0)
    terminal.cmd_vline(door_x + 60, door_y, 100, 0, 0, 0)

    # Windows
    window_y = house_y + 80

    # Left window
    terminal.cmd_rect(house_x + 20, window_y, 50, 50, 135, 206, 250)
    terminal.cmd_hline(house_x + 20, window_y + 25, 50, 64, 64, 64)
    terminal.cmd_vline(house_x + 45, window_y, 50, 64, 64, 64)

    # Right window
    terminal.cmd_rect(house_x + 130, window_y, 50, 50, 135, 206, 250)
    terminal.cmd_hline(house_x + 130, window_y + 25, 50, 64, 64, 64)
    terminal.cmd_vline(house_x + 155, window_y, 50, 64, 64, 64)

    # Tree
    print("[DEMO] Drawing tree...")
    tree_x = 550

    # Trunk
    terminal.cmd_rect(tree_x, 320, 20, 80, 101, 67, 33)

    # Foliage (green circles - approximated)
    foliage_centers = [
        (tree_x + 10, 280),
        (tree_x - 15, 300),
        (tree_x + 35, 300),
        (tree_x + 10, 315),
    ]

    for cx, cy in foliage_centers:
        radius = 25
        for r in range(radius, 0, -1):
            y_offset = int((radius**2 - r**2)**0.5)
            terminal.cmd_hline(cx - r, cy - y_offset, 2*r, 34, 139, 34)
            terminal.cmd_hline(cx - r, cy + y_offset, 2*r, 34, 139, 34)

    # Fence
    print("[DEMO] Drawing fence...")
    fence_y = 400
    for x in range(50, 750, 40):
        # Vertical post
        terminal.cmd_vline(x, fence_y - 40, 40, 139, 69, 19)
        terminal.cmd_vline(x + 1, fence_y - 40, 40, 139, 69, 19)
        # Pointed top
        terminal.cmd_pixel(x, fence_y - 41, 139, 69, 19)
        terminal.cmd_pixel(x + 1, fence_y - 41, 139, 69, 19)

    # Horizontal fence rails
    terminal.cmd_hline(50, fence_y - 30, 700, 139, 69, 19)
    terminal.cmd_hline(50, fence_y - 31, 700, 139, 69, 19)
    terminal.cmd_hline(50, fence_y - 10, 700, 139, 69, 19)
    terminal.cmd_hline(50, fence_y - 11, 700, 139, 69, 19)

    # Clouds
    print("[DEMO] Drawing clouds...")
    def draw_cloud(cx, cy):
        """Draw a simple cloud using overlapping circles"""
        cloud_parts = [
            (cx - 20, cy, 15),
            (cx, cy - 5, 18),
            (cx + 20, cy, 15),
            (cx + 10, cy + 5, 12),
        ]
        for cloud_x, cloud_y, radius in cloud_parts:
            for r in range(radius, 0, -1):
                y_offset = int((radius**2 - r**2)**0.5)
                terminal.cmd_hline(cloud_x - r, cloud_y - y_offset, 2*r, 255, 255, 255)
                terminal.cmd_hline(cloud_x - r, cloud_y + y_offset, 2*r, 255, 255, 255)

    draw_cloud(150, 80)
    draw_cloud(400, 60)
    draw_cloud(600, 90)

    # Ground details (grass tufts)
    print("[DEMO] Drawing grass details...")
    import random
    random.seed(42)  # Reproducible randomness
    for _ in range(50):
        x = random.randint(0, 800)
        y = random.randint(420, 580)
        height = random.randint(5, 15)
        terminal.cmd_vline(x, y, height, 0, 100, 0)

    print("[DEMO] Scene complete!")


def main():
    """Main demo"""

    print("=" * 70)
    print("pxOS GPU Terminal - Complete Stack Demo")
    print("=" * 70)
    print()
    print("Demonstrating:")
    print("  1. Frozen shader v0.1 (WGSL - never changes)")
    print("  2. GPU terminal (Python API)")
    print("  3. All 5 drawing primitives:")
    print("     - CLEAR (screen fill)")
    print("     - PIXEL (individual pixels)")
    print("     - RECT (filled rectangles)")
    print("     - HLINE (horizontal lines)")
    print("     - VLINE (vertical lines)")
    print()
    print("Building scene with ~1000+ drawing commands...")
    print()

    # Create terminal
    terminal = PxOSTerminalGPU()

    # Draw the demo scene
    draw_demo_scene(terminal)

    print()
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print()
    print("What you're seeing:")
    print("  - Sky gradient (HLINE)")
    print("  - Ground (RECT)")
    print("  - Sun (HLINE + approximated circle)")
    print("  - House with windows and door (RECT + HLINE + VLINE)")
    print("  - Tree with foliage (RECT + HLINE approximated circles)")
    print("  - Fence (VLINE + HLINE)")
    print("  - Clouds (HLINE approximated circles)")
    print("  - Grass tufts (VLINE with random placement)")
    print()
    print("All rendered using:")
    print("  - CPU-side numpy operations (debuggable!)")
    print("  - Frozen shader v0.1 (never changed!)")
    print("  - Zero GPU complexity")
    print()
    print("This is what an LLM or pxVM could generate.")
    print()
    print("Close window to exit.")
    print("=" * 70)

    # Run the event loop
    terminal.run()


if __name__ == "__main__":
    main()
