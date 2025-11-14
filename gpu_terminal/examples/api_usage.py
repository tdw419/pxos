"""
Example: How to use the GPU terminal API from Python

This shows how an LLM or pxVM would interact with the terminal
programmatically (without using the text protocol).
"""

from pxos_gpu_terminal import PxOSTerminalGPU


def draw_gradient_demo():
    """Example: Draw a rainbow gradient"""
    terminal = PxOSTerminalGPU()

    # Clear to black
    terminal.cmd_clear(0, 0, 0)

    # Draw horizontal rainbow gradient
    for x in range(800):
        # Calculate hue (0-360 degrees)
        hue = (x / 800) * 360

        # Simple HSV to RGB conversion (S=1, V=1)
        h = hue / 60
        i = int(h)
        f = h - i
        q = int((1 - f) * 255)
        t = int(f * 255)

        if i == 0:
            r, g, b = 255, t, 0
        elif i == 1:
            r, g, b = q, 255, 0
        elif i == 2:
            r, g, b = 0, 255, t
        elif i == 3:
            r, g, b = 0, q, 255
        elif i == 4:
            r, g, b = t, 0, 255
        else:
            r, g, b = 255, 0, q

        # Draw vertical bar
        terminal.cmd_rect(x, 200, 1, 200, r, g, b)

    print("[Demo] Rainbow gradient complete")
    terminal.run()


def draw_mandelbrot():
    """Example: Draw a simple Mandelbrot set"""
    terminal = PxOSTerminalGPU()

    print("[Demo] Computing Mandelbrot set...")

    # Mandelbrot parameters
    width, height = 800, 600
    max_iter = 50
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.0, 1.0

    for py in range(height):
        if py % 50 == 0:
            print(f"[Demo] Row {py}/{height}")

        for px in range(width):
            # Map pixel to complex plane
            x0 = x_min + (px / width) * (x_max - x_min)
            y0 = y_min + (py / height) * (y_max - y_min)

            # Mandelbrot iteration
            x, y = 0, 0
            iteration = 0

            while x*x + y*y < 4 and iteration < max_iter:
                xtemp = x*x - y*y + x0
                y = 2*x*y + y0
                x = xtemp
                iteration += 1

            # Color based on iteration count
            if iteration == max_iter:
                # In the set: black
                r, g, b = 0, 0, 0
            else:
                # Outside: color by iteration
                ratio = iteration / max_iter
                r = int(255 * ratio)
                g = int(128 * (1 - ratio))
                b = int(255 * (1 - ratio))

            terminal.cmd_pixel(px, py, r, g, b)

    print("[Demo] Mandelbrot complete")
    terminal.run()


def draw_game_of_life():
    """Example: Animate Conway's Game of Life"""
    import numpy as np
    import time

    terminal = PxOSTerminalGPU()

    width, height = 800, 600
    cell_size = 4
    grid_w, grid_h = width // cell_size, height // cell_size

    # Initialize random grid
    grid = np.random.randint(0, 2, (grid_h, grid_w), dtype=np.uint8)

    print("[Demo] Running Game of Life (this will run indefinitely)")
    print("[Demo] Close window to exit")

    # Note: This won't actually animate since we can't easily
    # hook into the wgpu event loop yet. This is just to show
    # what the pattern would look like.
    # In a real implementation, you'd need a timer callback.

    for generation in range(100):
        # Clear screen
        terminal.cmd_clear(0, 0, 0)

        # Draw current generation
        for y in range(grid_h):
            for x in range(grid_w):
                if grid[y, x]:
                    terminal.cmd_rect(
                        x * cell_size,
                        y * cell_size,
                        cell_size,
                        cell_size,
                        0, 255, 0  # Green cells
                    )

        # Compute next generation
        new_grid = np.zeros_like(grid)
        for y in range(grid_h):
            for x in range(grid_w):
                # Count neighbors (with wrapping)
                neighbors = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ny = (y + dy) % grid_h
                        nx = (x + dx) % grid_w
                        neighbors += grid[ny, nx]

                # Apply rules
                if grid[y, x]:
                    # Cell is alive
                    if neighbors in [2, 3]:
                        new_grid[y, x] = 1
                else:
                    # Cell is dead
                    if neighbors == 3:
                        new_grid[y, x] = 1

        grid = new_grid

        print(f"[Demo] Generation {generation}")

        # Note: In a real implementation, you'd render each frame
        # Currently this just prepares the last frame

    terminal.run()


if __name__ == "__main__":
    # Run one of the demos:

    # Uncomment the demo you want to run:
    draw_gradient_demo()
    # draw_mandelbrot()  # This takes a while to compute
    # draw_game_of_life()  # This doesn't animate yet (needs timer integration)
