"""
Test script for HLINE and VLINE commands

This verifies that the line drawing primitives work correctly.
"""

from pxos_gpu_terminal import PxOSTerminalGPU


def test_lines():
    """Test horizontal and vertical line drawing"""

    terminal = PxOSTerminalGPU()

    print("[TEST] Testing line drawing commands...")

    # Clear to dark gray
    terminal.cmd_clear(32, 32, 32)

    # Test HLINE - draw horizontal rainbow
    print("[TEST] Drawing horizontal rainbow gradient...")
    for y in range(200, 400, 2):
        # Calculate hue based on y position
        hue_ratio = (y - 200) / 200
        if hue_ratio < 0.33:
            r, g, b = 255, int(hue_ratio * 3 * 255), 0
        elif hue_ratio < 0.67:
            r, g, b = int((0.67 - hue_ratio) * 3 * 255), 255, 0
        else:
            r, g, b = 0, 255, int((hue_ratio - 0.67) * 3 * 255)

        terminal.cmd_hline(0, y, 800, r, g, b)

    # Test VLINE - draw vertical bars
    print("[TEST] Drawing vertical color bars...")
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 255, 255) # White
    ]

    x = 50
    for r, g, b in colors:
        for i in range(20):  # Make bars 20 pixels wide
            terminal.cmd_vline(x + i, 50, 100, r, g, b)
        x += 30  # Space between bars

    # Test bounds checking
    print("[TEST] Testing bounds checking...")
    terminal.cmd_hline(-100, 10, 200, 255, 0, 0)  # Should clip to screen
    terminal.cmd_vline(10, -100, 200, 0, 255, 0)  # Should clip to screen
    terminal.cmd_hline(700, 20, 200, 0, 0, 255)   # Should clip to screen
    terminal.cmd_vline(20, 500, 200, 255, 255, 0) # Should clip to screen

    # Test out of bounds (should be rejected)
    terminal.cmd_hline(0, 1000, 100, 255, 0, 0)   # y out of bounds
    terminal.cmd_vline(1000, 0, 100, 0, 255, 0)   # x out of bounds

    # Draw a crosshair at center
    print("[TEST] Drawing center crosshair...")
    terminal.cmd_hline(0, 300, 800, 255, 255, 255)
    terminal.cmd_vline(400, 0, 600, 255, 255, 255)

    print("\n[TEST] Line drawing tests complete!")
    print("[TEST] Visual verification:")
    print("  - Horizontal rainbow gradient (top)")
    print("  - Vertical color bars (left)")
    print("  - White crosshair at center")
    print("  - Clipped lines at edges (should not crash)")
    print("\n[TEST] Close window to exit.")

    # Start event loop
    terminal.run()


if __name__ == "__main__":
    test_lines()
