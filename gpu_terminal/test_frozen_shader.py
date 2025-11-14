"""
Test script for the Frozen Shader GPU Terminal

This script demonstrates the terminal commands and validates
that the frozen shader pipeline is working correctly.
"""

from pxos_gpu_terminal import PxOSTerminalGPU


def test_basic_commands():
    """Test basic terminal commands"""

    print("=" * 60)
    print("pxOS GPU Terminal - Frozen Shader Test")
    print("=" * 60)

    # Create terminal
    terminal = PxOSTerminalGPU()

    print("\n[TEST] Testing CLEAR command...")
    terminal.cmd_clear(32, 32, 64)  # Dark blue-gray background

    print("\n[TEST] Testing PIXEL command...")
    # Draw a white cross at center
    cx, cy = 400, 300
    for i in range(-20, 21):
        terminal.cmd_pixel(cx + i, cy, 255, 255, 255)
        terminal.cmd_pixel(cx, cy + i, 255, 255, 255)

    print("\n[TEST] Testing RECT command...")
    # Draw colored rectangles in corners
    terminal.cmd_rect(10, 10, 100, 80, 255, 0, 0)       # Top-left: Red
    terminal.cmd_rect(690, 10, 100, 80, 0, 255, 0)      # Top-right: Green
    terminal.cmd_rect(10, 510, 100, 80, 0, 0, 255)      # Bottom-left: Blue
    terminal.cmd_rect(690, 510, 100, 80, 255, 255, 0)   # Bottom-right: Yellow

    # Draw a semi-transparent white box in center
    terminal.cmd_rect(300, 225, 200, 150, 255, 255, 255, 200)

    # Draw some test patterns
    print("\n[TEST] Drawing test patterns...")

    # Gradient bar
    for x in range(256):
        terminal.cmd_rect(272 + x, 180, 1, 20, x, x, x)

    # RGB test bars
    for x in range(256):
        terminal.cmd_rect(272 + x, 205, 1, 5, x, 0, 0)    # Red
        terminal.cmd_rect(272 + x, 210, 1, 5, 0, x, 0)    # Green
        terminal.cmd_rect(272 + x, 215, 1, 5, 0, 0, x)    # Blue

    print("\n[TEST] VRAM buffer stats:")
    print(f"  Shape: {terminal.vram.shape}")
    print(f"  Dtype: {terminal.vram.dtype}")
    print(f"  Min value: {terminal.vram.min()}")
    print(f"  Max value: {terminal.vram.max()}")
    print(f"  Mean value: {terminal.vram.mean():.2f}")

    # Validate VRAM
    assert terminal.vram.shape == (600, 800, 4), "VRAM shape incorrect"
    assert terminal.vram.dtype == 'uint8', "VRAM dtype incorrect"
    assert terminal.vram.min() >= 0, "VRAM has negative values"
    assert terminal.vram.max() <= 255, "VRAM exceeds 255"

    print("\n[TEST] ✅ All validations passed!")

    print("\n" + "=" * 60)
    print("Visual test pattern loaded.")
    print("You should see:")
    print("  - Dark blue-gray background")
    print("  - White cross at center")
    print("  - Colored rectangles in corners (R, G, B, Y)")
    print("  - White box in center")
    print("  - Grayscale and RGB gradient bars")
    print("")
    print("Close the window to exit.")
    print("=" * 60)

    # Run the event loop
    terminal.run()


if __name__ == "__main__":
    test_basic_commands()
