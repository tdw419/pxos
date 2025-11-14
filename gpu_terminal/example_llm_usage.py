"""
Example: How an LLM would interact with pxOS GPU Terminal

This demonstrates the LLM-friendly API design.
The LLM can issue simple terminal commands to create graphics.
"""

from pxos_gpu_terminal import PxOSTerminalGPU


def llm_drawing_session():
    """
    Simulated LLM drawing session.

    The LLM receives requests like:
    - "Draw a red square"
    - "Clear the screen to blue"
    - "Draw a white pixel at the center"

    And translates them to terminal commands.
    """

    terminal = PxOSTerminalGPU()

    # ========================================================================
    # User: "Clear the screen to black"
    # ========================================================================
    print("\n[LLM] User request: 'Clear the screen to black'")
    print("[LLM] Executing: CLEAR 0 0 0")
    terminal.cmd_clear(0, 0, 0)

    # ========================================================================
    # User: "Draw a red square in the top-left corner"
    # ========================================================================
    print("\n[LLM] User request: 'Draw a red square in the top-left corner'")
    print("[LLM] Executing: RECT 20 20 100 100 255 0 0")
    terminal.cmd_rect(20, 20, 100, 100, 255, 0, 0)

    # ========================================================================
    # User: "Draw a white border around the screen"
    # ========================================================================
    print("\n[LLM] User request: 'Draw a white border around the screen'")
    print("[LLM] Executing border commands...")
    border = 5
    terminal.cmd_rect(0, 0, 800, border, 255, 255, 255)           # Top
    terminal.cmd_rect(0, 595, 800, border, 255, 255, 255)         # Bottom
    terminal.cmd_rect(0, 0, border, 600, 255, 255, 255)           # Left
    terminal.cmd_rect(795, 0, border, 600, 255, 255, 255)         # Right

    # ========================================================================
    # User: "Draw a pixel art smiley face"
    # ========================================================================
    print("\n[LLM] User request: 'Draw a pixel art smiley face'")
    print("[LLM] Planning: I'll draw a simple 20x20 pixel face at center")

    cx, cy = 400, 300  # Center
    yellow = (255, 255, 0)

    # Face background (circle approximation with rect)
    terminal.cmd_rect(cx - 10, cy - 10, 20, 20, *yellow)

    # Eyes (black pixels)
    terminal.cmd_pixel(cx - 4, cy - 3, 0, 0, 0)
    terminal.cmd_pixel(cx + 4, cy - 3, 0, 0, 0)

    # Smile (black pixels)
    for x in range(-4, 5):
        y_offset = abs(x) // 2
        terminal.cmd_pixel(cx + x, cy + 3 + y_offset, 0, 0, 0)

    print("[LLM] Smiley face complete!")

    # ========================================================================
    # User: "Draw a rainbow gradient from left to right"
    # ========================================================================
    print("\n[LLM] User request: 'Draw a rainbow gradient from left to right'")
    print("[LLM] Planning: Create a horizontal rainbow gradient bar")

    y_start = 500
    bar_height = 80
    bar_width = 800

    for x in range(bar_width):
        # Simple rainbow: interpolate through RGB
        hue = x / bar_width
        if hue < 0.17:  # Red to yellow
            r, g, b = 255, int(hue * 6 * 255), 0
        elif hue < 0.34:  # Yellow to green
            r, g, b = int((0.34 - hue) * 6 * 255), 255, 0
        elif hue < 0.51:  # Green to cyan
            r, g, b = 0, 255, int((hue - 0.34) * 6 * 255)
        elif hue < 0.68:  # Cyan to blue
            r, g, b = 0, int((0.68 - hue) * 6 * 255), 255
        elif hue < 0.85:  # Blue to magenta
            r, g, b = int((hue - 0.68) * 6 * 255), 0, 255
        else:  # Magenta to red
            r, g, b = 255, 0, int((1.0 - hue) * 6 * 255)

        terminal.cmd_rect(x, y_start, 1, bar_height, r, g, b)

    print("[LLM] Rainbow gradient complete!")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("LLM Drawing Session Complete!")
    print("=" * 60)
    print("\nCommands executed:")
    print("  1. CLEAR - Clear screen to black")
    print("  2. RECT - Draw red square")
    print("  3. RECT (x4) - Draw white border")
    print("  4. RECT + PIXEL - Draw smiley face")
    print("  5. RECT (x800) - Draw rainbow gradient")
    print("\nAll graphics created via CPU-side terminal commands.")
    print("Frozen shader just displays the VRAM buffer.")
    print("\nClose window to exit.")
    print("=" * 60)

    terminal.run()


def llm_debugging_example():
    """
    Example: How an LLM would debug graphics issues

    Since all logic is CPU-side, the LLM can easily inspect and debug.
    """

    terminal = PxOSTerminalGPU()

    print("\n" + "=" * 60)
    print("LLM Debugging Example")
    print("=" * 60)

    # User reports: "The red square I drew doesn't look right"

    print("\n[LLM] Drawing the red square as reported...")
    terminal.cmd_rect(100, 100, 50, 50, 255, 0, 0)

    print("\n[LLM] Let me inspect the VRAM to debug...")

    # LLM can directly inspect VRAM
    region = terminal.vram[100:150, 100:150]  # Get the square region

    print(f"[LLM] VRAM region shape: {region.shape}")
    print(f"[LLM] Red channel min/max: {region[:,:,0].min()}/{region[:,:,0].max()}")
    print(f"[LLM] Green channel min/max: {region[:,:,1].min()}/{region[:,:,1].max()}")
    print(f"[LLM] Blue channel min/max: {region[:,:,2].min()}/{region[:,:,2].max()}")

    # Check if it's actually red
    is_red = (region[:, :, 0] == 255).all() and \
             (region[:, :, 1] == 0).all() and \
             (region[:, :, 2] == 0).all()

    if is_red:
        print("[LLM] ✅ The VRAM data is correct - it's pure red (255,0,0)")
        print("[LLM] The issue must be in display or user expectation.")
    else:
        print("[LLM] ❌ The VRAM data is wrong! Let me fix it...")
        terminal.cmd_rect(100, 100, 50, 50, 255, 0, 0)

    print("\n[LLM] Debug complete. This is why frozen shaders are great:")
    print("  - Direct VRAM access for inspection")
    print("  - Can print/validate data at any time")
    print("  - Shader is not a variable in the debugging process")

    terminal.run()


if __name__ == "__main__":
    # Run the LLM drawing session
    llm_drawing_session()

    # Uncomment to run debugging example instead:
    # llm_debugging_example()
