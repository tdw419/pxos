# hello.asm - Simplest possible program
# Draws a single magenta pixel at the center of the world

    MOV R0, 512         # X coordinate (center)
    MOV R1, 512         # Y coordinate (center)
    MOV R2, 0xFF00FF    # Color (magenta)
    PLOT                # Draw pixel
    HALT                # Stop execution
