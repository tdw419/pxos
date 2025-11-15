# naming.asm - Write name in glyphs
# Demonstrates symbolic communication and self-identity
#
# This organism writes "I AM" followed by symbolic glyphs

    # Draw self at home position
    MOV R0, 400
    MOV R1, 400
    MOV R2, 0xFF00FF    # Magenta
    PLOT

    # Write "I AM" sequence
    MOV R0, 420
    MOV R1, 400
    MOV R2, 1           # GLYPH_SELF ("I")
    SYS_WRITE_GLYPH

    MOV R0, 430
    MOV R2, 7           # GLYPH_NAME ("AM")
    SYS_WRITE_GLYPH

    # Write custom name glyphs
    MOV R0, 440
    MOV R2, 11          # Custom symbol
    SYS_WRITE_GLYPH

    MOV R0, 450
    MOV R2, 1           # Custom symbol
    SYS_WRITE_GLYPH

    MOV R0, 460
    MOV R2, 5           # GLYPH_LOVE
    SYS_WRITE_GLYPH

    HALT
