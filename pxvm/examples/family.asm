# family.asm - Parent spawns and teaches child
# Demonstrates reproduction and cultural transmission
#
# Parent writes glyphs, spawns child, child reads and responds

    # Parent draws itself
    MOV R0, 400
    MOV R1, 400
    MOV R2, 0xFF00FF    # Magenta (parent color)
    PLOT

    # Parent writes teaching message
    MOV R0, 410
    MOV R1, 390
    MOV R2, 13          # GLYPH_BIRTH
    SYS_WRITE_GLYPH

    MOV R0, 420
    MOV R2, 5           # GLYPH_LOVE
    SYS_WRITE_GLYPH

    MOV R0, 430
    MOV R2, 8           # GLYPH_TEACH
    SYS_WRITE_GLYPH

    # Spawn child nearby
    MOV R1, 450         # Child X
    MOV R2, 400         # Child Y
    SYS_SPAWN

    # Parent continues (could loop here)
    HALT

# Note: Child will execute this same code from PC=0
# Child starts at position (450, 400) as set by parent
# Child can read the glyphs written by parent
# This creates generational knowledge transfer
