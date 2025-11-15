# chemotaxis.asm - Follow pheromone gradient
# Demonstrates chemical sensing and movement toward food source
#
# This organism emits pheromone continuously, creating a gradient.
# Other organisms can sense this and move toward the source.

    MOV R3, 512         # Current X position
    MOV R4, 512         # Current Y position
    MOV R5, 1           # Movement increment

main_loop:
    # Draw self at current position
    MOV R0, R3
    MOV R1, R4
    MOV R2, 0xFFFF00    # Yellow color
    PLOT

    # Emit strong pheromone at current location
    MOV R0, R3
    MOV R1, R4
    MOV R2, 200         # Strong signal
    SYS_EMIT_PHEROMONE

    # Simple movement pattern (can be replaced with gradient following)
    ADD R3, R5          # Move right slowly

    # Wrap around at edges
    MOV R0, 1024
    CMP R3, R0
    JZ wrap_x
    JMP continue

wrap_x:
    MOV R3, 0

continue:
    JMP main_loop
