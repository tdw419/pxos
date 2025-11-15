# line.asm - Draw a diagonal line
# Demonstrates loops, comparisons, and conditional jumps

    MOV R0, 100         # Starting X
    MOV R1, 100         # Starting Y
    MOV R2, 0x00FFFF    # Cyan color
    MOV R3, 1           # Increment
    MOV R4, 500         # End position

loop:
    PLOT                # Draw pixel at (R0, R1)
    ADD R0, R3          # X++
    ADD R1, R3          # Y++
    CMP R0, R4          # Check if X >= 500
    JZ done             # If yes, stop
    JMP loop            # Otherwise continue

done:
    HALT
