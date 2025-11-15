; follow.asm — First organism with a nose
main:
    MOV R7, 50          ; search radius
scan:
    MOV R5, -50         ; dx
dx_loop:
    MOV R6, -50         ; dy
dy_loop:
    MOV R0, R5
    ADD R0, R3          ; R3 = current x (self position)
    MOV R1, R6
    ADD R1, R4          ; R4 = current y
    SYS_SENSE_PHEROMONE
    CMP R0, R7
    JZ found_stronger
    ADD R6, 1
    CMP R6, 50
    JNZ dy_loop
    ADD R5, 1
    CMP R5, 50
    JNZ dx_loop
    JMP move_random
found_stronger:
    MOV R0, R3
    ADD R0, R5          ; move toward scent
    MOV R1, R4
    ADD R1, R6
    MOV R2, 0x00FFFF    ; cyan trail
    SYS_PLOT
    MOV R3, R0          ; update position
    MOV R4, R1
    JMP main

move_random:
    ; placeholder for random movement
    JMP main
