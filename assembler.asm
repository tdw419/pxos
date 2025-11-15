; =========================================================
; assembler.asm — The Self-Hosting Compiler Process
; pxOS Kernel v1.0 — Built-In Assembler v0.2
; =========================================================

ORG 0x0000

ENTRY:
    ; Initialize state
    IMM32 R7, 0xFFFFFFF0      ; Stack pointer

    PRINT "Assembler process started."

MAIN_LOOP:
    SYSCALL 13 ; SLEEP 100
    PRINT "Assembler is alive."
    JMP MAIN_LOOP

HALT
