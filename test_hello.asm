; test_hello.asm - Simple test program
; Demonstrates basic pxVM assembly

START:
    ; Print startup message
    IMM32 R1, 1          ; message_id = 1 ("PXVM booting...")
    SYSCALL 1            ; SYS_PRINT_ID

    ; Draw a window
    IMM32 R1, 200        ; x = 200
    IMM32 R2, 150        ; y = 150
    IMM32 R3, 400        ; w = 400
    IMM32 R4, 300        ; h = 300
    IMM32 R5, 1          ; color_id = 1 (window frame)
    SYSCALL 2            ; SYS_RECT_ID

    ; Draw title bar
    IMM32 R1, 200
    IMM32 R2, 150
    IMM32 R3, 400
    IMM32 R4, 40
    IMM32 R5, 2          ; color_id = 2 (title bar)
    SYSCALL 2

    ; Display text
    IMM32 R1, 220        ; x
    IMM32 R2, 160        ; y
    IMM32 R3, 4          ; color_id = 4 (white)
    IMM32 R4, 2          ; message_id = 2 ("PXVM ready")
    SYSCALL 3            ; SYS_TEXT_ID

    ; Print completion message
    IMM32 R1, 3          ; message_id = 3 ("Task complete")
    SYSCALL 1

DONE:
    HALT
