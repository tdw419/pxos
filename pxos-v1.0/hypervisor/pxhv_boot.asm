; pxHV Boot Sector - Stage 1
; Loads hypervisor from disk and transfers control
;
; Memory layout:
;   0x7C00: This boot sector
;   0x8000: Stack
;   0x10000: Stage 2 hypervisor loader (loaded here)
;
; Compile: nasm -f bin -o pxhv_boot.bin pxhv_boot.asm

BITS 16
ORG 0x7C00

; Constants
STAGE2_SEGMENT equ 0x1000    ; Load stage 2 at 0x10000
STAGE2_OFFSET  equ 0x0000
STAGE2_SECTORS equ 40        ; Load 40 sectors (20KB) for stage 2

start:
    ; Setup segments
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00           ; Stack grows down from boot sector

    ; Save boot drive
    mov [boot_drive], dl

    ; Print boot message
    mov si, msg_boot
    call print_string

    ; Load stage 2 from disk
    call load_stage2

    ; Enable A20 line (required for >1MB access)
    call enable_a20

    ; Print jump message
    mov si, msg_jump
    call print_string

    ; Jump to stage 2
    jmp STAGE2_SEGMENT:STAGE2_OFFSET

;-----------------------------------------------------------------------------
; load_stage2: Load hypervisor from disk
;-----------------------------------------------------------------------------
load_stage2:
    pusha

    ; Setup disk read
    mov ah, 0x02                ; BIOS read sectors function
    mov al, STAGE2_SECTORS      ; Number of sectors
    mov ch, 0                   ; Cylinder 0
    mov cl, 2                   ; Sector 2 (sector 1 is boot sector)
    mov dh, 0                   ; Head 0
    mov dl, [boot_drive]        ; Drive number

    ; Destination: ES:BX = 0x1000:0x0000 = 0x10000
    mov bx, STAGE2_SEGMENT
    mov es, bx
    mov bx, STAGE2_OFFSET

    int 0x13                    ; BIOS disk interrupt
    jc disk_error               ; CF set on error

    ; Verify we read the correct number of sectors
    cmp al, STAGE2_SECTORS
    jne disk_error

    popa
    ret

disk_error:
    mov si, msg_disk_error
    call print_string
    cli
    hlt

;-----------------------------------------------------------------------------
; enable_a20: Enable A20 line for >1MB memory access
;-----------------------------------------------------------------------------
enable_a20:
    pusha

    ; Method 1: Fast A20 gate (most modern)
    in al, 0x92
    or al, 2
    out 0x92, al

    ; Method 2: Keyboard controller (fallback)
    call a20_wait
    mov al, 0xAD                ; Disable keyboard
    out 0x64, al

    call a20_wait
    mov al, 0xD0                ; Read output port
    out 0x64, al

    call a20_wait2
    in al, 0x60                 ; Read current state
    push ax

    call a20_wait
    mov al, 0xD1                ; Write output port
    out 0x64, al

    call a20_wait
    pop ax
    or al, 2                    ; Set A20 bit
    out 0x60, al

    call a20_wait
    mov al, 0xAE                ; Enable keyboard
    out 0x64, al

    call a20_wait

    popa
    ret

a20_wait:
    in al, 0x64
    test al, 2
    jnz a20_wait
    ret

a20_wait2:
    in al, 0x64
    test al, 1
    jz a20_wait2
    ret

;-----------------------------------------------------------------------------
; print_string: Print null-terminated string
; Input: SI = string address
;-----------------------------------------------------------------------------
print_string:
    pusha
    mov ah, 0x0E                ; BIOS teletype output
.loop:
    lodsb                       ; Load byte from [SI] into AL
    test al, al
    jz .done
    int 0x10                    ; Print character
    jmp .loop
.done:
    popa
    ret

;-----------------------------------------------------------------------------
; Data
;-----------------------------------------------------------------------------
boot_drive:     db 0
msg_boot:       db 'pxHV: Pixel Hypervisor v0.1', 13, 10, 0
msg_jump:       db 'Jumping to hypervisor...', 13, 10, 0
msg_disk_error: db 'DISK ERROR!', 13, 10, 0

;-----------------------------------------------------------------------------
; Boot signature
;-----------------------------------------------------------------------------
times 510-($-$$) db 0
dw 0xAA55
