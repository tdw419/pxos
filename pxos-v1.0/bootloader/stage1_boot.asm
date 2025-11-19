; pxOS Bootloader - Stage 1 (Boot Sector)
; Minimal 512-byte boot sector that loads stage 2
;
; Memory layout:
;   0x7C00: This boot sector (512 bytes)
;   0x7E00: Stage 2 bootloader (loaded here)
;
; Design philosophy: Keep stage 1 minimal, do real work in stage 2

BITS 16
ORG 0x7C00

start:
    ; Disable interrupts during setup
    cli

    ; Set up segments (DS, ES, SS all to 0)
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax

    ; Set up stack (grows down from 0x7C00)
    mov sp, 0x7C00

    ; Enable interrupts
    sti

    ; Save boot drive number (BIOS passes in DL)
    mov [boot_drive], dl

    ; Print boot message
    mov si, msg_boot
    call print

    ; Load stage 2 from disk
    ; Stage 2 starts at sector 2 (sector 1 is this boot sector)
    mov bx, 0x7E00          ; Load address for stage 2
    mov dh, 8               ; Number of sectors to load (stage 2 = 4KB = 8 sectors)
    mov dl, [boot_drive]    ; Boot drive
    call load_sectors

    ; Check if load succeeded
    jc disk_error

    ; Print success message
    mov si, msg_loaded
    call print

    ; Jump to stage 2
    jmp 0x0000:0x7E00

disk_error:
    mov si, msg_disk_error
    call print
    jmp halt

halt:
    cli
    hlt
    jmp halt

;-----------------------------------------------------------------------------
; print - Print null-terminated string
; Input: SI = pointer to string
;-----------------------------------------------------------------------------
print:
    pusha
.loop:
    lodsb                   ; Load byte from [SI] into AL, increment SI
    test al, al             ; Check if null terminator
    jz .done
    mov ah, 0x0E            ; BIOS teletype output
    mov bh, 0               ; Page 0
    int 0x10                ; BIOS video services
    jmp .loop
.done:
    popa
    ret

;-----------------------------------------------------------------------------
; load_sectors - Load sectors from disk using BIOS int 0x13
; Input:
;   BX = destination address
;   DH = number of sectors to load
;   DL = drive number
;-----------------------------------------------------------------------------
load_sectors:
    pusha

    ; Set up disk read parameters
    mov ah, 0x02            ; BIOS read sectors function
    mov al, dh              ; Number of sectors to read
    mov ch, 0               ; Cylinder 0
    mov cl, 2               ; Start from sector 2 (sector 1 is boot sector)
    mov dh, 0               ; Head 0
    ; DL already contains drive number

    ; ES:BX already points to destination
    int 0x13                ; BIOS disk services

    ; CF is set on error
    jc .error

    popa
    clc                     ; Clear carry flag (success)
    ret

.error:
    popa
    stc                     ; Set carry flag (error)
    ret

;-----------------------------------------------------------------------------
; Data
;-----------------------------------------------------------------------------
boot_drive:     db 0

msg_boot:       db "pxOS Bootloader v1.0", 13, 10, 0
msg_loaded:     db "Stage 2 loaded, transferring control...", 13, 10, 0
msg_disk_error: db "DISK ERROR! Cannot load stage 2.", 13, 10, 0

;-----------------------------------------------------------------------------
; Boot sector padding and signature
;-----------------------------------------------------------------------------
; Pad with zeros to byte 510
times 510-($-$$) db 0

; Boot sector signature (required by BIOS)
dw 0xAA55
