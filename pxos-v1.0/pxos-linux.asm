; pxos-linux.asm - Enhanced pxOS Bootloader with Linux Chain-Loading
; Compiles to a 512-byte boot sector that loads and boots a Linux kernel
; Build: nasm -f bin pxos-linux.asm -o pxos-linux.bin

[BITS 16]
[ORG 0x7C00]

; === Constants ===
%define KERNEL_LOAD_SEG   0x1000   ; Load kernel at 0x10000 (64KB)
%define KERNEL_SECTORS    50       ; Number of sectors to read (25KB)
%define BOOT_PARAMS_ADDR  0x90000  ; Linux boot parameters location
%define STACK_SEGMENT     0x9000   ; Stack segment
%define STACK_POINTER     0xFFFF   ; Stack pointer

; === Entry Point ===
start:
    ; Disable interrupts during setup
    cli

    ; Set up segments
    xor ax, ax
    mov ds, ax
    mov es, ax

    ; Set up stack
    mov ax, STACK_SEGMENT
    mov ss, ax
    mov sp, STACK_POINTER

    ; Enable interrupts
    sti

    ; Save boot drive (BIOS sets DL to boot drive)
    mov [boot_drive], dl

    ; Clear screen
    mov ah, 0x00
    mov al, 0x03        ; 80x25 text mode
    int 0x10

    ; Print boot message
    mov si, msg_boot
    call print_string

    ; Load Linux kernel from disk
    call load_kernel

    ; Set up Linux boot parameters
    call setup_boot_params

    ; Switch to protected mode
    call enter_protected_mode

    ; Should never reach here
    jmp $

; === Load Kernel from Disk ===
load_kernel:
    mov si, msg_loading
    call print_string

    ; Set up destination: ES:BX = 0x1000:0x0000
    mov ax, KERNEL_LOAD_SEG
    mov es, ax
    xor bx, bx

    ; Read kernel sectors
    mov ah, 0x02            ; BIOS function: read sectors
    mov al, KERNEL_SECTORS  ; Number of sectors
    mov ch, 0x00            ; Cylinder 0
    mov cl, 0x02            ; Start at sector 2 (after boot sector)
    mov dh, 0x00            ; Head 0
    mov dl, [boot_drive]    ; Boot drive

    int 0x13                ; Call BIOS
    jc .disk_error          ; Jump if error (carry flag set)

    ; Success
    mov si, msg_ok
    call print_string
    ret

.disk_error:
    mov si, msg_disk_error
    call print_string
    jmp $                   ; Halt

; === Set Up Linux Boot Parameters ===
setup_boot_params:
    mov si, msg_params
    call print_string

    ; Point ES to boot parameters area
    mov ax, BOOT_PARAMS_ADDR >> 4
    mov es, ax
    xor di, di

    ; Zero out boot parameters (512 bytes)
    mov cx, 256
    xor ax, ax
    rep stosw

    ; Set Linux boot protocol header
    mov di, 0
    mov dword [es:di + 0x202], 0x53726448  ; "HdrS" signature
    mov word [es:di + 0x206], 0x0206       ; Protocol version 2.06
    mov byte [es:di + 0x210], 0xE1         ; Loader type
    mov byte [es:di + 0x211], 0x81         ; Heap end pointer
    mov word [es:di + 0x224], 0xFFFF       ; Heap end
    mov byte [es:di + 0x226], 0x00         ; Extended loader type
    mov dword [es:di + 0x228], 0x1000000   ; Command line pointer

    mov si, msg_ok
    call print_string
    ret

; === Enter Protected Mode ===
enter_protected_mode:
    mov si, msg_protected
    call print_string

    cli                     ; Disable interrupts

    ; Load GDT
    lgdt [gdt_descriptor]

    ; Enable protected mode (set PE bit in CR0)
    mov eax, cr0
    or al, 1
    mov cr0, eax

    ; Far jump to flush pipeline and enter 32-bit code
    jmp CODE_SEG:protected_mode_entry

; === Protected Mode Entry (32-bit) ===
[BITS 32]
protected_mode_entry:
    ; Set up data segments
    mov ax, DATA_SEG
    mov ds, ax
    mov ss, ax
    mov es, ax
    mov fs, ax
    mov gs, ax

    ; Set up stack in protected mode
    mov ebp, 0x90000
    mov esp, ebp

    ; Jump to Linux kernel entry point
    ; The kernel is loaded at 0x10000, but entry is typically at 0x100000
    ; For a bzImage, we need to jump to the 32-bit entry point
    jmp KERNEL_LOAD_SEG:0x0000

; === Helper Functions (16-bit) ===
[BITS 16]

; Print null-terminated string (DS:SI)
print_string:
    pusha
    mov ah, 0x0E        ; BIOS teletype function
.loop:
    lodsb               ; Load byte from DS:SI into AL
    test al, al         ; Check if null terminator
    jz .done
    int 0x10            ; Print character
    jmp .loop
.done:
    popa
    ret

; === Data Section ===
boot_drive:     db 0

; Messages
msg_boot:       db "pxOS Linux Loader v1.0", 0x0D, 0x0A, 0
msg_loading:    db "Loading kernel... ", 0
msg_params:     db "Setup boot params... ", 0
msg_protected:  db "Entering protected mode...", 0x0D, 0x0A, 0
msg_ok:         db "OK", 0x0D, 0x0A, 0
msg_disk_error: db "DISK ERROR!", 0x0D, 0x0A, 0

; === Global Descriptor Table (GDT) ===
gdt_start:
    ; Null descriptor (required)
    dq 0x0000000000000000

gdt_code:
    ; Code segment descriptor
    dw 0xFFFF           ; Limit (bits 0-15)
    dw 0x0000           ; Base (bits 0-15)
    db 0x00             ; Base (bits 16-23)
    db 10011010b        ; Access byte (present, ring 0, code, executable, readable)
    db 11001111b        ; Flags (granularity, 32-bit) + Limit (bits 16-19)
    db 0x00             ; Base (bits 24-31)

gdt_data:
    ; Data segment descriptor
    dw 0xFFFF           ; Limit (bits 0-15)
    dw 0x0000           ; Base (bits 0-15)
    db 0x00             ; Base (bits 16-23)
    db 10010010b        ; Access byte (present, ring 0, data, writable)
    db 11001111b        ; Flags (granularity, 32-bit) + Limit (bits 16-19)
    db 0x00             ; Base (bits 24-31)

gdt_end:

; GDT descriptor
gdt_descriptor:
    dw gdt_end - gdt_start - 1  ; Size of GDT - 1
    dd gdt_start                 ; GDT address

; Segment selectors
CODE_SEG equ gdt_code - gdt_start
DATA_SEG equ gdt_data - gdt_start

; === Boot Signature ===
times 510-($-$$) db 0
dw 0xAA55
