; pxOS Bootloader - Stage 2
; Main bootloader that sets up environment and loads kernel
;
; Loaded at 0x7E00 by stage 1
; Flow:
;   1. Check CPU capabilities (64-bit support)
;   2. Enable A20 line
;   3. Detect GPU and read BAR0
;   4. Set up page tables (including BAR0 mapping)
;   5. Enter protected mode
;   6. Switch to long mode
;   7. Load kernel from disk
;   8. Jump to kernel with GPU info

[BITS 16]
[ORG 0x7E00]

start:
    ; Set up segments
    cli
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00
    sti

    ; Print stage 2 banner
    mov si, msg_stage2
    call print16
    call newline

    ;-------------------------------------------------------------------------
    ; Check if CPU supports 64-bit long mode
    ;-------------------------------------------------------------------------
    mov si, msg_check_cpu
    call print16

    call check_long_mode
    jc .no_long_mode

    mov si, msg_ok
    call print16
    call newline

    ;-------------------------------------------------------------------------
    ; Enable A20 line
    ;-------------------------------------------------------------------------
    mov si, msg_enable_a20
    call print16

    call enable_a20

    mov si, msg_ok
    call print16
    call newline

    ;-------------------------------------------------------------------------
    ; Detect GPU
    ;-------------------------------------------------------------------------
    call detect_gpu
    jc .halt                ; Halt if no GPU (required for pxOS)

    ;-------------------------------------------------------------------------
    ; Set up page tables with BAR0 mapping
    ;-------------------------------------------------------------------------
    call setup_page_tables

    ;-------------------------------------------------------------------------
    ; Load GDT for protected mode
    ;-------------------------------------------------------------------------
    mov si, msg_load_gdt
    call print16

    lgdt [gdt_descriptor]

    mov si, msg_ok
    call print16
    call newline

    ;-------------------------------------------------------------------------
    ; Enter protected mode
    ;-------------------------------------------------------------------------
    mov si, msg_protected_mode
    call print16

    ; Set PE bit in CR0
    mov eax, cr0
    or eax, 1
    mov cr0, eax

    ; Far jump to flush pipeline and enter protected mode
    jmp 0x08:protected_mode_entry

.no_long_mode:
    mov si, msg_no_64bit
    call print16
    jmp .halt

.halt:
    cli
    hlt
    jmp .halt

;-----------------------------------------------------------------------------
; 32-bit Protected Mode Code
;-----------------------------------------------------------------------------
[BITS 32]

protected_mode_entry:
    ; Set up segment registers
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    ; Enable paging (which will activate long mode)
    call enable_paging

    ; Load 64-bit GDT
    lgdt [gdt64_descriptor]

    ; Jump to 64-bit code
    jmp 0x08:long_mode_entry

;-----------------------------------------------------------------------------
; 64-bit Long Mode Code
;-----------------------------------------------------------------------------
[BITS 64]

long_mode_entry:
    ; Set up segment registers for 64-bit
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    ; Set up a minimal stack
    mov rsp, 0x7C00

    ; We're now in 64-bit mode!
    ; Print success message to VGA (direct write)
    mov edi, 0xB8000
    mov rax, 0x0F36307840530F36 ; '64OS'
    mov [edi], rax
    mov rax, 0x0F4D0F520F410F42 ; 'BAR '
    mov [edi + 8], rax

    ; Pass GPU information to kernel (when we load it)
    ; For now, just halt successfully
    mov rbx, [gpu_bar0]     ; Pass BAR0 address in RBX

.halt64:
    hlt
    jmp .halt64

;-----------------------------------------------------------------------------
; Include other modules
;-----------------------------------------------------------------------------
%include "utils.asm"
%include "gpu_detect.asm"
%include "paging.asm"

;-----------------------------------------------------------------------------
; GDT for Protected Mode and Long Mode
;-----------------------------------------------------------------------------
[BITS 16]

align 16
gdt_start:
    ; Null descriptor
    dq 0

    ; Code segment (32-bit)
    dw 0xFFFF       ; Limit low
    dw 0            ; Base low
    db 0            ; Base middle
    db 10011010b    ; Access: present, ring 0, code, executable, readable
    db 11001111b    ; Flags: 4KB granularity, 32-bit
    db 0            ; Base high

    ; Data segment (32-bit)
    dw 0xFFFF
    dw 0
    db 0
    db 10010010b    ; Access: present, ring 0, data, writable
    db 11001111b
    db 0

gdt_end:

gdt_descriptor:
    dw gdt_end - gdt_start - 1
    dd gdt_start

; 64-bit GDT
align 16
gdt64_start:
    ; Null descriptor
    dq 0

    ; Code segment (64-bit)
    dw 0xFFFF
    dw 0
    db 0
    db 10011010b    ; Access: present, ring 0, code
    db 10101111b    ; Flags: 64-bit, granularity
    db 0

    ; Data segment (64-bit)
    dw 0xFFFF
    dw 0
    db 0
    db 10010010b
    db 00000000b
    db 0

gdt64_end:

gdt64_descriptor:
    dw gdt64_end - gdt64_start - 1
    dd gdt64_start

;-----------------------------------------------------------------------------
; Messages
;-----------------------------------------------------------------------------
msg_stage2:         db "pxOS Stage 2 Bootloader", 0
msg_check_cpu:      db "Checking CPU... ", 0
msg_enable_a20:     db "Enabling A20... ", 0
msg_load_gdt:       db "Loading GDT... ", 0
msg_protected_mode: db "Entering protected mode... ", 0
msg_ok:             db "OK", 0
msg_no_64bit:       db "ERROR: 64-bit mode not supported!", 13, 10, 0

;-----------------------------------------------------------------------------
; Pad to 4KB (8 sectors)
;-----------------------------------------------------------------------------
times 4096-($-$$) db 0
