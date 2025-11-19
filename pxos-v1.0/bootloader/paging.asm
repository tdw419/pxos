; pxOS Bootloader - Page Table Setup
; Creates 4-level page tables with GPU BAR0 pre-mapped

BITS 16

; Page table locations in memory
PML4_ADDR equ 0x1000
PDP_ADDR  equ 0x2000
PD_ADDR   equ 0x3000
PT_ADDR   equ 0x4000

;-----------------------------------------------------------------------------
; setup_page_tables - Create page tables with BAR0 mapping
; Uses gpu_bar0 from gpu_detect.asm to map GPU memory
;-----------------------------------------------------------------------------
setup_page_tables:
    pusha

    mov si, msg_setup_paging
    call print16

    ; Clear page table area (16KB: PML4 + PDP + PD + PT)
    mov edi, PML4_ADDR
    mov ecx, 4096           ; 4 pages * 1024 dwords
    xor eax, eax
    rep stosd

    ;-------------------------------------------------------------------------
    ; Set up PML4 (Page Map Level 4)
    ;-------------------------------------------------------------------------
    mov edi, PML4_ADDR
    mov eax, PDP_ADDR
    or eax, 0x03            ; Present + Writable
    mov [edi], eax          ; PML4[0] -> PDP

    ;-------------------------------------------------------------------------
    ; Set up PDP (Page Directory Pointer)
    ;-------------------------------------------------------------------------
    mov edi, PDP_ADDR
    mov eax, PD_ADDR
    or eax, 0x03            ; Present + Writable
    mov [edi], eax          ; PDP[0] -> PD

    ; For BAR0 (if it's in a different PDP entry), we'll handle it below

    ;-------------------------------------------------------------------------
    ; Set up PD (Page Directory)
    ; Identity map first 2MB (for kernel code)
    ;-------------------------------------------------------------------------
    mov edi, PD_ADDR
    mov eax, 0x00000083     ; 0MB, Present + Writable + Page Size (2MB)
    mov [edi], eax          ; PD[0] maps first 2MB

    ;-------------------------------------------------------------------------
    ; Map GPU BAR0
    ; Calculate which PD entry we need based on BAR0 address
    ;-------------------------------------------------------------------------
    mov eax, [gpu_bar0]
    test eax, eax
    jz .skip_bar0           ; Skip if no BAR0

    ; Calculate PD index: (BAR0 >> 21) & 0x1FF
    mov ebx, eax
    shr ebx, 21
    and ebx, 0x1FF

    ; Align BAR0 to 2MB boundary and set flags
    and eax, 0xFFE00000     ; Mask to 2MB boundary
    or eax, 0x0B            ; Present + Writable + PCD (cache disable for MMIO)

    ; Install PD entry for BAR0
    mov edi, PD_ADDR
    mov [edi + ebx * 8], eax

    ; Print BAR0 mapping info
    mov si, msg_mapped_bar0
    call print16
    mov eax, [gpu_bar0]
    call print_hex32
    mov si, msg_at_pd
    call print16
    movzx eax, bx
    call print_hex16
    call newline

.skip_bar0:

    ;-------------------------------------------------------------------------
    ; Load CR3 with PML4 address
    ;-------------------------------------------------------------------------
    mov eax, PML4_ADDR
    mov cr3, eax

    mov si, msg_paging_done
    call print16

    popa
    ret

;-----------------------------------------------------------------------------
; enable_paging - Enable PAE and paging
;-----------------------------------------------------------------------------
enable_paging:
    pusha

    ; Enable PAE (Physical Address Extension)
    mov eax, cr4
    or eax, 1 << 5          ; Set PAE bit
    mov cr4, eax

    ; Set long mode bit in EFER MSR
    mov ecx, 0xC0000080     ; EFER MSR
    rdmsr
    or eax, 1 << 8          ; Set LM bit
    wrmsr

    ; Enable paging
    mov eax, cr0
    or eax, 1 << 31         ; Set PG bit
    mov cr0, eax

    popa
    ret

;-----------------------------------------------------------------------------
; Data
;-----------------------------------------------------------------------------
msg_setup_paging: db "Setting up page tables... ", 0
msg_paging_done:  db "Done!", 13, 10, 0
msg_mapped_bar0:  db "  Mapped BAR0 ", 0
msg_at_pd:        db " at PD[", 0
