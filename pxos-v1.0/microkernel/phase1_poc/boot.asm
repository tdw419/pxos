; pxOS Phase 1 Bootloader
; Minimal boot: Real mode → Protected mode → Long mode → Microkernel
; Target size: 512 bytes
; Based on hypervisor/pxhv_boot.asm but simplified

BITS 16
ORG 0x7C00

; Entry point
start:
    ; Setup segments
    cli
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00              ; Stack grows down from boot sector
    sti

    ; Save boot drive
    mov [boot_drive], dl

    ; Print banner
    mov si, msg_boot
    call print_string

    ; Enable A20 line (for >1MB memory access)
    call enable_a20

    ; Load microkernel from disk
    mov si, msg_loading
    call print_string

    ; Read 16 sectors (8KB microkernel) from disk
    mov ah, 0x02                ; Read sectors
    mov al, 16                  ; Number of sectors
    mov ch, 0                   ; Cylinder 0
    mov cl, 2                   ; Start from sector 2
    mov dh, 0                   ; Head 0
    mov dl, [boot_drive]        ; Drive number
    mov bx, 0x1000              ; Load at 0x0000:0x1000
    int 0x13
    jc disk_error

    ; Setup GDT
    lgdt [gdt_descriptor]

    ; Enter protected mode
    mov eax, cr0
    or eax, 1
    mov cr0, eax

    ; Far jump to 32-bit code
    jmp 0x08:start32

disk_error:
    mov si, msg_disk_error
    call print_string
    cli
    hlt

;-----------------------------------------------------------------------------
; print_string: Print null-terminated string (16-bit real mode)
; Input: SI = pointer to string
;-----------------------------------------------------------------------------
print_string:
    pusha
.loop:
    lodsb
    test al, al
    jz .done
    mov ah, 0x0E
    mov bx, 0x0007              ; Page 0, white on black
    int 0x10
    jmp .loop
.done:
    popa
    ret

;-----------------------------------------------------------------------------
; enable_a20: Enable A20 line for accessing >1MB memory
;-----------------------------------------------------------------------------
enable_a20:
    pusha

    ; Fast A20 enable via port 0x92
    in al, 0x92
    test al, 2
    jnz .done                   ; Already enabled
    or al, 2
    out 0x92, al

.done:
    popa
    ret

;-----------------------------------------------------------------------------
; 32-bit Protected Mode Code
;-----------------------------------------------------------------------------
BITS 32
start32:
    ; Setup segments for 32-bit mode
    mov ax, 0x10                ; Data segment selector
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    mov esp, 0x90000            ; Stack at 640KB

    ; Debug: Write 'P' to VGA (Protected mode reached)
    mov byte [0xB8000], 'P'
    mov byte [0xB8001], 0x0F

    ; Transition to 64-bit long mode
    ; Enable PAE (Physical Address Extension)
    mov eax, cr4
    or eax, 0x20                ; Set PAE bit
    mov cr4, eax

    ; Setup minimal paging (identity map first 1GB)
    call setup_paging_32

    ; Load CR3 with PML4 address
    mov eax, 0x2000             ; PML4 at 0x2000
    mov cr3, eax

    ; Enable long mode (set EFER.LME)
    mov ecx, 0xC0000080         ; EFER MSR
    rdmsr
    or eax, 0x100               ; Set LME bit
    wrmsr

    ; Debug: Write 'G' to VGA (Paging about to be enabled)
    mov byte [0xB8002], 'G'
    mov byte [0xB8003], 0x0F

    ; Enable paging (activates long mode)
    mov eax, cr0
    or eax, 0x80000000          ; Set PG bit
    mov cr0, eax

    ; Debug: Write 'L' to VGA (Long mode enabled!)
    mov byte [0xB8004], 'L'
    mov byte [0xB8005], 0x0F

    ; Use far return to jump to 64-bit code (safer than far jmp)
    push 0x08                   ; Code segment selector
    push start64                ; Return address
    retf                        ; Far return to 64-bit code

;-----------------------------------------------------------------------------
; setup_paging_32: Setup minimal page tables for long mode
; Identity maps first 1GB using 2MB pages
;-----------------------------------------------------------------------------
setup_paging_32:
    push eax
    push ecx
    push edi

    ; Clear page table memory (16KB total)
    mov edi, 0x2000
    mov ecx, 4096               ; 16KB / 4 = 4096 dwords
    xor eax, eax
    rep stosd

    ; Setup PML4 (Page Map Level 4)
    mov edi, 0x2000
    mov dword [edi], 0x3003     ; PML4[0] -> PDPT at 0x3000 (present, writable)

    ; Setup PDPT (Page Directory Pointer Table)
    mov edi, 0x3000
    mov dword [edi], 0x4003     ; PDPT[0] -> PD at 0x4000 (present, writable)

    ; Setup PD (Page Directory) with 2MB pages
    mov edi, 0x4000
    mov eax, 0x83               ; Present, writable, 2MB page
    mov ecx, 512                ; 512 entries * 2MB = 1GB

.pd_loop:
    mov [edi], eax
    add eax, 0x200000           ; Next 2MB
    add edi, 8
    loop .pd_loop

    pop edi
    pop ecx
    pop eax
    ret

;-----------------------------------------------------------------------------
; 64-bit Long Mode Code
;-----------------------------------------------------------------------------
BITS 64
start64:
    ; Debug: Write '6' to VGA (64-bit mode reached!)
    mov byte [0xB8006], '6'
    mov byte [0xB8007], 0x0F

    ; Setup segments for 64-bit mode
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    ; Setup stack
    mov rsp, 0x90000

    ; Debug: Write 'M' to VGA (About to jump to microkernel)
    mov byte [0xB8008], 'M'
    mov byte [0xB8009], 0x0F

    ; Jump to microkernel at 0x1000
    jmp 0x1000

;-----------------------------------------------------------------------------
; Data Section
;-----------------------------------------------------------------------------
BITS 16
msg_boot:       db 'pxOS Phase 1 Boot', 13, 10, 0
msg_loading:    db 'Loading microkernel...', 13, 10, 0
msg_disk_error: db 'Disk error!', 13, 10, 0

boot_drive:     db 0

; GDT (Global Descriptor Table)
align 8
gdt_start:
    ; Null descriptor
    dq 0

    ; Code segment (32/64-bit)
    dw 0xFFFF                   ; Limit low
    dw 0x0000                   ; Base low
    db 0x00                     ; Base middle
    db 10011010b                ; Access: present, ring 0, code, executable, readable
    db 10101111b                ; Flags + Limit high: 4KB granularity, LONG MODE (L=1, D=0)
    db 0x00                     ; Base high

    ; Data segment
    dw 0xFFFF                   ; Limit low
    dw 0x0000                   ; Base low
    db 0x00                     ; Base middle
    db 10010010b                ; Access: present, ring 0, data, writable
    db 11001111b                ; Flags + Limit high
    db 0x00                     ; Base high

gdt_end:

gdt_descriptor:
    dw gdt_end - gdt_start - 1  ; Size
    dd gdt_start                ; Offset

; Boot signature
times 510-($-$$) db 0
dw 0xAA55
