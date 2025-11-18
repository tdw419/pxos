; Minimal proven 64-bit bootloader
; Based on osdev.org standard long mode boot sequence
BITS 16
ORG 0x7C00

start:
    ; Clear interrupts and set up segments
    cli
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00

    ; Save boot drive
    mov [boot_drive], dl

    ; Load microkernel to 0x10000 (64KB)
    mov ah, 0x02                ; Read sectors
    mov al, 4                   ; 4 sectors (2KB)
    mov ch, 0                   ; Cylinder 0
    mov cl, 2                   ; Sector 2
    mov dh, 0                   ; Head 0
    mov dl, [boot_drive]
    mov bx, 0x1000              ; ES:BX = 0x1000
    mov es, bx
    xor bx, bx                  ; 0x1000:0x0000 = 0x10000
    int 0x13

    ; Reset ES
    xor ax, ax
    mov es, ax

    ; Enable A20
    in al, 0x92
    or al, 2
    out 0x92, al

    ; Load GDT
    lgdt [gdt.pointer]

    ; Enter protected mode
    mov eax, cr0
    or al, 1
    mov cr0, eax

    jmp 0x08:protected_mode

BITS 32
protected_mode:
    ; Set up segment registers
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    mov esp, 0x90000

    ; Build page tables
    ; PML4[0] -> PDPT
    mov edi, 0x1000
    mov cr3, edi
    xor eax, eax
    mov ecx, 4096
    rep stosd
    mov edi, cr3

    mov dword [edi], 0x2003      ; PML4[0] -> 0x2000 (PDPT)
    mov dword [0x2000], 0x3003   ; PDPT[0] -> 0x3000 (PD)
    mov dword [0x3000], 0x4003   ; PD[0] -> 0x4000 (PT)

    ; Identity map first 2MB with 4KB pages
    mov edi, 0x4000
    mov eax, 0x03
    mov ecx, 512
.build_pt:
    mov [edi], eax
    add eax, 0x1000
    add edi, 8
    loop .build_pt

    ; Enable PAE
    mov eax, cr4
    or eax, 1 << 5
    mov cr4, eax

    ; Enable long mode
    mov ecx, 0xC0000080
    rdmsr
    or eax, 1 << 8
    wrmsr

    ; Enable paging
    mov eax, cr0
    or eax, 1 << 31
    mov cr0, eax

    ; Jump to 64-bit code
    jmp 0x18:long_mode

BITS 64
long_mode:
    ; Clear segment registers
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    ; Jump to microkernel at 0x10000
    jmp 0x10000

; GDT
gdt:
    dq 0                                  ; Null descriptor
    dq 0x00CF9A000000FFFF                 ; 32-bit code
    dq 0x00CF92000000FFFF                 ; 32-bit data
    dq 0x00AF9A000000FFFF                 ; 64-bit code
    dq 0x00AF92000000FFFF                 ; 64-bit data
.pointer:
    dw $ - gdt - 1
    dd gdt

boot_drive: db 0

times 510-($-$$) db 0
dw 0xAA55
