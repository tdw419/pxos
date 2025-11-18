; boot_32bit.asm — Minimal 16→32 bootloader
; Loads microkernel at 0x10000 and jumps to it in 32-bit mode
BITS 16
ORG 0x7C00

start:
    cli
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00
    sti

    ; Save boot drive
    mov [boot_drive], dl

    ; 'R' – Real mode init
    mov byte [0xB8000], 'R'
    mov byte [0xB8001], 0x0F

    ; Enable A20
    in  al, 0x92
    or  al, 2
    out 0x92, al

    ; 'A' – A20 enabled
    mov byte [0xB8002], 'A'
    mov byte [0xB8003], 0x0F

    ; Load microkernel (32 sectors = 16KB at 0x10000)
    mov ax, 0x1000
    mov es, ax
    xor bx, bx            ; ES:BX = 1000:0000 → 0x10000

    mov ah, 0x02          ; Read sectors
    mov al, 32            ; Read 32 sectors
    mov ch, 0             ; Cylinder 0
    mov cl, 2             ; Sector 2
    mov dh, 0             ; Head 0
    mov dl, [boot_drive]
    int 0x13
    jc  .disk_error

    ; 'D' – Disk OK
    mov byte [0xB8004], 'D'
    mov byte [0xB8005], 0x0F

    ; Load GDT
    lgdt [gdt_descriptor]

    ; Enter Protected Mode
    mov eax, cr0
    or  eax, 1
    mov cr0, eax

    ; Far jump to 32-bit code
    jmp 0x08:start32

.disk_error:
    mov byte [0xB8000], 'E'
    mov byte [0xB8001], 0x4F
    cli
    hlt

; 32-bit Protected Mode
BITS 32
start32:
    ; Setup segments
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    mov esp, 0x90000

    ; 'P' – Protected Mode
    mov byte [0xB8006], 'P'
    mov byte [0xB8007], 0x0F

    ; '3' – 32-bit ready
    mov byte [0xB8008], '3'
    mov byte [0xB8009], 0x0F

    ; 'K' – Jumping to kernel
    mov byte [0xB800A], 'K'
    mov byte [0xB800B], 0x0F

    ; Jump to microkernel at 0x10000
    jmp 0x08:0x10000

; GDT
BITS 16
align 8
gdt_start:
    dq 0                    ; Null
    dq 0x00CF9A000000FFFF   ; 32-bit Code (0x08)
    dq 0x00CF92000000FFFF   ; 32-bit Data (0x10)
gdt_end:

gdt_descriptor:
    dw gdt_end - gdt_start - 1
    dd gdt_start

boot_drive: db 0

times 510-($-$$) db 0
dw 0xAA55
