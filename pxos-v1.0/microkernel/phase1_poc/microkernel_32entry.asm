; pxOS Microkernel v0.3 - 32-bit Entry Point Version
; Bootloader loads us at 0x10000 in 32-bit protected mode
; We handle the 32→64 transition

BITS 32
ORG 0x10000

; Entry from bootloader (32-bit mode)
kernel_entry_32:
    ; VGA debug: 'M' - Microkernel reached
    mov byte [0xB800C], 'M'
    mov byte [0xB800D], 0x0F

    ; Setup page tables
    call setup_page_tables_32

    ; VGA debug: 'T' - Page tables ready
    mov byte [0xB800E], 'T'
    mov byte [0xB800F], 0x0F

    ; Enable PAE
    mov eax, cr4
    or eax, 0x20
    mov cr4, eax

    ; Load CR3
    mov eax, pml4_table
    mov cr3, eax

    ; Enable Long Mode
    mov ecx, 0xC0000080   ; EFER MSR
    rdmsr
    or eax, 0x100         ; LME bit
    wrmsr

    ; Enable Paging
    mov eax, cr0
    or eax, 0x80000000    ; PG bit
    mov cr0, eax

    ; Load 64-bit GDT
    lgdt [gdt64_descriptor]

    ; Far jump to 64-bit code
    jmp 0x08:kernel_entry_64

setup_page_tables_32:
    ; Clear tables
    mov edi, pml4_table
    mov ecx, 4096 * 3 / 4  ; 3 tables × 4KB / 4 bytes
    xor eax, eax
    cld
    rep stosd

    ; PML4[0] → PDPT
    mov eax, pdpt_table
    or eax, 0x3
    mov [pml4_table], eax

    ; PDPT[0] → PD
    mov eax, pd_table
    or eax, 0x3
    mov [pdpt_table], eax

    ; PD entries (512 × 2MB = 1GB)
    mov edi, pd_table
    mov eax, 0x83         ; 2MB page, present, writable
    mov ecx, 512
.loop:
    mov [edi], eax
    add eax, 0x200000
    add edi, 8
    loop .loop

    ret

BITS 64
kernel_entry_64:
    ; VGA debug: '6' - 64-bit mode
    mov byte [0xB8010], '6'
    mov byte [0xB8011], 0x0F

    ; Setup 64-bit segments
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov rsp, 0x90000

    ; VGA debug: 'G' - Going to main
    mov byte [0xB8012], 'G'
    mov byte [0xB8013], 0x0F

    ; Continue with microkernel main
    call microkernel_main

    ; Hang if we return
    cli
.hang:
    hlt
    jmp .hang

; ---------------------------------------------------------------------------
; Microkernel Main (64-bit)
; ---------------------------------------------------------------------------
MAILBOX_ADDR equ 0x20000
UART_PORT    equ 0x3F8

OP_MMIO_WRITE_UART  equ 0x80
OP_CPU_HALT         equ 0x8F

microkernel_main:
    ; Print banner
    mov rsi, msg_microkernel
    call print_string

    ; Initialize (stubs)
    mov rsi, msg_scanning_pcie
    call print_string
    call print_ok

    mov rsi, msg_loading_os
    call print_string
    call print_ok

    ; Clear mailbox
    xor rax, rax
    mov qword [MAILBOX_ADDR], rax

    ; Enter dispatch loop
    jmp gpu_dispatch_loop

gpu_dispatch_loop:
    call simulate_gpu
    call check_mailbox
    test rax, rax
    jz .no_request

    call handle_request

.no_request:
    pause
    jmp gpu_dispatch_loop

simulate_gpu:
    cmp byte [hello_pos], 19
    jge .done

    ; Get current character from hello string
    movzx eax, byte [hello_pos]
    mov al, [hello_msg + rax]

    ; Build mailbox request: OP_MMIO_WRITE_UART | char
    shl eax, 0              ; char in low byte
    or eax, (OP_MMIO_WRITE_UART << 24)
    mov [MAILBOX_ADDR], eax

    inc byte [hello_pos]
    ret

.done:
    mov eax, (OP_CPU_HALT << 24)
    mov [MAILBOX_ADDR], eax
    ret

check_mailbox:
    mov eax, [MAILBOX_ADDR]
    test eax, eax
    jz .no_request
    mov rax, 1
    ret
.no_request:
    xor rax, rax
    ret

handle_request:
    push rbx
    push rdx

    mov ebx, [MAILBOX_ADDR]
    mov eax, ebx
    shr eax, 24           ; Extract opcode

    cmp al, OP_MMIO_WRITE_UART
    je .uart_write

    cmp al, OP_CPU_HALT
    je .halt

    jmp .clear

.uart_write:
    and ebx, 0xFFFF
    mov al, bl
    mov dx, UART_PORT
    out dx, al
    jmp .clear

.halt:
    mov rsi, msg_gpu_halted
    call print_string
    cli
    hlt

.clear:
    xor eax, eax
    mov [MAILBOX_ADDR], eax
    pop rdx
    pop rbx
    ret

print_string:
    push rax
.loop:
    lodsb
    test al, al
    jz .done

    ; Write to VGA
    mov rdi, [vga_ptr]
    mov [rdi], al
    inc rdi
    mov byte [rdi], 0x0F
    inc rdi
    mov [vga_ptr], rdi

    ; Write to UART
    push rdx
    mov dx, UART_PORT
    out dx, al
    pop rdx

    jmp .loop

.done:
    pop rax
    ret

print_ok:
    push rsi
    mov rsi, msg_ok
    call print_string
    pop rsi
    ret

; ---------------------------------------------------------------------------
; Data Section
; ---------------------------------------------------------------------------
align 8
msg_microkernel:   db 'pxOS CPU Microkernel v0.3', 13, 10, 0
msg_scanning_pcie: db 'Scanning PCIe bus 0... ', 0
msg_loading_os:    db 'OK', 13, 10, 'Executing GPU program... ', 0
msg_gpu_halted:    db 13, 10, 'System halted.', 13, 10, 0
msg_ok:            db 'OK', 13, 10, 0

hello_msg:         db 'Hello from GPU OS!', 10
hello_pos:         db 0
vga_ptr:           dq 0xB8014

; 64-bit GDT
align 16
gdt64_start:
    dq 0                        ; Null descriptor
    dq 0x00AF9A000000FFFF       ; 64-bit code segment (0x08)
    dq 0x00AF92000000FFFF       ; 64-bit data segment (0x10)
gdt64_end:

gdt64_descriptor:
    dw gdt64_end - gdt64_start - 1
    dq gdt64_start

; Page tables (aligned)
align 4096
pml4_table: times 4096 db 0
pdpt_table: times 4096 db 0
pd_table:   times 4096 db 0
