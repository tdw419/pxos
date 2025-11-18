; pxOS CPU Microkernel v0.1 - Multiboot Version
; Compatible with GRUB bootloader

BITS 32                         ; GRUB loads us in 32-bit mode

; Multiboot header
MULTIBOOT_MAGIC        equ 0x1BADB002
MULTIBOOT_FLAGS        equ 0x00000003
MULTIBOOT_CHECKSUM     equ -(MULTIBOOT_MAGIC + MULTIBOOT_FLAGS)

section .multiboot
align 4
    dd MULTIBOOT_MAGIC
    dd MULTIBOOT_FLAGS
    dd MULTIBOOT_CHECKSUM

section .text
global _start

_start:
    ; We're in 32-bit mode, need to enter 64-bit
    ; Setup stack
    mov esp, stack_top

    ; Check if CPU supports long mode
    call check_long_mode
    test eax, eax
    jz .no_long_mode

    ; Setup page tables
    call setup_paging

    ; Enable PAE
    mov eax, cr4
    or eax, 1 << 5
    mov cr4, eax

    ; Load PML4
    mov eax, pml4_table
    mov cr3, eax

    ; Enable long mode
    mov ecx, 0xC0000080
    rdmsr
    or eax, 1 << 8
    wrmsr

    ; Enable paging
    mov eax, cr0
    or eax, 1 << 31
    mov cr0, eax

    ; Load 64-bit GDT
    lgdt [gdt64.pointer]

    ; Jump to 64-bit code
    jmp gdt64.code:long_mode_start

.no_long_mode:
    mov dword [0xB8000], 0x4F214F45  ; "E!"
    hlt

check_long_mode:
    mov eax, 0x80000000
    cpuid
    cmp eax, 0x80000001
    jb .no_long_mode

    mov eax, 0x80000001
    cpuid
    test edx, 1 << 29
    jz .no_long_mode

    mov eax, 1
    ret

.no_long_mode:
    xor eax, eax
    ret

setup_paging:
    ; Clear page tables
    mov edi, pml4_table
    mov ecx, 4096
    xor eax, eax
    rep stosd

    ; Setup PML4
    mov eax, pdpt_table
    or eax, 0x03
    mov [pml4_table], eax

    ; Setup PDPT
    mov eax, pd_table
    or eax, 0x03
    mov [pdpt_table], eax

    ; Setup PD with 2MB pages
    mov edi, pd_table
    mov eax, 0x83               ; Present, writable, 2MB page
    mov ecx, 512
.loop:
    mov [edi], eax
    add eax, 0x200000
    add edi, 8
    loop .loop

    ret

BITS 64
long_mode_start:
    ; Clear segment registers
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    ; Setup stack
    mov rsp, stack_top

    ; Call our microkernel
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
    cmp byte [hello_pos], 1
    jge .done

    mov eax, (OP_MMIO_WRITE_UART << 24) | 0x48
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
    mov al, bh

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

section .data
msg_microkernel:   db 'pxOS Microkernel v0.1', 0
msg_scanning_pcie: db ' Scanning PCIe... ', 0
msg_loading_os:    db ' Loading os.pxi... ', 0
msg_gpu_halted:    db ' GPU HALT received', 0
msg_ok:            db 'OK', 0

hello_pos:         db 0
vga_ptr:           dq 0xB8000

section .bss
align 4096
pml4_table:
    resb 4096
pdpt_table:
    resb 4096
pd_table:
    resb 4096

stack_bottom:
    resb 16384
stack_top:

section .rodata
gdt64:
    dq 0
.code: equ $ - gdt64
    dq 0x00AF9A000000FFFF
.data: equ $ - gdt64
    dq 0x00AF92000000FFFF
.pointer:
    dw $ - gdt64 - 1
    dq gdt64
