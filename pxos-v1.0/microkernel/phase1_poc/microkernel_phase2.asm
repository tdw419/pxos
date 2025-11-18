; pxOS Microkernel v0.4 - Phase 2: GPU Hardware Integration
; 32-bit entry point with BAR memory mapping support

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
    ; Map all memory as WB by default
    ; BAR mapping will override specific regions later
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
; Microkernel Main (64-bit) - Phase 2 Version
; ---------------------------------------------------------------------------
MAILBOX_ADDR equ 0x20000
UART_PORT    equ 0x3F8

OP_MMIO_WRITE_UART  equ 0x80
OP_CPU_HALT         equ 0x8F

microkernel_main:
    ; Print banner
    mov rsi, msg_microkernel
    call print_string

    ; Phase 2: Initialize PAT for custom cache types
    mov rsi, msg_init_pat
    call print_string
    call init_pat_cpu  ; Initialize PAT MSR
    call print_ok

    ; Phase 2: Scan PCIe bus
    mov rsi, msg_scanning_pcie
    call print_string
    call pcie_scan
    call print_ok

    ; Phase 2: Map GPU BAR regions
    mov rsi, msg_mapping_bars
    call print_string
    call init_bar_mapping
    call print_ok

    ; Phase 2: Initialize hardware mailbox
    mov rsi, msg_init_mailbox
    call print_string
    call init_hw_mailbox
    call print_ok

    ; Legacy: Load simulated GPU program
    mov rsi, msg_loading_os
    call print_string
    call print_ok

    ; Clear mailbox (simulated for now)
    xor rax, rax
    mov qword [MAILBOX_ADDR], rax

    ; Enter dispatch loop (will use hardware mailbox in future)
    jmp gpu_dispatch_loop

; ---------------------------------------------------------------------------
; init_pat_cpu - Initialize Page Attribute Table
; ---------------------------------------------------------------------------
init_pat_cpu:
    push rax
    push rcx
    push rdx

    ; Read current PAT MSR (0x277)
    mov ecx, 0x277
    rdmsr

    ; Set our custom PAT layout:
    ; PAT0=WB, PAT1=WT, PAT2=UC, PAT3=UC
    ; PAT4=WB, PAT5=WT, PAT6=UC, PAT7=WC
    mov eax, 0x00000406  ; PAT3-0
    mov edx, 0x01000406  ; PAT7-4 (PAT7=WC)

    wrmsr

    pop rdx
    pop rcx
    pop rax
    ret

; ---------------------------------------------------------------------------
; pcie_scan - Scan PCIe bus for GPU device
; ---------------------------------------------------------------------------
global pcie_scan
pcie_scan:
    push rax
    push rbx
    push rcx
    push rdx

    ; Scan bus 0, devices 0-31
    xor rbx, rbx  ; Device number

.scan_loop:
    cmp rbx, 32
    jge .scan_done

    ; Read vendor ID
    mov rax, rbx
    shl rax, 15  ; Device << 15
    or rax, 0x80000000  ; Enable bit
    mov dx, 0xCF8
    out dx, eax

    ; Read data
    mov dx, 0xCFC
    in eax, dx

    ; Check if device exists (vendor ID != 0xFFFF)
    cmp ax, 0xFFFF
    je .next_device

    ; Check if it's a VGA controller (class 03h, subclass 00h)
    mov rax, rbx
    shl rax, 15
    or rax, 0x80000008  ; Offset 0x08 (class code)
    or rax, 0x80000000
    mov dx, 0xCF8
    out dx, eax

    mov dx, 0xCFC
    in eax, dx
    shr eax, 16  ; Get class/subclass

    cmp ax, 0x0300  ; VGA controller
    je .found_gpu

.next_device:
    inc rbx
    jmp .scan_loop

.found_gpu:
    ; Save GPU device number
    mov [gpu_device], rbx

    ; Read BAR0
    mov rax, rbx
    shl rax, 15
    or rax, 0x80000010  ; Offset 0x10 (BAR0)
    or rax, 0x80000000
    mov dx, 0xCF8
    out dx, eax

    mov dx, 0xCFC
    in eax, dx
    and eax, 0xFFFFFFF0  ; Clear flag bits
    mov [gpu_bar0], rax

    ; For simplicity, assume 16MB BAR
    mov qword [gpu_bar0_size], 0x1000000

.scan_done:
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

; ---------------------------------------------------------------------------
; init_bar_mapping - Map GPU BARs with correct cache attributes
; ---------------------------------------------------------------------------
init_bar_mapping:
    push rax
    push rbx
    push rcx
    push rdx

    ; Check if we found a GPU
    mov rax, [gpu_bar0]
    test rax, rax
    jz .no_gpu

    ; Map mailbox region as UC (Uncacheable)
    ; Find PD entry for BAR0
    mov rbx, rax
    shr rbx, 21
    and rbx, 0x1FF

    mov rcx, pd_table
    shl rbx, 3
    add rcx, rbx

    ; Build PTE with UC attribute (PAT2)
    mov rdx, rax
    and rdx, 0xFFFFFFFFFFE00000
    or rdx, 0x83      ; P=1, RW=1, PS=1
    or rdx, (1 << 12) ; PAT bit for UC

    mov [rcx], rdx
    invlpg [rax]

    ; Map buffer regions as WC (Write-Combining)
    ; Map next 8x 2MB pages as WC (16MB total)
    add rax, 0x200000
    mov r8, 7  ; 7 more pages

.map_wc_loop:
    mov rbx, rax
    shr rbx, 21
    and rbx, 0x1FF

    mov rcx, pd_table
    shl rbx, 3
    add rcx, rbx

    ; Build PTE with WC attribute (PAT7)
    mov rdx, rax
    and rdx, 0xFFFFFFFFFFE00000
    or rdx, 0x83       ; P=1, RW=1, PS=1
    or rdx, (1 << 7)   ; PAT bit 0
    or rdx, (1 << 12)  ; PAT bit 1 (PAT7=WC)

    mov [rcx], rdx
    invlpg [rax]

    add rax, 0x200000
    dec r8
    jnz .map_wc_loop

.no_gpu:
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

; ---------------------------------------------------------------------------
; init_hw_mailbox - Initialize hardware mailbox
; ---------------------------------------------------------------------------
init_hw_mailbox:
    push rax

    ; Clear hardware mailbox
    mov rax, [gpu_bar0]
    test rax, rax
    jz .no_mailbox

    ; Zero out mailbox region
    mov dword [rax], 0

.no_mailbox:
    pop rax
    ret

; ---------------------------------------------------------------------------
; GPU Dispatch Loop (Legacy simulation for now)
; ---------------------------------------------------------------------------
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

    ; Build mailbox request
    shl eax, 0
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
    shr eax, 24

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
msg_microkernel:   db 'pxOS CPU Microkernel v0.4 (Phase 2)', 13, 10, 0
msg_init_pat:      db 'Initializing PAT (cache types)... ', 0
msg_scanning_pcie: db 'Scanning PCIe bus 0... ', 0
msg_mapping_bars:  db 'Mapping GPU BARs (UC/WC)... ', 0
msg_init_mailbox:  db 'Initializing hardware mailbox... ', 0
msg_loading_os:    db 'Loading GPU program... ', 0
msg_gpu_halted:    db 13, 10, 'System halted.', 13, 10, 0
msg_ok:            db 'OK', 13, 10, 0

hello_msg:         db 'Hello from GPU OS!', 10
hello_pos:         db 0
vga_ptr:           dq 0xB8014

; GPU PCIe information
gpu_device:        dq 0
gpu_bar0:          dq 0
gpu_bar0_size:     dq 0x1000000  ; 16MB default
gpu_bar2:          dq 0

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
global pml4_table
global pdpt_table
global pd_table
pml4_table: times 4096 db 0
pdpt_table: times 4096 db 0
pd_table:   times 4096 db 0
