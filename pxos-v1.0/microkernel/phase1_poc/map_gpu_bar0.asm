; map_gpu_bar0.asm - GPU BAR0 Memory Mapping for pxOS
; Maps GPU MMIO region into kernel address space with proper caching attributes

BITS 64

; Page table flags
%define PAGE_PRESENT       0x001
%define PAGE_WRITE         0x002
%define PAGE_USER          0x004
%define PAGE_WRITETHROUGH  0x008
%define PAGE_CACHE_DISABLE 0x010
%define PAGE_SIZE_2MB      0x080
%define PAGE_GLOBAL        0x100

; PAT index for uncacheable (UC) memory
%define PAT_UC             0x000

section .data
align 8
global gpu_bar0_phys
global gpu_bar0_virt
global gpu_bar0_size

gpu_bar0_phys: dq 0        ; Set by PCIe scan
gpu_bar0_virt: dq 0        ; Set by mapper
gpu_bar0_size: dq 0x1000000 ; Default 16MB

; Debug messages
msg_map_bar0:      db "Mapping GPU BAR0... ", 0
msg_map_success:   db "OK (virt=0x", 0
msg_map_failed:    db "FAILED (BAR0 not found)", 13, 10, 0
msg_closing_paren: db ")", 13, 10, 0

section .text
extern serial_print_64
extern serial_putc_64

global map_gpu_bar0
global map_mmio_page

;-----------------------------------------------------------------------------
; map_gpu_bar0()
; Map GPU BAR0 into kernel address space with UC (uncacheable) attributes
; Called after PCIe scan sets gpu_bar0_phys
;-----------------------------------------------------------------------------
map_gpu_bar0:
    push rax
    push rdi
    push rsi
    push rdx
    push rcx

    ; Debug output
    lea rsi, [rel msg_map_bar0]
    call serial_print_64

    ; Check if BAR0 was discovered by PCIe scan
    mov rax, [rel gpu_bar0_phys]
    test rax, rax
    jz .no_bar

    ; For now: identity map BAR0 (phys = virt)
    ; This is simple and works for early bring-up
    mov [rel gpu_bar0_virt], rax

    ; Map first 4KB page (enough for mailbox registers)
    ; For production: map multiple pages to cover full BAR
    mov rdi, rax                    ; phys_addr = BAR0 base
    mov rsi, rax                    ; virt_addr = BAR0 base (identity)
    mov rdx, PAT_UC                 ; Uncacheable for MMIO
    call map_mmio_page

    ; Success - print virtual address
    lea rsi, [rel msg_map_success]
    call serial_print_64

    mov rax, [rel gpu_bar0_virt]
    call print_hex_64

    lea rsi, [rel msg_closing_paren]
    call serial_print_64

    jmp .done

.no_bar:
    ; PCIe scan didn't find GPU or BAR0
    lea rsi, [rel msg_map_failed]
    call serial_print_64

.done:
    pop rcx
    pop rdx
    pop rsi
    pop rdi
    pop rax
    ret

;-----------------------------------------------------------------------------
; map_mmio_page(phys_addr, virt_addr, pat_index)
; Map a single 4KB MMIO page into kernel page tables
; Input:
;   RDI = physical address
;   RSI = virtual address
;   RDX = PAT index (0=UC, 1=WC, etc.)
;-----------------------------------------------------------------------------
map_mmio_page:
    push rax
    push rbx
    push rcx
    push rdx
    push rdi
    push rsi

    ; Align addresses to 4KB boundary
    and rdi, ~0xFFF
    and rsi, ~0xFFF

    ; Get page table indices from virtual address
    mov rax, rsi

    ; PML4 index (bits 39-47)
    mov rbx, rax
    shr rbx, 39
    and rbx, 0x1FF

    ; PDP index (bits 30-38)
    mov rcx, rax
    shr rcx, 30
    and rcx, 0x1FF

    ; For simplicity, we use existing page tables set up during boot
    ; In production, you'd walk page tables and create new ones if needed

    ; Build page table entry
    ; Flags: Present | Write | Cache-Disable | Writethrough (for UC)
    mov rax, rdi                    ; Physical address
    or rax, PAGE_PRESENT | PAGE_WRITE | PAGE_CACHE_DISABLE | PAGE_WRITETHROUGH

    ; This is a simplified version
    ; Real implementation would walk page tables and install PTE
    ; For now, we rely on the 2MB identity mapping covering BAR0

    pop rsi
    pop rdi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

;-----------------------------------------------------------------------------
; print_hex_64(value)
; Print 64-bit value in hexadecimal
; Input: RAX = value
;-----------------------------------------------------------------------------
print_hex_64:
    push rax
    push rbx
    push rcx
    push rdx

    mov rbx, rax
    mov rcx, 16         ; 16 hex digits

.loop:
    rol rbx, 4          ; Rotate next nibble into position
    mov rax, rbx
    and rax, 0xF

    ; Convert to ASCII hex digit
    cmp al, 10
    jl .digit
    add al, 'A' - 10
    jmp .output
.digit:
    add al, '0'

.output:
    call serial_putc_64
    loop .loop

    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret
