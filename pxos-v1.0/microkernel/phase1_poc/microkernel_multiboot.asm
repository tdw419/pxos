; pxOS Phase 2 - GRUB Multiboot Kernel
; GPU-Centric Operating System Architecture
; Version 0.4 - PCIe Enumeration + Long Mode

BITS 32

; ==============================================================================
; Multiboot2 Header
; ==============================================================================
section .multiboot
align 8
multiboot_header_start:
    dd 0xE85250D6                           ; Magic number (multiboot2)
    dd 0                                    ; Architecture: i386
    dd multiboot_header_end - multiboot_header_start ; Header length
    ; Checksum
    dd 0x100000000 - (0xE85250D6 + 0 + (multiboot_header_end - multiboot_header_start))

    ; End tag
    align 8
    dw 0    ; type
    dw 0    ; flags
    dd 8    ; size
multiboot_header_end:

; ==============================================================================
; Data Section
; ==============================================================================
section .data
align 8

; VGA text buffer
VGA_BUFFER equ 0xB8000
VGA_COLOR  equ 0x0F

; Messages
msg_boot:       db "pxOS CPU Microkernel v0.4", 0
msg_longmode:   db "Entering Long Mode...", 0
msg_pcie_scan:  db "Scanning PCIe bus 0...", 0
msg_gpu_found:  db "GPU found at: ", 0
msg_bar0:       db "  BAR0: 0x", 0
msg_hello:      db "Hello from GPU OS!", 0
msg_done:       db "System halted.", 0

; Serial port (COM1)
SERIAL_PORT equ 0x3F8

; PCIe Configuration
pcie_config_addr equ 0xCF8
pcie_config_data equ 0xCFC

; GPU state
gpu_bus     db 0
gpu_dev     db 0
gpu_func    db 0
gpu_bar0    dd 0

; External symbols from map_gpu_bar0.asm
extern gpu_bar0_phys
extern gpu_bar0_virt
extern map_gpu_bar0

; External symbols from mailbox_protocol.asm
extern mailbox_init
extern mailbox_test

; Page table pointers
pml4_table  equ 0x1000
pdp_table   equ 0x2000
pd_table    equ 0x3000

; ==============================================================================
; BSS Section
; ==============================================================================
section .bss
align 4096
stack_bottom:
    resb 16384                              ; 16KB stack
stack_top:

; ==============================================================================
; Code Section
; ==============================================================================
section .text
global _start
global serial_print_64
global serial_putc_64

_start:
    ; Clear interrupts
    cli

    ; VERY EARLY DEBUG: Write '0' to VGA to prove we got here
    mov byte [VGA_BUFFER], '0'
    mov byte [VGA_BUFFER + 1], 0x0F

    ; Set up stack
    mov esp, stack_top

    ; DEBUG: Write '1' after stack setup
    mov byte [VGA_BUFFER + 2], '1'
    mov byte [VGA_BUFFER + 3], 0x0F

    ; Initialize serial port
    call init_serial

    ; DEBUG: Write '2' after serial init
    mov byte [VGA_BUFFER + 4], '2'
    mov byte [VGA_BUFFER + 5], 0x0F

    ; Clear screen and show boot message
    call clear_screen
    mov esi, msg_boot
    call print_string_32
    call print_newline

    ; Also print to serial
    mov esi, msg_boot
    call serial_print
    call serial_newline

    ; DEBUG: Write '3' before page table setup
    mov byte [VGA_BUFFER + 6], '3'
    mov byte [VGA_BUFFER + 7], 0x0F

    ; Set up page tables for long mode
    call setup_page_tables

    ; DEBUG: Write '4' after page table setup
    mov byte [VGA_BUFFER + 8], '4'
    mov byte [VGA_BUFFER + 9], 0x0F

    ; Enter long mode
    mov esi, msg_longmode
    call print_string_32
    call print_newline

    mov esi, msg_longmode
    call serial_print
    call serial_newline

    ; DEBUG: Write '5' before entering long mode
    mov byte [VGA_BUFFER + 10], '5'
    mov byte [VGA_BUFFER + 11], 0x0F

    call enter_long_mode

    ; Should not reach here
    hlt

; ==============================================================================
; 32-bit Helper Functions
; ==============================================================================

; Clear VGA screen (32-bit)
clear_screen:
    push eax
    push ecx
    push edi

    mov edi, VGA_BUFFER
    mov ecx, 2000                           ; 80x25 characters
    mov ax, 0x0720                          ; Space with color
    rep stosw

    pop edi
    pop ecx
    pop eax
    ret

; Initialize serial port (COM1)
init_serial:
    push eax
    push edx

    mov dx, SERIAL_PORT + 1
    mov al, 0x00
    out dx, al              ; Disable interrupts

    mov dx, SERIAL_PORT + 3
    mov al, 0x80
    out dx, al              ; Enable DLAB

    mov dx, SERIAL_PORT + 0
    mov al, 0x03
    out dx, al              ; Set divisor low byte (38400 baud)

    mov dx, SERIAL_PORT + 1
    mov al, 0x00
    out dx, al              ; Set divisor high byte

    mov dx, SERIAL_PORT + 3
    mov al, 0x03
    out dx, al              ; 8 bits, no parity, one stop bit

    mov dx, SERIAL_PORT + 2
    mov al, 0xC7
    out dx, al              ; Enable FIFO, clear, 14-byte threshold

    mov dx, SERIAL_PORT + 4
    mov al, 0x0B
    out dx, al              ; IRQs enabled, RTS/DSR set

    pop edx
    pop eax
    ret

; Write character to serial port - AL = character
serial_putc:
    push eax
    push edx

    ; Wait for transmit buffer to be empty
    mov dx, SERIAL_PORT + 5
.wait:
    in al, dx
    test al, 0x20          ; Test transmit empty bit
    jz .wait

    ; Restore character and send it
    pop edx
    pop eax
    push eax
    push edx

    mov dx, SERIAL_PORT
    out dx, al

    pop edx
    pop eax
    ret

; Print string to serial - ESI = string pointer
serial_print:
    push eax
    push esi

.loop:
    lodsb
    test al, al
    jz .done
    call serial_putc
    jmp .loop

.done:
    pop esi
    pop eax
    ret

; Print newline to serial
serial_newline:
    push eax
    mov al, 13
    call serial_putc
    mov al, 10
    call serial_putc
    pop eax
    ret

; Print string (32-bit) - ESI = string pointer
print_string_32:
    push eax
    push ebx
    push edi

    mov edi, VGA_BUFFER
    xor ebx, ebx
.loop:
    lodsb
    test al, al
    jz .done
    mov ah, VGA_COLOR
    mov [edi + ebx * 2], ax
    inc ebx
    cmp ebx, 2000
    jl .loop
.done:
    pop edi
    pop ebx
    pop eax
    ret

; Print newline (32-bit)
print_newline:
    ; For simplicity, just add 160 bytes (80 chars * 2)
    ret

; Setup page tables for long mode
setup_page_tables:
    push eax
    push edi
    push ecx

    ; Clear page tables
    mov edi, pml4_table
    mov ecx, 0x3000
    xor eax, eax
    rep stosb

    ; Set up PML4
    mov edi, pml4_table
    mov dword [edi], pdp_table | 3          ; Present + RW

    ; Set up PDP
    mov edi, pdp_table
    mov dword [edi], pd_table | 3           ; Present + RW

    ; Set up PD - identity map first 2MB
    mov edi, pd_table
    mov eax, 0x83                           ; Present + RW + PS (2MB pages)
    mov ecx, 512                            ; 512 entries
.map_loop:
    mov [edi], eax
    add eax, 0x200000                       ; Next 2MB
    add edi, 8
    loop .map_loop

    pop ecx
    pop edi
    pop eax
    ret

; Enter long mode
enter_long_mode:
    ; Load CR3 with PML4
    mov eax, pml4_table
    mov cr3, eax

    ; Enable PAE in CR4
    mov eax, cr4
    or eax, 1 << 5
    mov cr4, eax

    ; Set long mode bit in EFER MSR
    mov ecx, 0xC0000080
    rdmsr
    or eax, 1 << 8
    wrmsr

    ; Enable paging in CR0
    mov eax, cr0
    or eax, 1 << 31
    mov cr0, eax

    ; Load GDT and jump to 64-bit code
    lgdt [gdt_descriptor]
    jmp 0x08:long_mode_start

; ==============================================================================
; GDT for Long Mode
; ==============================================================================
section .data
align 16
gdt_start:
    dq 0                                    ; Null descriptor
    dq 0x00AF9A000000FFFF                   ; 64-bit code segment
    dq 0x00AF92000000FFFF                   ; 64-bit data segment
gdt_end:

gdt_descriptor:
    dw gdt_end - gdt_start - 1
    dd gdt_start

; ==============================================================================
; 64-bit Code
; ==============================================================================
BITS 64
section .text64
long_mode_start:
    ; Set up segment registers
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    ; Update stack pointer to 64-bit
    mov rsp, stack_top

    ; Mark that we're in long mode
    mov byte [VGA_BUFFER + 4], 'L'
    mov byte [VGA_BUFFER + 5], 0x0A

    ; Print to serial in 64-bit mode
    lea rsi, [rel msg_pcie_scan]
    call serial_print_64

    ; Scan PCIe bus
    call pcie_scan_64

    ; Map GPU BAR0 into kernel address space
    call map_gpu_bar0

    ; Initialize mailbox protocol
    call mailbox_init

    ; Test mailbox with UART write
    call mailbox_test

    ; Print hello message
    call print_hello_64

    ; Print to serial
    lea rsi, [rel msg_hello]
    call serial_print_64

    ; Halt
.halt:
    hlt
    jmp .halt

; ==============================================================================
; 64-bit PCIe Enumeration
; ==============================================================================
pcie_scan_64:
    push rax
    push rbx
    push rcx
    push rdx

    ; Mark scanning
    mov byte [VGA_BUFFER + 6], 'P'
    mov byte [VGA_BUFFER + 7], 0x0F

    ; Scan bus 0, devices 0-31
    xor rbx, rbx                            ; Device counter
.scan_device:
    ; Build config address for device
    mov eax, 0x80000000                     ; Enable bit
    shl ebx, 11                             ; Device << 11
    or eax, ebx
    shr ebx, 11                             ; Restore device number

    ; Read vendor ID
    mov dx, pcie_config_addr
    out dx, eax

    mov dx, pcie_config_data
    in eax, dx

    ; Check if device exists (vendor ID != 0xFFFF)
    cmp ax, 0xFFFF
    je .next_device

    ; Device found - check if it's VGA (class code 0x0300)
    push rax

    mov eax, 0x80000000
    mov ecx, ebx
    shl ecx, 11
    or eax, ecx
    or eax, 0x08                            ; Offset 0x08 (class code)

    mov dx, pcie_config_addr
    out dx, eax

    mov dx, pcie_config_data
    in eax, dx

    shr eax, 16                             ; Get class code
    cmp ax, 0x0300                          ; VGA controller?
    pop rax
    jne .next_device

    ; VGA device found!
    mov [gpu_dev], bl

    ; Read BAR0
    mov eax, 0x80000000
    mov ecx, ebx
    shl ecx, 11
    or eax, ecx
    or eax, 0x10                            ; Offset 0x10 (BAR0)

    mov dx, pcie_config_addr
    out dx, eax

    mov dx, pcie_config_data
    in eax, dx

    and eax, 0xFFFFFFF0                     ; Mask off flags
    mov [gpu_bar0], eax

    ; Save to 64-bit variable for BAR mapping
    mov rdx, rax
    mov [rel gpu_bar0_phys], rdx

    ; Mark GPU found
    mov byte [VGA_BUFFER + 8], 'G'
    mov byte [VGA_BUFFER + 9], 0x0A

    jmp .done

.next_device:
    inc rbx
    cmp rbx, 32
    jl .scan_device

.done:
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

; ==============================================================================
; 64-bit Print Function
; ==============================================================================
print_hello_64:
    push rax
    push rbx
    push rcx

    ; Print "Hello from GPU OS!" at row 2
    mov rbx, VGA_BUFFER + 160               ; Row 2
    lea rcx, [rel msg_hello]

.loop:
    mov al, [rcx]
    test al, al
    jz .done

    mov ah, 0x0A                            ; Green color
    mov [rbx], ax

    inc rcx
    add rbx, 2
    jmp .loop

.done:
    pop rcx
    pop rbx
    pop rax
    ret

; ==============================================================================
; 64-bit Serial Functions
; ==============================================================================
; Write character to serial port (64-bit) - AL = character
serial_putc_64:
    push rax
    push rdx

    ; Wait for transmit buffer empty
    mov dx, SERIAL_PORT + 5
.wait:
    in al, dx
    test al, 0x20
    jz .wait

    ; Restore character and send
    pop rdx
    pop rax
    push rax
    push rdx

    mov dx, SERIAL_PORT
    out dx, al

    pop rdx
    pop rax
    ret

; Print string to serial (64-bit) - RSI = string pointer
serial_print_64:
    push rax
    push rsi

.loop:
    lodsb
    test al, al
    jz .done
    call serial_putc_64
    jmp .loop

.done:
    ; Print newline
    mov al, 13
    call serial_putc_64
    mov al, 10
    call serial_putc_64

    pop rsi
    pop rax
    ret
