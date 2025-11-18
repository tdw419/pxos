; pxHV Stage 2 - Hypervisor Loader
; Transitions to long mode and enables VT-x
;
; Entry: Real mode, loaded at 0x10000
; Exit: Long mode, VT-x enabled, jumps to stage3
;
; Compile: nasm -f bin -o pxhv_stage2.bin pxhv_stage2.asm

BITS 16
ORG 0x10000

; Memory layout constants
PML4_ADDR       equ 0x70000     ; Page Map Level 4
PDPT_ADDR       equ 0x71000     ; Page Directory Pointer Table
PD_ADDR         equ 0x72000     ; Page Directory
PT_ADDR         equ 0x73000     ; Page Table

VMXON_REGION    equ 0x15000     ; VMXON region (4KB aligned)
VMCS_REGION     equ 0x16000     ; VMCS region (4KB aligned)
STACK_TOP       equ 0x9F000     ; Stack at ~640KB

stage2_entry:
    ; Setup segments
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x8000

    ; Print stage 2 message
    mov si, msg_stage2
    call print_string

    ; Check for long mode support
    call check_long_mode
    jc .no_long_mode

    ; Check for VT-x support
    call check_vmx
    jc .no_vmx

    ; Setup page tables for long mode
    call setup_page_tables

    ; Enable paging and long mode
    call enter_long_mode

    ; We should never get here
    cli
    hlt

.no_long_mode:
    mov si, msg_no_long_mode
    call print_string
    cli
    hlt

.no_vmx:
    mov si, msg_no_vmx
    call print_string
    cli
    hlt

;-----------------------------------------------------------------------------
; check_long_mode: Check if CPU supports 64-bit long mode
;-----------------------------------------------------------------------------
check_long_mode:
    pusha

    ; Check if CPUID is supported
    pushfd
    pop eax
    mov ecx, eax
    xor eax, 0x200000           ; Flip ID bit
    push eax
    popfd
    pushfd
    pop eax
    xor eax, ecx
    jz .no_cpuid

    ; Check for extended CPUID
    mov eax, 0x80000000
    cpuid
    cmp eax, 0x80000001
    jb .no_long_mode

    ; Check for long mode support
    mov eax, 0x80000001
    cpuid
    test edx, (1 << 29)         ; LM bit
    jz .no_long_mode

    popa
    clc
    ret

.no_cpuid:
.no_long_mode:
    popa
    stc
    ret

;-----------------------------------------------------------------------------
; check_vmx: Check if CPU supports Intel VT-x
;-----------------------------------------------------------------------------
check_vmx:
    pusha

    ; Check CPUID.1:ECX.VMX[bit 5]
    mov eax, 1
    cpuid
    test ecx, (1 << 5)
    jz .no_vmx

    ; Print VMX supported message
    mov si, msg_vmx_ok
    call print_string

    popa
    clc
    ret

.no_vmx:
    popa
    stc
    ret

;-----------------------------------------------------------------------------
; setup_page_tables: Create identity-mapped page tables for first 2MB
;-----------------------------------------------------------------------------
setup_page_tables:
    pusha

    ; Clear page table memory
    mov edi, PML4_ADDR
    mov ecx, 0x4000             ; 16KB (4 pages)
    xor eax, eax
    rep stosd

    ; Setup PML4 (maps to PDPT)
    mov edi, PML4_ADDR
    mov dword [edi], PDPT_ADDR | 3  ; Present, writable

    ; Setup PDPT (maps to PD)
    mov edi, PDPT_ADDR
    mov dword [edi], PD_ADDR | 3    ; Present, writable

    ; Setup PD with 2MB pages (identity mapped)
    ; This maps first 1GB with 2MB pages
    mov edi, PD_ADDR
    mov eax, 0x83                   ; Present, writable, 2MB page
    mov ecx, 512                    ; 512 entries = 1GB
.pd_loop:
    mov [edi], eax
    add eax, 0x200000               ; Next 2MB
    add edi, 8
    loop .pd_loop

    popa
    ret

;-----------------------------------------------------------------------------
; enter_long_mode: Enable PAE, long mode, and paging
;-----------------------------------------------------------------------------
enter_long_mode:
    ; Disable interrupts
    cli

    ; Load GDT
    lgdt [gdt_descriptor]

    ; Enable PAE (Physical Address Extension)
    mov eax, cr4
    or eax, (1 << 5)                ; Set PAE bit
    mov cr4, eax

    ; Load PML4 address
    mov eax, PML4_ADDR
    mov cr3, eax

    ; Enable long mode (set EFER.LME)
    mov ecx, 0xC0000080             ; EFER MSR
    rdmsr
    or eax, (1 << 8)                ; Set LME bit
    wrmsr

    ; Enable paging and protected mode
    mov eax, cr0
    or eax, (1 << 31) | (1 << 0)   ; Set PG and PE
    mov cr0, eax

    ; Jump to 64-bit code segment
    jmp 0x08:long_mode_entry

;-----------------------------------------------------------------------------
; print_string: Print null-terminated string (16-bit real mode)
;-----------------------------------------------------------------------------
print_string:
    pusha
    mov ah, 0x0E
.loop:
    lodsb
    test al, al
    jz .done
    int 0x10
    jmp .loop
.done:
    popa
    ret

;-----------------------------------------------------------------------------
; GDT for long mode
;-----------------------------------------------------------------------------
align 8
gdt_start:
    ; Null descriptor
    dq 0x0000000000000000

    ; 64-bit code segment
    dq 0x00AF9A000000FFFF          ; Base=0, Limit=0xFFFFF, Present, 64-bit, Code

    ; 64-bit data segment
    dq 0x00CF92000000FFFF          ; Base=0, Limit=0xFFFFF, Present, 64-bit, Data

gdt_end:

gdt_descriptor:
    dw gdt_end - gdt_start - 1     ; GDT size
    dd gdt_start                    ; GDT address

;-----------------------------------------------------------------------------
; 64-bit code (long mode entry point)
;-----------------------------------------------------------------------------
BITS 64
long_mode_entry:
    ; Setup segment registers
    mov ax, 0x10                    ; Data segment selector
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    ; Setup stack
    mov rsp, STACK_TOP

    ; Print long mode message
    mov rsi, msg_long_mode
    call print_string_64

    ; Enable VT-x
    call enable_vmx

    ; Initialize VMXON region
    call vmxon_init

    ; Execute VMXON
    call vmxon_exec

    ; If we get here, VMXON succeeded!
    mov rsi, msg_vmxon_ok
    call print_string_64

    ; Jump to stage 3 (VMCS setup and guest launch)
    ; For now, just halt
    cli
    hlt

;-----------------------------------------------------------------------------
; enable_vmx: Enable VT-x by setting CR4.VMXE
;-----------------------------------------------------------------------------
enable_vmx:
    ; Set CR4.VMXE (bit 13)
    mov rax, cr4
    or rax, (1 << 13)
    mov cr4, rax

    ; Configure IA32_FEATURE_CONTROL MSR
    mov ecx, 0x3A
    rdmsr
    test eax, 1                     ; Check if locked
    jnz .already_configured

    ; Lock bit + enable VMX outside SMX
    or eax, 5
    wrmsr

.already_configured:
    ret

;-----------------------------------------------------------------------------
; vmxon_init: Initialize VMXON region
;-----------------------------------------------------------------------------
vmxon_init:
    ; Clear VMXON region
    mov rdi, VMXON_REGION
    mov rcx, 1024                   ; 4KB = 1024 qwords
    xor rax, rax
    rep stosq

    ; Read VMCS revision ID from IA32_VMX_BASIC MSR
    mov ecx, 0x480
    rdmsr

    ; Write revision ID to first 4 bytes of VMXON region
    mov [VMXON_REGION], eax

    ret

;-----------------------------------------------------------------------------
; vmxon_exec: Execute VMXON instruction
;-----------------------------------------------------------------------------
vmxon_exec:
    ; VMXON [VMXON_REGION]
    mov rax, VMXON_REGION
    vmxon [rax]

    jc .vmxon_failed                ; CF=1: VMfailInvalid
    jz .vmxon_failed                ; ZF=1: VMfailValid

    ret

.vmxon_failed:
    mov rsi, msg_vmxon_fail
    call print_string_64
    cli
    hlt

;-----------------------------------------------------------------------------
; print_string_64: Print null-terminated string (64-bit mode)
; Input: RSI = string address
; Uses direct VGA text mode writes (0xB8000)
;-----------------------------------------------------------------------------
print_string_64:
    push rax
    push rbx
    push rcx

    mov rbx, 0xB8000                ; VGA text buffer
    mov ah, 0x0F                    ; White on black

.loop:
    lodsb
    test al, al
    jz .done
    mov [rbx], ax
    add rbx, 2
    jmp .loop

.done:
    pop rcx
    pop rbx
    pop rax
    ret

;-----------------------------------------------------------------------------
; Strings
;-----------------------------------------------------------------------------
msg_stage2:         db 'Stage 2: Hypervisor loader', 13, 10, 0
msg_no_long_mode:   db 'ERROR: Long mode not supported!', 13, 10, 0
msg_no_vmx:         db 'ERROR: VT-x not supported!', 13, 10, 0
msg_vmx_ok:         db 'VT-x supported', 13, 10, 0
msg_long_mode:      db 'Long mode enabled', 0
msg_vmxon_ok:       db 'VMXON executed successfully!', 0
msg_vmxon_fail:     db 'VMXON failed!', 0

;-----------------------------------------------------------------------------
; Pad to sector boundary (optional)
;-----------------------------------------------------------------------------
times 20480-($-$$) db 0             ; Pad to 20KB
