; pxHV Stage 2+3 - Hypervisor with Real Mode Guest Support
; Stages 2+3: Long mode, VT-x, VMCS setup, and guest launch
;
; Entry: Real mode, loaded at 0x10000
; Exit: Guest launched in real mode
;
; Compile: nasm -f bin -o pxhv_stage2.bin pxhv_stage2.asm

BITS 16
ORG 0x10000

; Memory layout constants
PML4_ADDR       equ 0x70000     ; Page Map Level 4
PDPT_ADDR       equ 0x71000     ; Page Directory Pointer Table
PD_ADDR         equ 0x72000     ; Page Directory

VMXON_REGION    equ 0x15000     ; VMXON region (4KB aligned)
VMCS_REGION     equ 0x16000     ; VMCS region (4KB aligned)
EPT_PML4        equ 0x80000     ; EPT Page Map Level 4 (4KB aligned)
EPT_PDPT        equ 0x81000     ; EPT PDPT
EPT_PD          equ 0x82000     ; EPT PD
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
    xor eax, 0x200000
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
    test edx, (1 << 29)
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
; setup_page_tables: Create identity-mapped page tables
;-----------------------------------------------------------------------------
setup_page_tables:
    pusha

    ; Clear page table memory
    mov edi, PML4_ADDR
    mov ecx, 0x4000
    xor eax, eax
    rep stosd

    ; Setup PML4
    mov edi, PML4_ADDR
    mov dword [edi], PDPT_ADDR | 3

    ; Setup PDPT
    mov edi, PDPT_ADDR
    mov dword [edi], PD_ADDR | 3

    ; Setup PD with 2MB pages
    mov edi, PD_ADDR
    mov eax, 0x83
    mov ecx, 512
.pd_loop:
    mov [edi], eax
    add eax, 0x200000
    add edi, 8
    loop .pd_loop

    popa
    ret

;-----------------------------------------------------------------------------
; enter_long_mode: Enable PAE, long mode, and paging
;-----------------------------------------------------------------------------
enter_long_mode:
    cli

    ; Load GDT
    lgdt [gdt_descriptor]

    ; Enable PAE
    mov eax, cr4
    or eax, (1 << 5)
    mov cr4, eax

    ; Load PML4
    mov eax, PML4_ADDR
    mov cr3, eax

    ; Enable long mode
    mov ecx, 0xC0000080
    rdmsr
    or eax, (1 << 8)
    wrmsr

    ; Enable paging
    mov eax, cr0
    or eax, (1 << 31) | (1 << 0)
    mov cr0, eax

    ; Jump to 64-bit code
    jmp 0x08:long_mode_entry

;-----------------------------------------------------------------------------
; print_string: Print null-terminated string (16-bit)
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
    dq 0x0000000000000000       ; Null
    dq 0x00AF9A000000FFFF       ; 64-bit code
    dq 0x00CF92000000FFFF       ; 64-bit data
gdt_end:

gdt_descriptor:
    dw gdt_end - gdt_start - 1
    dd gdt_start

;-----------------------------------------------------------------------------
; 64-bit code - Stage 3 begins here
;-----------------------------------------------------------------------------
BITS 64
long_mode_entry:
    ; Setup segments
    mov ax, 0x10
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

    ; VMXON succeeded!
    mov rsi, msg_vmxon_ok
    call print_string_64

    ; ========== STAGE 3: Pixel Driver Test ==========

    ; Execute the pixel-encoded serial driver
    call execute_pixel_driver

    ; Halt the system after the driver has finished
    mov rsi, msg_driver_done
    call print_string_64
    cli
    hlt

;-----------------------------------------------------------------------------
; enable_vmx: Enable VT-x
;-----------------------------------------------------------------------------
enable_vmx:
    mov rax, cr4
    or rax, (1 << 13)
    mov cr4, rax

    mov ecx, 0x3A
    rdmsr
    test eax, 1
    jnz .configured

    or eax, 5
    wrmsr

.configured:
    ret

;-----------------------------------------------------------------------------
; vmxon_init: Initialize VMXON region
;-----------------------------------------------------------------------------
vmxon_init:
    ; Clear VMXON region
    mov rdi, VMXON_REGION
    mov rcx, 1024
    xor rax, rax
    rep stosq

    ; Write revision ID
    mov ecx, 0x480
    rdmsr
    mov [VMXON_REGION], eax

    ret

;-----------------------------------------------------------------------------
; vmxon_exec: Execute VMXON
;-----------------------------------------------------------------------------
vmxon_exec:
    mov rax, VMXON_REGION
    vmxon [rax]

    jc .failed
    jz .failed

    ret

.failed:
    mov rsi, msg_vmxon_fail
    call print_string_64
    cli
    hlt

;-----------------------------------------------------------------------------
; vmcs_init: Initialize VMCS region
;-----------------------------------------------------------------------------
vmcs_init:
    ; Clear VMCS region
    mov rdi, VMCS_REGION
    mov rcx, 1024
    xor rax, rax
    rep stosq

    ; Write VMCS revision ID
    mov ecx, 0x480
    rdmsr
    and eax, 0x7FFFFFFF         ; Clear bit 31 (shadow VMCS indicator)
    mov [VMCS_REGION], eax

    ; Load VMCS pointer
    mov rax, VMCS_REGION
    vmptrld [rax]
    jc .error
    jz .error

    ret

.error:
    mov rsi, msg_vmcs_fail
    call print_string_64
    cli
    hlt

;-----------------------------------------------------------------------------
; setup_ept_tables: Build Extended Page Tables for guest
;-----------------------------------------------------------------------------
setup_ept_tables:
    ; Clear EPT memory
    mov rdi, EPT_PML4
    mov rcx, 3072               ; 12KB (3 pages)
    xor rax, rax
    rep stosq

    ; EPT PML4[0] → EPT PDPT
    mov rax, EPT_PDPT
    or rax, 7                   ; Read/Write/Execute
    mov [EPT_PML4], rax

    ; EPT PDPT[0] → EPT PD
    mov rax, EPT_PD
    or rax, 7
    mov [EPT_PDPT], rax

    ; EPT PD: Identity map first 1GB with 2MB pages
    mov rdi, EPT_PD
    xor rax, rax
    or rax, 0x87                ; Read/Write/Execute + 2MB page
    mov rcx, 512
.ept_loop:
    mov [rdi], rax
    add rax, 0x200000
    add rdi, 8
    loop .ept_loop

    ret

;-----------------------------------------------------------------------------
; load_guest_code: Copy guest code to 0x7C00
;-----------------------------------------------------------------------------
load_guest_code:
    push rsi
    push rdi
    push rcx

    ; Source: embedded guest binary
    mov rsi, guest_code_data
    ; Destination: 0x7C00
    mov rdi, 0x7C00
    ; Length: 512 bytes
    mov rcx, 512
    rep movsb

    pop rcx
    pop rdi
    pop rsi
    ret

;-----------------------------------------------------------------------------
; vmcs_configure: Configure all VMCS fields
;-----------------------------------------------------------------------------
vmcs_configure:
    ; === Guest State ===

    ; Guest CR0 (Real mode: PE=0, PG=0, but with NW=0, CD=0)
    mov rax, 0x20               ; NE bit for FPU
    mov rdx, 0x6800
    vmwrite rdx, rax
    jc .error

    ; Guest CR3 (doesn't matter in real mode)
    xor rax, rax
    mov rdx, 0x6802
    vmwrite rdx, rax
    jc .error

    ; Guest CR4 (VMX requires certain bits)
    mov rax, 0x2000             ; VMXE bit
    mov rdx, 0x6804
    vmwrite rdx, rax
    jc .error

    ; Guest RIP (entry point at 0x7C00)
    mov rax, 0x7C00
    mov rdx, 0x681E
    vmwrite rdx, rax
    jc .error

    ; Guest RSP
    mov rax, 0x7C00
    mov rdx, 0x681C
    vmwrite rdx, rax
    jc .error

    ; Guest RFLAGS (bit 1 always 1, IF=1 for interrupts)
    mov rax, 0x202
    mov rdx, 0x6820
    vmwrite rdx, rax
    jc .error

    ; Guest CS (real mode segment)
    xor rax, rax
    mov rdx, 0x0802             ; CS selector
    vmwrite rdx, rax
    jc .error

    mov rax, 0x0                ; CS base
    mov rdx, 0x6808
    vmwrite rdx, rax
    jc .error

    mov rax, 0xFFFF             ; CS limit (64KB)
    mov rdx, 0x4800
    vmwrite rdx, rax
    jc .error

    mov rax, 0x9B               ; CS access rights (present, code, readable)
    mov rdx, 0x4816
    vmwrite rdx, rax
    jc .error

    ; Guest DS/ES/SS (all same for real mode)
    xor rax, rax
    mov rdx, 0x0804             ; DS selector
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x0806             ; ES selector
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x0808             ; SS selector
    vmwrite rdx, rax
    jc .error

    xor rax, rax
    mov rdx, 0x6804             ; DS base
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x6806             ; ES base
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x6808             ; SS base
    vmwrite rdx, rax
    jc .error

    mov rax, 0xFFFF
    mov rdx, 0x4802             ; DS limit
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x4804             ; ES limit
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x4806             ; SS limit
    vmwrite rdx, rax
    jc .error

    mov rax, 0x93               ; DS/ES/SS access (present, data, writable)
    mov rdx, 0x4818             ; DS access
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x481A             ; ES access
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x481C             ; SS access
    vmwrite rdx, rax
    jc .error

    ; Guest FS/GS
    xor rax, rax
    mov rdx, 0x080A             ; FS selector
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x080C             ; GS selector
    vmwrite rdx, rax
    jc .error

    xor rax, rax
    mov rdx, 0x680A             ; FS base
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x680C             ; GS base
    vmwrite rdx, rax
    jc .error

    mov rax, 0xFFFF
    mov rdx, 0x4808             ; FS limit
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x480A             ; GS limit
    vmwrite rdx, rax
    jc .error

    mov rax, 0x93
    mov rdx, 0x481E             ; FS access
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x4820             ; GS access
    vmwrite rdx, rax
    jc .error

    ; Guest GDTR
    xor rax, rax
    mov rdx, 0x6816             ; GDTR base
    vmwrite rdx, rax
    jc .error

    mov rax, 0xFFFF
    mov rdx, 0x4810             ; GDTR limit
    vmwrite rdx, rax
    jc .error

    ; Guest IDTR
    xor rax, rax
    mov rdx, 0x6818             ; IDTR base
    vmwrite rdx, rax
    jc .error

    mov rax, 0xFFFF
    mov rdx, 0x4812             ; IDTR limit
    vmwrite rdx, rax
    jc .error

    ; Guest LDTR
    xor rax, rax
    mov rdx, 0x080E             ; LDTR selector
    vmwrite rdx, rax
    jc .error

    xor rax, rax
    mov rdx, 0x680E             ; LDTR base
    vmwrite rdx, rax
    jc .error

    xor rax, rax
    mov rdx, 0x480C             ; LDTR limit
    vmwrite rdx, rax
    jc .error

    mov rax, 0x10000            ; Unusable
    mov rdx, 0x4822             ; LDTR access
    vmwrite rdx, rax
    jc .error

    ; Guest TR
    xor rax, rax
    mov rdx, 0x0810             ; TR selector
    vmwrite rdx, rax
    jc .error

    xor rax, rax
    mov rdx, 0x6810             ; TR base
    vmwrite rdx, rax
    jc .error

    mov rax, 0xFFFF
    mov rdx, 0x480E             ; TR limit
    vmwrite rdx, rax
    jc .error

    mov rax, 0x8B               ; TR access (present, TSS)
    mov rdx, 0x4824             ; TR access
    vmwrite rdx, rax
    jc .error

    ; === Host State ===

    ; Host CR0
    mov rax, cr0
    mov rdx, 0x6C00
    vmwrite rdx, rax
    jc .error

    ; Host CR3
    mov rax, cr3
    mov rdx, 0x6C02
    vmwrite rdx, rax
    jc .error

    ; Host CR4
    mov rax, cr4
    mov rdx, 0x6C04
    vmwrite rdx, rax
    jc .error

    ; Host RIP (VM exit handler)
    mov rax, vm_exit_handler
    mov rdx, 0x6C16
    vmwrite rdx, rax
    jc .error

    ; Host RSP
    mov rax, STACK_TOP
    mov rdx, 0x6C14
    vmwrite rdx, rax
    jc .error

    ; Host CS/SS/DS/ES/FS/GS
    mov rax, 0x08
    mov rdx, 0x0C00             ; Host CS
    vmwrite rdx, rax
    jc .error

    mov rax, 0x10
    mov rdx, 0x0C02             ; Host SS
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x0C04             ; Host DS
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x0C06             ; Host ES
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x0C08             ; Host FS
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x0C0A             ; Host GS
    vmwrite rdx, rax
    jc .error

    ; Host TR
    xor rax, rax
    mov rdx, 0x0C0C
    vmwrite rdx, rax
    jc .error

    ; Host FS/GS base
    xor rax, rax
    mov rdx, 0x6C08             ; Host FS base
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x6C0A             ; Host GS base
    vmwrite rdx, rax
    jc .error

    ; Host TR base
    xor rax, rax
    mov rdx, 0x6C0C
    vmwrite rdx, rax
    jc .error

    ; Host GDTR base
    mov rax, gdt_start
    mov rdx, 0x6C0E
    vmwrite rdx, rax
    jc .error

    ; Host IDTR base
    xor rax, rax
    mov rdx, 0x6C10
    vmwrite rdx, rax
    jc .error

    ; === Control Fields ===

    ; Pin-based controls
    mov ecx, 0x481
    rdmsr
    mov rdx, 0x4000
    vmwrite rdx, rax
    jc .error

    ; Primary processor-based controls
    mov ecx, 0x482
    rdmsr
    or eax, (1 << 31)           ; Activate secondary controls
    mov rdx, 0x4002
    vmwrite rdx, rax
    jc .error

    ; Secondary processor-based controls (EPT + unrestricted guest)
    mov ecx, 0x48B
    rdmsr
    or eax, (1 << 1)            ; Enable EPT
    or eax, (1 << 7)            ; Enable unrestricted guest (real mode)
    mov rdx, 0x401E
    vmwrite rdx, rax
    jc .error

    ; EPT pointer (bits 11:0 = 0x1E for WB memory type + 4-level paging)
    mov rax, EPT_PML4
    or rax, 0x1E
    mov rdx, 0x201A
    vmwrite rdx, rax
    jc .error

    ; VM-exit controls
    mov ecx, 0x483
    rdmsr
    or eax, (1 << 9)            ; Host address-space size (64-bit)
    mov rdx, 0x400C
    vmwrite rdx, rax
    jc .error

    ; VM-entry controls (no IA-32e mode for real mode guest)
    mov ecx, 0x484
    rdmsr
    and eax, ~(1 << 9)          ; Clear IA-32e mode guest
    mov rdx, 0x4012
    vmwrite rdx, rax
    jc .error

    ret

.error:
    mov rsi, msg_vmcs_config_fail
    call print_string_64
    cli
    hlt

;-----------------------------------------------------------------------------
; vm_exit_handler: Handle VM exits from guest
;-----------------------------------------------------------------------------
vm_exit_handler:
    ; Read exit reason
    mov rdx, 0x4402
    vmread rax, rdx

    ; Mask to get basic exit reason
    and eax, 0xFFFF

    ; Check exit reasons
    cmp eax, 12                 ; HLT
    je .handle_hlt

    cmp eax, 48                 ; EPT violation
    je .handle_ept_violation

    cmp eax, 2                  ; Triple fault
    je .handle_triple_fault

    cmp eax, 33                 ; Invalid guest state
    je .handle_invalid_state

    ; Unknown exit
    mov rsi, msg_unknown_exit
    call print_string_64
    call print_hex_64
    jmp .halt

.handle_hlt:
    mov rsi, msg_guest_hlt
    call print_string_64
    jmp .halt

.handle_ept_violation:
    mov rsi, msg_ept_violation
    call print_string_64
    ; Read guest physical address
    mov rdx, 0x2400
    vmread rax, rdx
    call print_hex_64
    jmp .halt

.handle_triple_fault:
    mov rsi, msg_triple_fault
    call print_string_64
    jmp .halt

.handle_invalid_state:
    mov rsi, msg_invalid_state
    call print_string_64
    mov rdx, 0x4400
    vmread rax, rdx
    call print_hex_64
    jmp .halt

.halt:
    cli
    hlt

;-----------------------------------------------------------------------------
; print_string_64: Print null-terminated string (64-bit)
;-----------------------------------------------------------------------------
print_string_64:
    push rax
    push rbx

    mov rbx, 0xB8000
    mov ah, 0x0F

.loop:
    lodsb
    test al, al
    jz .done
    mov [rbx], ax
    add rbx, 2
    jmp .loop

.done:
    pop rbx
    pop rax
    ret

;-----------------------------------------------------------------------------
; print_hex_64: Print RAX as hexadecimal
;-----------------------------------------------------------------------------
print_hex_64:
    push rax
    push rbx
    push rcx
    push rdx

    mov rbx, 0xB8000
    ; Find current position (after last character)
.find_pos:
    cmp word [rbx], 0
    je .found_pos
    add rbx, 2
    cmp rbx, 0xB8FA0
    jae .done
    jmp .find_pos

.found_pos:
    ; Print "0x"
    mov word [rbx], 0x0F30
    add rbx, 2
    mov word [rbx], 0x0F78
    add rbx, 2

    ; Print 16 hex digits
    mov rcx, 16
.hex_loop:
    rol rax, 4
    mov dl, al
    and dl, 0x0F
    add dl, '0'
    cmp dl, '9'
    jbe .digit_ok
    add dl, 7               ; 'A'-'9'-1
.digit_ok:
    mov dh, 0x0F
    mov [rbx], dx
    add rbx, 2
    loop .hex_loop

.done:
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

;-----------------------------------------------------------------------------
; Strings
;-----------------------------------------------------------------------------
msg_stage2:             db 'Stage 2: Hypervisor loader', 13, 10, 0
msg_no_long_mode:       db 'ERROR: Long mode not supported!', 13, 10, 0
msg_no_vmx:             db 'ERROR: VT-x not supported!', 13, 10, 0
msg_vmx_ok:             db 'VT-x supported', 13, 10, 0
msg_long_mode:          db 'Long mode enabled', 0
msg_vmxon_ok:           db 'VMXON executed successfully!', 0
msg_vmxon_fail:         db 'VMXON failed!', 0
msg_vmcs_fail:          db 'VMCS initialization failed!', 0
msg_vmcs_config_fail:   db 'VMCS configuration failed!', 0
msg_launching_guest:    db 'Launching guest...', 0
msg_vmlaunch_fail:      db 'VMLAUNCH failed!', 0
msg_guest_hlt:          db 'Guest executed HLT successfully!', 0
msg_unknown_exit:       db 'Unknown VM exit: ', 0
msg_ept_violation:      db 'EPT violation at GPA: ', 0
msg_triple_fault:       db 'Guest triple fault!', 0
msg_invalid_state:      db 'Invalid guest state, error: ', 0
msg_driver_done:        db 'Pixel driver execution complete.', 0

;-----------------------------------------------------------------------------
; execute_pixel_driver: Emulates a GPU executing a pixel-encoded driver.
; Reads the embedded serial_driver.pxi and outputs characters to COM1.
;-----------------------------------------------------------------------------
execute_pixel_driver:
    push rsi
    push rax
    push rdx

    mov rsi, serial_driver_data
.loop:
    cmp rsi, serial_driver_end
    jge .done

    ; Each pixel is 4 bytes (RGBA). The character is in the first byte (R).
    mov al, [rsi]

    ; Output character to serial port
    mov dx, 0x3F8 ; COM1
    out dx, al

    ; Wait for transmit buffer to be empty
    push rax
.wait:
    mov dx, 0x3F8 + 5 ; Line Status Register
    in al, dx
    and al, 0x20      ; Check THRE bit
    jz .wait
    pop rax

    add rsi, 4 ; Move to the next pixel
    jmp .loop

.done:
    pop rdx
    pop rax
    pop rsi
    ret

;-----------------------------------------------------------------------------
; Embedded pixel driver data
;-----------------------------------------------------------------------------
align 16
serial_driver_data:
incbin "build/serial_driver.pxi"
serial_driver_end:

;-----------------------------------------------------------------------------
; Pad to 20KB
;-----------------------------------------------------------------------------
times 20480-($-$$) db 0
