; pxHV Helper Functions (16-bit)

check_cpuid:
    pushfd
    pop eax
    mov ecx, eax
    xor eax, 1 << 21
    push eax
    popfd
    pushfd
    pop eax
    xor eax, ecx
    jz .no_cpuid
    ret
.no_cpuid:
    mov si, msg_no_cpuid
    call print_string_16
    cli
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
    ret
.no_long_mode:
    mov si, msg_no_long_mode
    call print_string_16
    cli
    hlt

check_vmx:
    mov eax, 1
    cpuid
    test ecx, 1 << 5
    jz .no_vmx
    ret
.no_vmx:
    mov si, msg_no_vmx
    call print_string_16
    cli
    hlt

print_string_16:
    lodsb
    test al, al
    jz .done
    mov ah, 0x0E
    int 0x10
    jmp print_string_16
.done:
    ret

msg_no_cpuid      db 'CPUID not supported!', 13, 10, 0
msg_no_long_mode  db 'Long mode not supported!', 13, 10, 0
msg_no_vmx        db 'VMX not supported!', 13, 10, 0
