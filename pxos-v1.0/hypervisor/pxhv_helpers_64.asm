; pxHV Helper Functions (64-bit)

enable_vmx:
    mov eax, cr4
    or eax, 1 << 13
    mov cr4, eax
    mov ecx, 0x3A
    rdmsr
    or eax, 5
    wrmsr
    ret

vmxon_init:
    mov ecx, 0x480
    rdmsr
    mov [VMXON_REGION], eax
    ret

vmxon_exec:
    mov rax, VMXON_REGION
    vmxon [rax]
    jc .vmxon_fail
    ret
.vmxon_fail:
    mov rsi, msg_vmxon_fail
    call print_string_64
    cli
    hlt

print_string_64:
    mov rdi, 0xB8000
.loop:
    lodsb
    test al, al
    jz .done
    mov [rdi], al
    inc rdi
    mov byte [rdi], 0x0F
    inc rdi
    jmp .loop
.done:
    ret

msg_vmxon_fail    db 'VMXON failed!', 13, 10, 0
