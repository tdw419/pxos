; Flat binary microkernel for QEMU direct loading
; QEMU loads at 0x100000 in long mode with paging enabled

BITS 64
ORG 0x100000

start:
    ; QEMU has already set up long mode for us!
    ; Just setup segments and stack
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    mov rsp, 0x200000        ; Stack at 2MB

    ; Write success marker to VGA
    mov rdi, 0xB8000
    mov rax, 0x4F4B4F4F      ; "OK" in white on red
    mov [rdi], rax

    ; Print banner via VGA
    mov rsi, msg_start
    call print_vga

    ; Test mailbox communication
    call test_mailbox

    ; Halt
    cli
    hlt

print_vga:
    push rax
    push rdi
    mov rdi, 0xB8000
    add rdi, [vga_pos]
.loop:
    lodsb
    test al, al
    jz .done
    mov [rdi], al
    inc rdi
    mov byte [rdi], 0x0F     ; White on black
    inc rdi
    jmp .loop
.done:
    sub rdi, 0xB8000
    mov [vga_pos], rdi
    pop rdi
    pop rax
    ret

test_mailbox:
    ; Simulate GPU writing to mailbox
    mov dword [0x20000], 0x80000048    ; Request to print 'H'

    ; CPU reads and handles
    mov ebx, [0x20000]
    mov al, bh                          ; Extract opcode
    cmp al, 0x80
    jne .fail

    and ebx, 0xFFFF                     ; Extract char
    ; Success - write to VGA
    mov rdi, 0xB8000
    add rdi, [vga_pos]
    mov byte [rdi], bl
    inc rdi
    mov byte [rdi], 0x0A                ; Green
    mov rdi, 0xB8000
    add rdi, [vga_pos]
    add rdi, 2
    mov [vga_pos], rdi

    ; Clear mailbox
    mov dword [0x20000], 0

    mov rsi, msg_success
    call print_vga
    ret

.fail:
    mov rsi, msg_fail
    call print_vga
    ret

msg_start:    db 'pxOS Test: ', 0
msg_success:  db ' PASS', 0
msg_fail:     db ' FAIL', 0
vga_pos:      dq 0

times 4096-($-$$) db 0
