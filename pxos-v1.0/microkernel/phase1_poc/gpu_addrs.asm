; gpu_addrs.asm - Global addresses for GPU BARs
BITS 64

section .data
align 8
gpu_bar0_phys:      dq 0          ; set by PCIe enumeration
gpu_bar0_virt:      dq 0          ; set by map_gpu_bars
mailbox_virt_addr:  dq 0          ; BAR0 + offset, used by mailbox_hw

; Optional: simulated mailbox fallback in low memory (identity-mapped)
sim_mailbox_phys:   dq 0x00020000 ; or whatever you already use
