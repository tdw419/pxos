#!/usr/bin/env python3
"""
Self-Evolution Demonstration
Shows pxVM kernel replacing itself with an evolved version

Workflow:
1. Kernel v1.0 boots
2. Kernel reads evolved_kernel.bin from filesystem
3. Kernel calls SYS_SELF_MODIFY to replace itself
4. Host detects the request and restarts with new kernel
5. Kernel v2.0 boots - evolved and improved
"""
from pxvm_extended import PxVMExtended
from pxvm_assembler import Assembler
from pxvm_persistent_fs import PersistentFilesystem


def create_kernel_v1() -> bytes:
    """
    Create the original kernel that will evolve itself
    """
    asm = Assembler()
    source = """
    ; Kernel v1.0 - The original kernel
    ; This kernel will evolve itself

    START:
        ; Announce ourselves
        IMM32 R1, 1          ; "PXVM booting..."
        SYSCALL 1

        IMM32 R1, 4          ; "Kernel init done"
        SYSCALL 1

        ; Draw our generation marker (v1.0 = blue square)
        IMM32 R1, 50         ; x
        IMM32 R2, 50         ; y
        IMM32 R3, 100        ; w
        IMM32 R4, 100        ; h
        IMM32 R5, 1          ; color = blue
        SYSCALL 2            ; RECT

        ; Print evolution message
        IMM32 R1, 5          ; "Process started"
        SYSCALL 1

        ; Now evolve ourselves!
        ; Load kernel_v2.bin and request self-modification
        IMM32 R1, 301        ; file_id = 301 (kernel_v2.bin)
        SYSCALL 99           ; SYS_SELF_MODIFY

        ; If we get here, check result
        ; R0 = 1 if success, 0 if failure
        ; (In reality, successful self-modify would halt execution)

        IMM32 R1, 6          ; "Process terminated"
        SYSCALL 1

        HALT
    """
    return asm.assemble(source)


def create_kernel_v2() -> bytes:
    """
    Create the evolved kernel (v2.0)
    This is what v1.0 will transform into
    """
    asm = Assembler()
    source = """
    ; Kernel v2.0 - The evolved kernel
    ; This is the future

    START:
        ; Announce evolution
        IMM32 R1, 4          ; "Kernel init done"
        SYSCALL 1

        ; Draw evolution marker (v2.0 = light blue square at different position)
        IMM32 R1, 200        ; x (different position)
        IMM32 R2, 50         ; y
        IMM32 R3, 150        ; w (larger)
        IMM32 R4, 150        ; h (larger)
        IMM32 R5, 5          ; color = light blue (evolved)
        SYSCALL 2            ; RECT

        ; Show we're the evolved version
        IMM32 R1, 5          ; "Process started"
        SYSCALL 1

        IMM32 R1, 2          ; "PXVM ready" (new capability)
        SYSCALL 1

        IMM32 R1, 6          ; "Process terminated"
        SYSCALL 1

        HALT
    """
    return asm.assemble(source)


def main():
    """Run self-evolution demonstration"""
    print("=" * 70)
    print("pxOS Self-Evolution Demonstration")
    print("=" * 70)

    # Step 1: Setup persistent filesystem
    print("\n[1] Initializing persistent filesystem...")
    pfs = PersistentFilesystem()

    # Step 2: Create both kernel versions
    print("[2] Compiling kernel versions...")
    kernel_v1 = create_kernel_v1()
    kernel_v2 = create_kernel_v2()
    print(f"    Kernel v1.0: {len(kernel_v1)} bytes (original)")
    print(f"    Kernel v2.0: {len(kernel_v2)} bytes (evolved)")

    # Register kernels in version system
    vid1 = pfs.register_kernel_version("v1.0", kernel_v1)
    vid2 = pfs.register_kernel_version("v2.0", kernel_v2, parent_version="v1.0")
    print(f"    Registered as version IDs {vid1} and {vid2}")

    # Step 3: Run kernel v1.0 (it will try to evolve)
    print("\n[3] Booting kernel v1.0 (original)...")
    print("=" * 70)

    vm = PxVMExtended(imperfect=True)

    # Pre-populate filesystem with kernel_v2
    vm.filesystem.files["build/kernel_v2.bin"] = bytearray(kernel_v2)
    print("    Loaded kernel_v2.bin into filesystem")

    # Boot kernel v1.0
    vm.load_program(kernel_v1)
    print("    Kernel v1.0 loaded as PID 1")

    print("\n--- Kernel v1.0 execution begins ---")
    cycles = vm.run(max_cycles=10000)
    print(f"--- Kernel v1.0 execution complete ({cycles} cycles) ---\n")

    # Check if self-modification was requested
    if vm.pending_kernel_replacement:
        print("[4] SELF-MODIFICATION REQUESTED!")
        print(f"    New kernel: {len(vm.pending_kernel_replacement)} bytes")
        print(f"    Source: {vm.pending_kernel_path}")

        # Verify it's the evolved kernel
        if vm.pending_kernel_replacement == kernel_v2:
            print("    ✓ Verified: New kernel is v2.0 (evolved version)")
        else:
            print("    ✗ Warning: Kernel mismatch")

        print("\n[5] Simulating kernel replacement...")
        print("    In a real system, the host would now:")
        print("      1. Save current kernel to pxos_fs/kernels/kernel_v1.0.bin")
        print("      2. Write new kernel to pxos_fs/kernels/kernel_current.bin")
        print("      3. Reboot the VM with new kernel")
        print()
        print("    For this demo, we'll manually boot kernel v2.0:")
        print("=" * 70)

        # Create new VM and boot kernel v2.0
        vm2 = PxVMExtended(imperfect=True)
        vm2.load_program(kernel_v2)

        print("\n--- Kernel v2.0 execution begins (EVOLVED) ---")
        cycles2 = vm2.run(max_cycles=10000)
        print(f"--- Kernel v2.0 execution complete ({cycles2} cycles) ---\n")

        # Collect outputs from both runs
        print("[6] Comparing kernel outputs...")
        print("\nKernel v1.0 output:")
        for line in vm.sysout:
            print(f"    {line}")

        print("\nKernel v2.0 output (evolved):")
        for line in vm2.sysout:
            print(f"    {line}")

        # Render both
        print("\n[7] Rendering outputs...")

        # V1 output
        with open("evolution_v1.pxterm", 'w') as f:
            f.write("CANVAS 800 600\n")
            for line in vm.sysout:
                f.write(line + '\n')

        # V2 output
        with open("evolution_v2.pxterm", 'w') as f:
            f.write("CANVAS 800 600\n")
            for line in vm2.sysout:
                f.write(line + '\n')

        try:
            from pxos_llm_terminal import run_pxterm_file
            run_pxterm_file("evolution_v1.pxterm", output="evolution_v1.png", imperfect=True)
            run_pxterm_file("evolution_v2.pxterm", output="evolution_v2.png", imperfect=True)
            print("    ✓ Rendered evolution_v1.png (original kernel)")
            print("    ✓ Rendered evolution_v2.png (evolved kernel)")
        except Exception as e:
            print(f"    Rendering: {e}")

    else:
        print("[4] No self-modification requested")
        print("    (Kernel v1.0 did not call SYS_SELF_MODIFY)")

    print("\n" + "=" * 70)
    print("SELF-EVOLUTION DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("What just happened:")
    print("  1. Kernel v1.0 booted and executed")
    print("  2. Kernel v1.0 loaded kernel_v2.bin from filesystem")
    print("  3. Kernel v1.0 called SYS_SELF_MODIFY with new kernel")
    print("  4. Host received the self-modification request")
    print("  5. Kernel v2.0 booted (simulated) with evolved capabilities")
    print("  6. Both kernels executed and produced different output")
    print()
    print("This demonstrates:")
    print("  ✓ SYS_SELF_MODIFY syscall (99)")
    print("  ✓ Kernel loading new bytecode from filesystem")
    print("  ✓ Kernel requesting its own replacement")
    print("  ✓ Persistent filesystem with version tracking")
    print("  ✓ The machine can choose to evolve itself")
    print()
    print("Next step: Automatic kernel replacement on reboot")
    print("  → Then the evolution cycle completes seamlessly")
    print()


if __name__ == '__main__':
    main()
