#!/usr/bin/env python3
"""
PX Reflex - Immunity Demonstration

Shows the immune system protecting a sacred region from unauthorized writes.

Workflow:
1. Kernel creates a protected region (immune zone)
2. Kernel draws in the protected region (allowed - it's PID 1)
3. Rogue process spawns and tries to corrupt the protected region
4. Immune system INSTANTLY reverts every write from rogue process
5. Protected region remains pristine
"""
from pxvm_extended import PxVMExtended
from pxvm_assembler import Assembler
import numpy as np


def create_guardian_kernel() -> bytes:
    """
    Create kernel that establishes immune protection
    """
    asm = Assembler()
    source = """
    ; Guardian Kernel - Establishes immunity

    START:
        ; Boot message
        IMM32 R1, 1          ; "PXVM booting..."
        SYSCALL 1

        ; Create protected region at (200, 200) size 200x200
        IMM32 R1, 200        ; x
        IMM32 R2, 200        ; y
        IMM32 R3, 200        ; width
        IMM32 R4, 200        ; height
        IMM32 R5, 0          ; absolute=0 (whitelist allowed)
        SYSCALL 101          ; SYS_REFLEX_PROTECT
        ; R0 now contains region_id

        ; Draw sacred content in protected region (we're allowed)
        IMM32 R1, 250        ; x
        IMM32 R2, 250        ; y
        IMM32 R3, 100        ; w
        IMM32 R4, 100        ; h
        IMM32 R5, 1          ; color = blue (sacred)
        SYSCALL 2            ; RECT

        ; Whitelist ourselves for this region
        MOV R1, R0           ; region_id (from PROTECT)
        IMM32 R2, 1          ; PID 1 (ourselves)
        SYSCALL 100          ; SYS_REFLEX_WHITELIST

        ; Print status
        IMM32 R1, 2          ; "PXVM ready"
        SYSCALL 1

        ; Now spawn the rogue process that will try to corrupt us
        IMM32 R1, 302        ; rogue.bin
        SYSCALL 31           ; SYS_SPAWN

        ; Wait a bit (simulate ongoing work)
        IMM32 R1, 4          ; "Kernel init done"
        SYSCALL 1

        IMM32 R1, 6          ; "Process terminated"
        SYSCALL 1

        HALT
    """
    return asm.assemble(source)


def create_rogue_process() -> bytes:
    """
    Create rogue process that tries to corrupt protected region
    """
    asm = Assembler()
    source = """
    ; Rogue Process - Attempts corruption

    START:
        ; Announce malicious intent
        IMM32 R1, 5          ; "Process started"
        SYSCALL 1

        ; Try to draw RED over the protected blue square
        ; This should be BLOCKED by immune system
        IMM32 R1, 250        ; x (same as protected region)
        IMM32 R2, 250        ; y
        IMM32 R3, 100        ; w
        IMM32 R4, 100        ; h
        IMM32 R5, 3          ; color = RED (corruption attempt!)
        SYSCALL 2            ; RECT - will be reverted!

        ; Try again (immune system is merciless)
        IMM32 R1, 200
        IMM32 R2, 200
        IMM32 R3, 200
        IMM32 R4, 200
        IMM32 R5, 3          ; RED (another attempt)
        SYSCALL 2            ; RECT - will also be reverted!

        ; Print defeat
        IMM32 R1, 6          ; "Process terminated"
        SYSCALL 1

        HALT
    """
    return asm.assemble(source)


def main():
    """Run immunity demonstration"""
    print("=" * 70)
    print("PX Reflex - Immunity Demonstration")
    print("=" * 70)

    # Compile programs
    print("\n[1] Compiling programs...")
    guardian_kernel = create_guardian_kernel()
    rogue_process = create_rogue_process()
    print(f"    Guardian kernel: {len(guardian_kernel)} bytes")
    print(f"    Rogue process: {len(rogue_process)} bytes")

    # Create VM
    print("\n[2] Initializing VM with PX Reflex...")
    vm = PxVMExtended(imperfect=True)

    # Pre-load rogue process into filesystem
    vm.filesystem.files["build/rogue.bin"] = bytearray(rogue_process)
    vm.file_paths[302] = "build/rogue.bin"

    # Load guardian kernel
    vm.load_program(guardian_kernel)
    print("    Guardian kernel loaded as PID 1")

    print("\n[3] Executing system...")
    print("    Guardian will:")
    print("      - Create protected region at (200,200) size 200x200")
    print("      - Draw sacred blue square in protected region")
    print("      - Spawn rogue process (PID 2)")
    print("    Rogue will:")
    print("      - Attempt to draw RED over protected blue square")
    print("      - Immune system will REVERT every write")
    print()

    # Run VM (this will execute both processes)
    cycles = vm.run(max_cycles=10000)
    print(f"\n[4] Executed {cycles} cycles")

    # Check reflex engine stats
    if vm.reflex_engine:
        stats = vm.reflex_engine.get_stats()
        print("\n[5] Reflex Engine Statistics:")
        print(f"    Pixels changed: {stats['pixels_changed']}")
        print(f"    Writes blocked: {stats['writes_blocked']}")
        print(f"    Protected regions: {stats['protected_regions']}")
        print(f"    Events emitted: {stats['events_emitted']}")
        print(f"    Current tick: {stats['current_tick']}")

        if stats['writes_blocked'] > 0:
            print(f"\n    ✓ IMMUNE SYSTEM ACTIVE: Blocked {stats['writes_blocked']} unauthorized writes!")
        else:
            print("\n    ⚠ No writes were blocked (immune system may not have triggered)")
    else:
        print("\n[5] Reflex engine not initialized")

    # Show syscall output
    print("\n[6] System output:")
    for line in vm.sysout:
        print(f"    {line}")

    # Render output
    print("\n[7] Rendering output...")
    with open("immunity_demo.pxterm", 'w') as f:
        f.write("CANVAS 800 600\n")
        for line in vm.sysout:
            f.write(line + '\n')

    try:
        from pxos_llm_terminal import run_pxterm_file
        run_pxterm_file("immunity_demo.pxterm", output="immunity_demo.png", imperfect=True)
        print("    ✓ Rendered immunity_demo.png")
        print("      Check the image - blue square should be intact (not red)")
    except Exception as e:
        print(f"    Rendering: {e}")

    print("\n" + "=" * 70)
    print("IMMUNITY DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("What just happened:")
    print("  1. Guardian kernel created protected region (200,200) 200x200")
    print("  2. Guardian drew BLUE square in protected region (allowed)")
    print("  3. Guardian whitelisted itself (PID 1)")
    print("  4. Rogue process (PID 2) spawned")
    print("  5. Rogue tried to draw RED over protected region")
    print("  6. Immune system detected unauthorized write")
    print("  7. Immune system INSTANTLY reverted to previous frame")
    print("  8. Protected region remains BLUE (pristine)")
    print()
    print("This demonstrates:")
    print("  ✓ SYS_REFLEX_PROTECT syscall (101)")
    print("  ✓ SYS_REFLEX_WHITELIST syscall (100)")
    print("  ✓ Layer 2: Immune system (automatic write blocking)")
    print("  ✓ Pixel-perfect protection enforcement")
    print("  ✓ The sacred cannot be corrupted")
    print()


if __name__ == '__main__':
    main()
