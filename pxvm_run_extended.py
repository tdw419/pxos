#!/usr/bin/env python3
"""
pxVM Extended Runner - Execute bytecode with extended VM
Supports multi-process, filesystem, IPC
"""
from __future__ import annotations
import sys
from pxvm_extended import PxVMExtended


def run_program(bytecode_file: str, output_png: str = None):
    """Run a pxVM bytecode program with extended features"""
    # Load bytecode
    with open(bytecode_file, 'rb') as f:
        bytecode = f.read()

    print(f"Loaded {len(bytecode)} bytes from {bytecode_file}")

    # Create extended VM
    vm = PxVMExtended(imperfect=True)

    # Load as init process (PID 1)
    pid = vm.load_program(bytecode)
    print(f"Created process PID {pid}")

    # Execute
    print("Executing VM...")
    cycles = vm.run(max_cycles=100000)
    print(f"Executed {cycles} instruction cycles")

    # Show process states
    for pid, proc in vm.processes.items():
        status = "HALTED" if proc.halted else ("WAITING" if proc.waiting_for_ipc else "RUNNING")
        print(f"  PID {pid}: {status}, PC={proc.pc}, R0-R7={proc.registers}")

    # Collect PXTERM output
    pxterm_lines = vm.sysout
    print(f"\nGenerated {len(pxterm_lines)} PXTERM instructions")

    if not pxterm_lines:
        print("No graphical output generated")
        return

    # Save PXTERM
    pxterm_file = bytecode_file.replace('.bin', '.pxterm')
    with open(pxterm_file, 'w') as f:
        # Add CANVAS header
        f.write("CANVAS 800 600\n")
        for line in pxterm_lines:
            f.write(line + '\n')
    print(f"Saved PXTERM to {pxterm_file}")

    # Render via PXTERM interpreter
    if output_png is None:
        output_png = bytecode_file.replace('.bin', '.png')

    print("\nRendering via PXTERM...")
    try:
        from pxos_llm_terminal import run_pxterm_file
        run_pxterm_file(pxterm_file, output=output_png, imperfect=True)
        print(f"\n✓ Program rendered successfully: {output_png}")
    except Exception as e:
        print(f"✗ Rendering failed: {e}")

    return vm


def main():
    if len(sys.argv) < 2:
        print("Usage: pxvm_run_extended.py <program.bin> [output.png]")
        sys.exit(1)

    bytecode_file = sys.argv[1]
    output_png = sys.argv[2] if len(sys.argv) > 2 else None

    run_program(bytecode_file, output_png)


if __name__ == '__main__':
    main()
