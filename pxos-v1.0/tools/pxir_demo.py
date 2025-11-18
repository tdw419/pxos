#!/usr/bin/env python3
"""
pxIR Demo - Complete Pipeline Demonstration

Shows the complete compilation pipeline:
  Python → pxIR (high-level) → PXI Assembly (low-level) → Primitives → Binary

This demonstrates the two-level IR architecture:
  1. pxIR: Typed, SSA, matrix-aware (optimization target)
  2. PXI Assembly: Assembly-level (code generation)
"""

import sys
import json
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent))

from pxir import IRBuilder, Type, Program
from pxir.python_frontend import compile_python
from pxir.codegen_pxi import generate_pxi


def demo_manual_ir():
    """Demo: Build pxIR manually"""
    print("=" * 60)
    print("Demo 1: Manual pxIR Construction")
    print("=" * 60)

    builder = IRBuilder()

    # Create entry block
    entry = builder.create_block("entry")
    builder.set_insert_point(entry)

    # Simple addition: result = 10 + 20
    const10 = builder.fresh_value(Type.i32(), "const")
    const10.metadata = {"constant": 10}

    const20 = builder.fresh_value(Type.i32(), "const")
    const20.metadata = {"constant": 20}

    result = builder.add(const10, const20)

    # Return result
    builder.ret(result)

    # Print IR
    print("\npxIR Program:")
    print(builder.program)

    # Generate PXI Assembly
    pxi_json = generate_pxi(builder.program)
    print("\nGenerated PXI Assembly:")
    print(json.dumps(pxi_json, indent=2))

    return builder.program


def demo_python_compile():
    """Demo: Compile Python to pxIR"""
    print("\n")
    print("=" * 60)
    print("Demo 2: Python → pxIR Compilation")
    print("=" * 60)

    python_source = """
def compute(x, y):
    result = x + y * 2
    return result
"""

    print("\nPython Source:")
    print(python_source)

    # Compile
    program = compile_python(python_source, "compute")

    print("\npxIR Program:")
    print(program)

    # Generate PXI Assembly
    pxi_json = generate_pxi(program)
    print("\nGenerated PXI Assembly:")
    print(json.dumps(pxi_json, indent=2))

    return program


def demo_matrix_ops():
    """Demo: Matrix operations in pxIR"""
    print("\n")
    print("=" * 60)
    print("Demo 3: Matrix Operations")
    print("=" * 60)

    python_source = """
def forward(h, W, b):
    logits = h @ W + b
    return relu(logits)
"""

    print("\nPython Source (matrix operations):")
    print(python_source)

    try:
        # Compile
        program = compile_python(python_source, "forward")

        print("\npxIR Program:")
        print(program)

        # Generate PXI Assembly
        pxi_json = generate_pxi(program)
        print("\nGenerated PXI Assembly:")
        print(json.dumps(pxi_json, indent=2))

        return program

    except Exception as e:
        print(f"\nNote: Full matrix support coming soon!")
        print(f"Current status: {e}")
        return None


def demo_print_hello():
    """Demo: Print hello world"""
    print("\n")
    print("=" * 60)
    print("Demo 4: Hello World with pxIR")
    print("=" * 60)

    python_source = """
def main():
    print("Hello from pxIR!")
    return 0
"""

    print("\nPython Source:")
    print(python_source)

    # Compile
    program = compile_python(python_source, "main")

    print("\npxIR Program:")
    print(program)

    # Generate PXI Assembly
    pxi_json = generate_pxi(program)
    print("\nGenerated PXI Assembly (excerpt):")

    # Print just the instructions, not full JSON
    for instr in pxi_json["instructions"][:10]:
        if instr["op"] == "COMMENT":
            print(f"  ; {instr['comment']}")
        else:
            ops_str = ", ".join(f"{k}={v}" for k, v in instr["operands"].items())
            comment = f" ; {instr['comment']}" if instr['comment'] else ""
            print(f"  {instr['op']} {ops_str}{comment}")

    if len(pxi_json["instructions"]) > 10:
        print(f"  ... ({len(pxi_json['instructions']) - 10} more instructions)")

    return program


def main():
    """Run all demos"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "pxIR Pipeline Demo" + " " * 24 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    print("This demonstrates the two-level IR architecture:")
    print("  1. pxIR: High-level, typed, SSA, matrix-aware")
    print("  2. PXI Assembly: Low-level, assembly instructions")
    print()

    demos = [
        ("Manual IR Construction", demo_manual_ir),
        ("Python Compilation", demo_python_compile),
        ("Matrix Operations", demo_matrix_ops),
        ("Hello World", demo_print_hello),
    ]

    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n")
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print()
    print("Architecture Summary:")
    print("  Python/C → pxIR → PXI Assembly → Primitives → Binary")
    print("                ↑           ↑")
    print("          High-level   Low-level")
    print("          Typed SSA    Assembly")
    print()
    print("Next steps:")
    print("  - Add optimization passes (constant folding, DCE)")
    print("  - Implement full matrix operations")
    print("  - Add more Python language support")
    print("  - Create pxVM backend (pxIR → bytecode)")
    print()


if __name__ == "__main__":
    main()
