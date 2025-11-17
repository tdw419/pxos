#!/usr/bin/env python3
"""
pxvm/examples/run_font_code_test.py

T016: Execute and verify the Font-Code test program.

This validates that ASCII opcodes (M, A, R, H) are correctly
interpreted by the pxVM interpreter.
"""

from pathlib import Path
import numpy as np
from PIL import Image

from pxvm.core.interpreter import run_program
from pxvm.core.opcodes import opcode_to_char


def main() -> None:
    """Execute Font-Code test program and verify results."""
    test_path = Path(__file__).parent / "font_code_test.pxi"

    if not test_path.exists():
        print(f"ERROR: Test program not found: {test_path}")
        print("Run: python3 pxvm/examples/make_font_code_test.py")
        return 1

    # Load program
    img = np.array(Image.open(test_path).convert('RGBA'), dtype=np.uint8)

    print("=" * 70)
    print(" FONT-CODE TEST EXECUTION (T016)")
    print("=" * 70)
    print()

    # Show opcodes
    opcode_0 = int(img[0, 0, 0])
    opcode_1 = int(img[0, 1, 0])

    print("Instruction Row:")
    print(f"  [0,0]: Opcode {opcode_0} = '{chr(opcode_0)}' = {opcode_to_char(opcode_0)}")
    print(f"  [0,1]: Opcode {opcode_1} = '{chr(opcode_1)}' = {opcode_to_char(opcode_1)}")
    print()

    # Execute
    print("Executing program...")
    result = run_program(img)
    print()

    # Extract result matrix C from row 3 (as specified in instruction)
    row_c = int(img[0, 0, 3])  # Get C row from instruction
    print(f"Output row C: {row_c}")
    print()

    from pxvm.utils.layout import read_quantized_matrix

    try:
        C_data = read_quantized_matrix(result, row_c)
        print(f"Output Matrix C ({C_data.shape[0]}×{C_data.shape[1]}):")
        C_data_int = np.round(C_data).astype(np.int32)
    except Exception as e:
        print(f"ERROR reading result matrix: {e}")
        return 1
    print(C_data_int)
    print()

    # Verify against expected result
    expected = np.array([[1, 2], [3, 4]], dtype=np.int32)

    print("Expected Result:")
    print(expected)
    print()

    # Check if values are close (within rounding error)
    if np.allclose(C_data, expected, atol=0.1):
        print("✅ SUCCESS: Font-Code MatMul executed correctly!")
        print()
        print("Font-Code Protocol Validated:")
        print("  - ASCII opcode 'M' (77) executes as MATMUL")
        print("  - ASCII opcode 'H' (72) executes as HALT")
        print("  - Interpreter correctly migrates and executes Font-Code programs")
        print("  - Quantization/dequantization preserves values")
        print()
        return 0
    else:
        print("❌ FAILURE: Execution mismatch")
        print(f"  Expected: {expected.tolist()}")
        print(f"  Got:      {C_data_int.tolist()}")
        print()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
