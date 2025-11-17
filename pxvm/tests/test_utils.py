#!/usr/bin/env python3
"""
pxvm/tests/test_utils.py

Tests for pxvm.utils layout and validation utilities.
"""

from __future__ import annotations

import numpy as np
import pytest

from pxvm.core.opcodes import OP_HALT, OP_MATMUL, OP_ADD
from pxvm.utils.layout import (
    calculate_matrix_rows,
    calculate_image_size,
    allocate_rows,
    write_matrix,
    read_matrix,
)
from pxvm.utils.validation import (
    validate_matrix_header,
    validate_instruction,
    validate_program_structure,
    validate_matmul_compatibility,
)


def test_calculate_matrix_rows():
    """Test matrix row calculation."""
    # Small matrix: 3×2 = 6 elements
    # With image_width=16, stride=15
    # Needs: 1 row header + 1 row data
    assert calculate_matrix_rows(3, 2, image_width=16) == 2

    # Larger matrix: 128×128 = 16,384 elements
    # With image_width=256, stride=255
    # Needs: 1 header + ceil(16384/255) = 1 + 65 = 66 rows
    assert calculate_matrix_rows(128, 128, image_width=256) == 66


def test_calculate_image_size():
    """Test image size calculation."""
    matrices = {
        'A': (3, 2),      # 6 elements → 2 rows (header + ceil(6/15))
        'B': (2, 3),      # 6 elements → 2 rows
        'C': (2, 2),      # 4 elements → 2 rows
    }

    width, height = calculate_image_size(matrices, width=16)

    assert width == 16
    # Row 0: instructions
    # Rows 1-2: A (3×2) = 2 rows
    # Rows 3-4: B (2×3) = 2 rows
    # Rows 5-6: C (2×2) = 2 rows
    # Total: 7 rows
    assert height == 7


def test_allocate_rows():
    """Test row allocation."""
    matrices = {
        'A': (3, 2),  # 2 rows total
        'B': (2, 3),  # 2 rows total
        'C': (2, 2),  # 2 rows total
    }

    allocations = allocate_rows(matrices, width=16, start_row=1)

    assert allocations['A'] == 1
    assert allocations['B'] == 3  # After A (2 rows)
    assert allocations['C'] == 5  # After B (2 rows)


def test_write_read_matrix():
    """Test writing and reading matrices."""
    img = np.zeros((16, 16, 4), dtype=np.uint8)

    # Write a small matrix
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    write_matrix(img, row_start=1, data=data, quantize=True)

    # Check header
    assert img[1, 0, 0] == 3  # cols_low
    assert img[1, 0, 1] == 0  # cols_high
    assert img[1, 0, 2] == 2  # rows_low
    assert img[1, 0, 3] == 0  # rows_high

    # Read back
    result = read_matrix(img, row_start=1)

    assert result.shape == (2, 3)
    # Values should be quantized but preserve relative order
    assert result[0, 0] < result[0, 2]  # 1 < 3
    assert result[1, 0] < result[1, 2]  # 4 < 6


def test_write_matrix_no_quantize():
    """Test writing matrix without quantization."""
    img = np.zeros((16, 16, 4), dtype=np.uint8)

    data = np.array([[10, 20], [30, 40]], dtype=np.uint8)

    write_matrix(img, row_start=1, data=data, quantize=False)

    result = read_matrix(img, row_start=1)

    # Should be exact match (no quantization)
    np.testing.assert_array_equal(result, data)


def test_validate_matrix_header():
    """Test matrix header validation."""
    img = np.zeros((16, 16, 4), dtype=np.uint8)

    # Write valid header: 3×2
    img[1, 0] = [3, 0, 2, 0]

    cols, rows = validate_matrix_header(img, 1)
    assert (cols, rows) == (3, 2)

    # Validate with expected shape
    validate_matrix_header(img, 1, expected_shape=(3, 2))

    # Should fail with wrong expected shape
    with pytest.raises(ValueError, match="Shape mismatch"):
        validate_matrix_header(img, 1, expected_shape=(2, 3))


def test_validate_instruction():
    """Test instruction validation."""
    img = np.zeros((16, 16, 4), dtype=np.uint8)

    # Write valid instruction
    img[0, 0] = [OP_MATMUL, 1, 2, 3]

    opcode, arg0, arg1, arg2 = validate_instruction(img, 0)
    assert opcode == OP_MATMUL
    assert arg0 == 1
    assert arg1 == 2
    assert arg2 == 3

    # Invalid opcode
    img[0, 1] = [99, 0, 0, 0]
    with pytest.raises(ValueError, match="Invalid opcode"):
        validate_instruction(img, 1)


def test_validate_program_structure():
    """Test program structure validation."""
    img = np.zeros((16, 16, 4), dtype=np.uint8)

    # Write simple program: MATMUL, ADD, HALT
    img[0, 0] = [OP_MATMUL, 1, 2, 3]
    img[0, 1] = [OP_ADD, 3, 4, 5]
    img[0, 2] = [OP_HALT, 0, 0, 0]

    result = validate_program_structure(img)

    assert result['valid'] is True
    assert result['instruction_count'] == 2  # Before HALT
    assert len(result['errors']) == 0

    # Program with expected instructions
    expected = [
        (OP_MATMUL, 1, 2, 3),
        (OP_ADD, 3, 4, 5),
        (OP_HALT, 0, 0, 0),
    ]

    result = validate_program_structure(img, expected_instructions=expected)
    assert result['valid'] is True


def test_validate_program_missing_halt():
    """Test validation catches missing HALT."""
    img = np.zeros((16, 16, 4), dtype=np.uint8)

    # Fill entire row 0 with non-HALT instructions (no room for HALT)
    for x in range(16):
        img[0, x] = [OP_MATMUL, 1, 2, 3]

    result = validate_program_structure(img)

    assert result['valid'] is False
    assert any('HALT' in err for err in result['errors'])


def test_validate_matmul_compatibility():
    """Test matrix multiplication compatibility check."""
    img = np.zeros((16, 16, 4), dtype=np.uint8)

    # A: 2×3 (M=2, K=3)
    img[1, 0] = [3, 0, 2, 0]

    # B: 3×4 (K=3, N=4) - compatible!
    img[2, 0] = [4, 0, 3, 0]

    assert validate_matmul_compatibility(img, row_a=1, row_b=2) is True

    # C: 2×4 (incompatible with A)
    img[3, 0] = [4, 0, 2, 0]

    assert validate_matmul_compatibility(img, row_a=1, row_b=3) is False


def test_write_matrix_vector():
    """Test writing 1D vector."""
    img = np.zeros((16, 16, 4), dtype=np.uint8)

    # 1D vector should be treated as 1×N
    vec = np.array([10, 20, 30, 40], dtype=np.uint8)

    write_matrix(img, row_start=1, data=vec, quantize=False)

    # Check header: should be 4×1 (cols=4, rows=1)
    assert img[1, 0, 0] == 4  # cols_low
    assert img[1, 0, 2] == 1  # rows_low

    result = read_matrix(img, row_start=1)

    # Should be 1×4 matrix
    assert result.shape == (1, 4)
    np.testing.assert_array_equal(result[0], vec)


def test_write_matrix_overflow():
    """Test that writing too large a matrix raises error."""
    img = np.zeros((16, 16, 4), dtype=np.uint8)

    # Try to write 200×200 matrix into 16×16 image
    large_data = np.ones((200, 200), dtype=np.uint8)

    with pytest.raises(ValueError, match="exceeds image height"):
        write_matrix(img, row_start=1, data=large_data, quantize=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
