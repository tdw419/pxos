#!/usr/bin/env python3
"""
pxvm/utils/layout.py

Utilities for calculating memory layouts and writing data to pixel programs.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

def calculate_matrix_rows(cols: int, rows: int, image_width: int) -> int:
    """Calculates the number of rows required to store a matrix."""
    stride = image_width - 1
    total_elements = cols * rows
    data_rows = (total_elements + stride - 1) // stride
    return 1 + data_rows # 1 for the header row

def calculate_image_size(matrices: Dict[str, Tuple[int, int]], instruction_count: int, width: int) -> Tuple[int, int]:
    """Calculates the minimum image dimensions required to store a program."""
    height = 1 # Start with 1 row for instructions
    for cols, rows in matrices.values():
        height += calculate_matrix_rows(cols, rows, width)
    return width, height

def allocate_rows(matrices: Dict[str, Tuple[int, int]], width: int, start_row: int) -> Dict[str, int]:
    """Assigns starting row indices to matrices."""
    allocations = {}
    current_row = start_row
    for name, (cols, rows) in matrices.items():
        allocations[name] = current_row
        current_row += calculate_matrix_rows(cols, rows, width)
    return allocations

def write_matrix(img: np.ndarray, row_start: int, data: np.ndarray, image_width: int):
    """Writes a matrix to the specified row in the image."""
    rows, cols = data.shape
    img[row_start, 0] = np.array([
        cols & 0xFF, (cols >> 8) & 0xFF,
        rows & 0xFF, (rows >> 8) & 0xFF
    ], dtype=np.uint8)

    flat_data = data.flatten()
    stride = image_width - 1
    for i, val in enumerate(flat_data):
        x = 1 + (i % stride)
        y = row_start + (i // stride)
        img[y, x, 0] = val

def write_quantized_matrix(
    img: np.ndarray,
    row_start: int,
    data: np.ndarray,
    image_width: int = 1024
) -> None:
    """
    Write matrix with embedded quantization metadata.

    Layout at row_start:
    - Column 0: Header [cols_lo, cols_hi, rows_lo, rows_hi]
    - Column 1: Scale bytes [s0, s1, s2, s3]
    - Column 2: Offset bytes [o0, o1, o2, o3]
    - Column 3+: Quantized data

    Args:
        img: Target image array
        row_start: Starting row index
        data: Float32 matrix data
        image_width: Image width in pixels
    """
    from pxvm.utils.quantization import (
        calculate_scale_offset,
        linear_quantize,
        pack_quantization_metadata
    )

    rows, cols = data.shape

    # 1. Compute quantization parameters
    scale, offset = calculate_scale_offset(data)

    # 2. Quantize the data
    quantized = linear_quantize(data, scale, offset)

    # 3. Write header at column 0
    img[row_start, 0] = np.array([
        cols & 0xFF,
        (cols >> 8) & 0xFF,
        rows & 0xFF,
        (rows >> 8) & 0xFF,
    ], dtype=np.uint8)

    # 4. Write quantization metadata at columns 1-2
    metadata = pack_quantization_metadata(scale, offset)
    img[row_start, 1] = metadata[0]  # Scale
    img[row_start, 2] = metadata[1]  # Offset

    # 5. Write quantized data starting at column 3
    flat_data = quantized.flatten()
    stride = image_width - 3  # Available columns per row

    for i, val in enumerate(flat_data):
        x = 3 + (i % stride)
        y = row_start + (i // stride)
        img[y, x, 0] = val

def read_quantized_matrix(
    img: np.ndarray,
    row_start: int,
    image_width: int = 1024
) -> np.ndarray:
    """
    Read quantized matrix and return dequantized float32 data.

    Args:
        img: Source image array
        row_start: Starting row index
        image_width: Image width in pixels

    Returns:
        Float32 matrix (dequantized)
    """
    from pxvm.utils.quantization import (
        unpack_quantization_metadata,
        linear_dequantize
    )

    # 1. Read header
    header = img[row_start, 0]
    cols = int(header[0]) | (int(header[1]) << 8)
    rows = int(header[2]) | (int(header[3]) << 8)

    # 2. Read quantization metadata
    metadata_pixels = img[row_start, 1:3, :]
    scale, offset = unpack_quantization_metadata(metadata_pixels)

    # 3. Read quantized data
    total_elements = cols * rows
    stride = image_width - 3

    quantized = np.zeros(total_elements, dtype=np.uint8)
    for i in range(total_elements):
        x = 3 + (i % stride)
        y = row_start + (i // stride)
        quantized[i] = img[y, x, 0]

    # 4. Reshape and dequantize
    quantized_matrix = quantized.reshape(rows, cols)
    float_matrix = linear_dequantize(quantized_matrix, scale, offset)

    return float_matrix

def calculate_matrix_rows_quantized(cols: int, rows: int, image_width: int) -> int:
    """
    Calculate rows needed for quantized matrix (with metadata).

    Layout: 1 header row + ceil(cols*rows / (width-3)) data rows
    """
    total_elements = cols * rows
    stride = image_width - 3  # Columns 0,1,2 reserved
    data_rows = (total_elements + stride - 1) // stride
    return 1 + data_rows
