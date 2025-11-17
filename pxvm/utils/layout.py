#!/usr/bin/env python3
"""
pxvm/utils/layout.py

Layout utilities for pixel programs.

These help calculate image sizes and row allocations for matrices,
but they operate ON pixels - they don't hide the pixel representation.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def calculate_matrix_rows(cols: int, rows: int, image_width: int) -> int:
    """
    Calculate how many image rows a matrix occupies.

    Args:
        cols: Matrix columns
        rows: Matrix rows
        image_width: Image width in pixels

    Returns:
        Number of image rows needed (including header row)
    """
    total_elements = cols * rows
    stride = image_width - 1  # Column 0 is header

    # How many rows for data
    data_rows = (total_elements + stride - 1) // stride  # Ceiling division

    # +1 for header row
    return 1 + data_rows


def calculate_image_size(
    matrices: Dict[str, Tuple[int, int]],
    instruction_count: int = 1,
    width: int = 256,
) -> Tuple[int, int]:
    """
    Calculate minimum image size needed for a pixel program.

    Args:
        matrices: Dict mapping name → (cols, rows)
        instruction_count: Number of instruction pixels needed
        width: Desired image width

    Returns:
        (width, height) tuple
    """
    # Row 0: Instructions
    total_rows = 1

    # Each matrix needs header + data rows
    for name, (cols, rows) in matrices.items():
        matrix_rows = calculate_matrix_rows(cols, rows, width)
        total_rows += matrix_rows

    # Round up to next power of 2 for efficiency (optional)
    height = total_rows

    return (width, height)


def allocate_rows(
    matrices: Dict[str, Tuple[int, int]],
    width: int = 256,
    start_row: int = 1,
) -> Dict[str, int]:
    """
    Allocate row assignments for matrices.

    Args:
        matrices: Dict mapping name → (cols, rows)
        width: Image width
        start_row: First available row (default 1, after instructions)

    Returns:
        Dict mapping name → row_start
    """
    allocations = {}
    current_row = start_row

    for name, (cols, rows) in matrices.items():
        allocations[name] = current_row
        matrix_rows = calculate_matrix_rows(cols, rows, width)
        current_row += matrix_rows

    return allocations


def write_matrix(
    img: np.ndarray,
    row_start: int,
    data: np.ndarray,
    quantize: bool = True,
) -> None:
    """
    Write matrix with header to pixel image.

    This is a convenience wrapper around the encoding logic,
    but it operates directly on pixels - no abstraction.

    Args:
        img: RGBA image array to write into
        row_start: Starting row for this matrix
        data: 2D numpy array (or 1D for vectors)
        quantize: If True, quantize float32 to uint8 (0-255)
    """
    height, width, _ = img.shape

    # Handle 1D vectors
    if data.ndim == 1:
        data = data.reshape(1, -1)

    rows, cols = data.shape

    # Write header: (cols_low, cols_high, rows_low, rows_high)
    img[row_start, 0, 0] = cols & 0xFF
    img[row_start, 0, 1] = (cols >> 8) & 0xFF
    img[row_start, 0, 2] = rows & 0xFF
    img[row_start, 0, 3] = (rows >> 8) & 0xFF

    # Flatten data (row-major)
    flat = data.flatten()

    if quantize:
        # Quantize float32 to uint8
        data_min = flat.min()
        data_max = flat.max()

        if data_max - data_min > 0:
            flat_q = ((flat - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        else:
            flat_q = np.zeros_like(flat, dtype=np.uint8)
    else:
        flat_q = flat.astype(np.uint8)

    # Write data starting at column 1, wrapping rows
    stride = width - 1  # Column 0 is header

    for i, val in enumerate(flat_q):
        x = 1 + (i % stride)
        y = row_start + (i // stride)

        if y >= height:
            raise ValueError(f"Matrix data exceeds image height at element {i}")

        img[y, x, 0] = val  # R channel only
        # G, B, A remain 0


def read_matrix(
    img: np.ndarray,
    row_start: int,
) -> np.ndarray:
    """
    Read matrix from pixel image.

    Args:
        img: RGBA image array
        row_start: Starting row for this matrix

    Returns:
        2D numpy array of uint8 values
    """
    height, width, _ = img.shape

    # Read header
    hdr = img[row_start, 0]
    cols = int(hdr[0]) + (int(hdr[1]) << 8)
    rows = int(hdr[2]) + (int(hdr[3]) << 8)

    # Read data
    total_elements = cols * rows
    stride = width - 1

    flat = np.zeros(total_elements, dtype=np.uint8)

    for i in range(total_elements):
        x = 1 + (i % stride)
        y = row_start + (i // stride)

        if y >= height:
            raise ValueError(f"Matrix data exceeds image height")

        flat[i] = img[y, x, 0]

    # Reshape to original matrix shape
    return flat.reshape(rows, cols)


def calculate_matrix_rows_quantized(cols: int, rows: int, image_width: int) -> int:
    """
    Calculate how many image rows a quantized matrix occupies.

    Quantized layout uses:
    - Column 0: Header (cols, rows)
    - Columns 1-2: Quantization metadata (scale, offset)
    - Columns 3+: Quantized data

    Args:
        cols: Matrix columns
        rows: Matrix rows
        image_width: Image width in pixels

    Returns:
        Number of image rows needed (including header row)
    """
    total_elements = cols * rows
    stride = image_width - 3  # Columns 0-2 reserved for header + metadata

    # How many rows for data
    data_rows = (total_elements + stride - 1) // stride  # Ceiling division

    # +1 for header row
    return 1 + data_rows


def write_quantized_matrix(
    img: np.ndarray,
    row_start: int,
    data: np.ndarray,
) -> None:
    """
    Write matrix with quantization metadata to pixel image.

    Layout at row_start:
    - Column 0: Header [cols_lo, cols_hi, rows_lo, rows_hi]
    - Column 1: Scale bytes [s0, s1, s2, s3]
    - Column 2: Offset bytes [o0, o1, o2, o3]
    - Column 3+: Quantized data

    Args:
        img: RGBA image array to write into
        row_start: Starting row for this matrix
        data: 2D numpy array (float32)
    """
    from pxvm.utils.quantization import (
        calculate_scale_offset,
        linear_quantize,
        pack_quantization_metadata
    )

    height, width, _ = img.shape

    # Handle 1D vectors
    if data.ndim == 1:
        data = data.reshape(1, -1)

    rows, cols = data.shape

    # 1. Compute quantization parameters
    scale, offset = calculate_scale_offset(data)

    # 2. Quantize the data
    quantized = linear_quantize(data, scale, offset)

    # 3. Write header at column 0
    img[row_start, 0, 0] = cols & 0xFF
    img[row_start, 0, 1] = (cols >> 8) & 0xFF
    img[row_start, 0, 2] = rows & 0xFF
    img[row_start, 0, 3] = (rows >> 8) & 0xFF

    # 4. Write quantization metadata at columns 1-2
    metadata = pack_quantization_metadata(scale, offset)
    img[row_start, 1] = metadata[0]  # Scale (4 bytes)
    img[row_start, 2] = metadata[1]  # Offset (4 bytes)

    # 5. Write quantized data starting at column 3
    flat_data = quantized.flatten()
    stride = width - 3  # Available columns per row

    for i, val in enumerate(flat_data):
        x = 3 + (i % stride)
        y = row_start + (i // stride)

        if y >= height:
            raise ValueError(f"Matrix data exceeds image height at element {i}")

        img[y, x, 0] = val  # R channel only
        # G, B, A remain 0


def read_quantized_matrix(
    img: np.ndarray,
    row_start: int,
) -> np.ndarray:
    """
    Read quantized matrix and return dequantized float32 data.

    Args:
        img: RGBA image array
        row_start: Starting row for this matrix

    Returns:
        2D numpy array of float32 values (dequantized)
    """
    from pxvm.utils.quantization import (
        unpack_quantization_metadata,
        linear_dequantize
    )

    height, width, _ = img.shape

    # 1. Read header from column 0
    hdr = img[row_start, 0]
    cols = int(hdr[0]) | (int(hdr[1]) << 8)
    rows = int(hdr[2]) | (int(hdr[3]) << 8)

    # 2. Read quantization metadata from columns 1-2
    metadata_pixels = img[row_start, 1:3, :]
    scale, offset = unpack_quantization_metadata(metadata_pixels)

    # 3. Read quantized data from column 3+
    total_elements = cols * rows
    stride = width - 3

    quantized = np.zeros(total_elements, dtype=np.uint8)
    for i in range(total_elements):
        x = 3 + (i % stride)
        y = row_start + (i // stride)

        if y >= height:
            raise ValueError(f"Matrix data exceeds image height")

        quantized[i] = img[y, x, 0]

    # 4. Reshape and dequantize
    quantized_matrix = quantized.reshape(rows, cols)
    float_matrix = linear_dequantize(quantized_matrix, scale, offset)

    return float_matrix


__all__ = [
    "calculate_matrix_rows",
    "calculate_matrix_rows_quantized",
    "calculate_image_size",
    "allocate_rows",
    "write_matrix",
    "read_matrix",
    "write_quantized_matrix",
    "read_quantized_matrix",
]
