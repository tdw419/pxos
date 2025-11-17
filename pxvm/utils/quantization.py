#!/usr/bin/env python3
"""
pxvm/utils/quantization.py

Implements the Linear Quantization (Scale/Offset) protocol used to store
float32 parameters in the pixel matrix metadata.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


# --- Float32 / Uint8 Conversion Tools ---

def float_to_bytes(value: np.float32) -> np.ndarray:
    """Converts a float32 into its 4 constituent uint8 bytes."""
    return np.frombuffer(value.tobytes(), dtype=np.uint8)


def bytes_to_float(bytes_array: np.ndarray) -> np.float32:
    """Converts a 4-byte uint8 array back into a single float32."""
    return np.frombuffer(bytes_array.astype(np.uint8).tobytes(), dtype=np.float32)[0]


# --- Protocol Specific Packing/Unpacking ---

def pack_quantization_metadata(scale: np.float32, offset: np.float32) -> np.ndarray:
    """
    Packs two float32 values (Scale, Offset) into 8 uint8 values across two pixels.

    Pixel 1 (Scale): (R0, G0, B0, A0)
    Pixel 2 (Offset): (R1, G1, B1, A1)
    """
    scale_bytes = float_to_bytes(scale) # [b0, b1, b2, b3]
    offset_bytes = float_to_bytes(offset) # [b4, b5, b6, b7]

    # Pixel 1 (X=1): Stores Scale bytes 0, 1, 2, 3
    pixel_scale = scale_bytes

    # Pixel 2 (X=2): Stores Offset bytes 0, 1, 2, 3
    pixel_offset = offset_bytes

    # Combined 2x4 array of pixels, ready to be written starting at (1, Row)
    # The Assembler/Layout must handle writing this 2-pixel wide block
    return np.stack([pixel_scale, pixel_offset], axis=0).astype(np.uint8)


def unpack_quantization_metadata(pixel_array: np.ndarray) -> Tuple[np.float32, np.float32]:
    """
    Unpacks two float32 values (Scale, Offset) from a 2-pixel array (8 uint8 values).

    Args: pixel_array: 2x4 numpy array containing the raw uint8 pixel data.
    """
    # Pixel 1 contains Scale bytes
    pixel_scale = pixel_array[0]
    scale = bytes_to_float(pixel_scale)

    # Pixel 2 contains Offset bytes
    pixel_offset = pixel_array[1]
    offset = bytes_to_float(pixel_offset)

    return scale, offset


# --- Linear Quantization Functions (for Assembler) ---

def calculate_scale_offset(data: np.ndarray) -> Tuple[np.float32, np.float32]:
    """Calculates the optimal linear scale and offset for a matrix."""
    min_val = data.min()
    max_val = data.max()

    # Standard linear quantization to map [min, max] -> [0, 255]
    if max_val == min_val:
        scale = np.float32(1.0)
    else:
        # Scale = 255 / (max - min)
        scale = np.float32(255.0 / (max_val - min_val))

    # Offset is the min_val in the float domain
    offset = np.float32(min_val)

    return scale, offset


def linear_quantize(data: np.ndarray, scale: np.float32, offset: np.float32) -> np.ndarray:
    """Applies the calculated scale and offset to the float data, resulting in uint8."""
    # q = round((f - offset) * scale)
    q = np.round((data - offset) * scale)
    q = np.clip(q, 0, 255).astype(np.uint8)
    return q


def linear_dequantize(quantized_data: np.ndarray, scale: np.float32, offset: np.float32) -> np.ndarray:
    """Dequantizes uint8 data back to float32 using the stored scale/offset."""
    # f = q / scale + offset
    f = quantized_data.astype(np.float32) / scale + offset
    return f


if __name__ == "__main__":
    # Test suite for packing/unpacking and quantization accuracy

    # 1. Test Float Packing/Unpacking Integrity
    test_float = np.float32(3.14159)
    packed_bytes = float_to_bytes(test_float)
    unpacked_float = bytes_to_float(packed_bytes)
    assert np.allclose(test_float, unpacked_float)

    # 2. Test Quantization Logic
    data_float = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    scale, offset = calculate_scale_offset(data_float)

    assert np.allclose(scale, 255.0 / 1.5)
    assert np.allclose(offset, 0.5)

    quantized = linear_quantize(data_float, scale, offset)
    assert np.array_equal(quantized, [0, 85, 170, 255]) # Check expected discrete values

    dequantized = linear_dequantize(quantized, scale, offset)

    print("\n--- Quantization Protocol Verification ---")
    print(f"Test Float (3.14159) → {float_to_bytes(test_float)}")
    print(f"Calculated Scale/Offset: {scale:.2f}, {offset:.2f}")
    print(f"Quantized (0.5..2.0): {quantized}")
    print(f"Dequantized Check (final value): {dequantized[-1]:.2f} (Should be 2.00)")

    # 3. Test Full Metadata Packing/Unpacking
    packed_metadata = pack_quantization_metadata(scale, offset)
    unpacked_scale, unpacked_offset = unpack_quantization_metadata(packed_metadata)

    assert np.allclose(scale, unpacked_scale)
    assert np.allclose(offset, unpacked_offset)

    print("\nMetadata Packing/Unpacking Verified: Protocol is sound.")
