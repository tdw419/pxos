#!/usr/bin/env python3
"""
model_to_pixels.py - Encode Neural Network Weights as Pixels

Converts pixellm_v0.npz → pixellm_v0.pxi (pixel image) + metadata

Encoding scheme:
1. Per-tensor uint8 quantization (w_min, w_max)
2. Flatten all quantized bytes
3. Pack into RGB pixels (3 bytes per pixel)
4. Save metadata (shapes, scales, offsets)

This makes model weights pixel-native - they literally live as RGB values.

Usage:
    python3 pixel_llm/core/model_to_pixels.py
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "pixel_llm" / "models"

def quantize_tensor(W: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Quantize float32 tensor to uint8.

    Returns:
        quantized: uint8 array (same shape as W)
        w_min: minimum value
        w_max: maximum value
    """
    W = W.astype("float32")
    w_min = float(W.min())
    w_max = float(W.max())

    # Avoid division by zero
    if w_max == w_min:
        w_max = w_min + 1e-6

    # Quantize to [0, 255]
    q = np.round((W - w_min) / (w_max - w_min) * 255.0)
    q = np.clip(q, 0, 255).astype("uint8")

    return q, w_min, w_max


def encode_model_to_pixels(npz_path: Path, out_pxi: Path, meta_path: Path):
    """
    Encode model weights from .npz to pixel image + metadata.

    Process:
    1. Load all tensors from .npz
    2. Quantize each tensor to uint8 (per-tensor min/max)
    3. Concatenate all bytes
    4. Pack bytes into RGB pixels (3 bytes → 1 pixel)
    5. Arrange into roughly square image
    6. Save as .pxi image + .meta.json

    The metadata tracks:
    - Per-tensor: name, shape, w_min, w_max, byte offset, length
    - Global: model name, version, total size
    """
    print("="*60)
    print("ENCODING MODEL → PIXELS")
    print("="*60)
    print(f"Input: {npz_path}")
    print()

    # Load weights
    data = np.load(npz_path)
    print(f"Loaded {len(data.files)} tensors")

    # Quantize each tensor and collect bytes
    tensors_meta = []
    all_bytes = []
    offset = 0

    for name in sorted(data.files):  # Sort for determinism
        W = data[name]
        print(f"  {name:12s}: {str(W.shape):20s} → ", end="")

        q, w_min, w_max = quantize_tensor(W)
        flat_bytes = q.flatten().tobytes()

        print(f"{len(flat_bytes):8d} bytes (range [{w_min:+.4f}, {w_max:+.4f}])")

        tensors_meta.append({
            "name": name,
            "shape": list(W.shape),
            "dtype": str(W.dtype),
            "w_min": w_min,
            "w_max": w_max,
            "offset": offset,
            "length": len(flat_bytes),
        })

        all_bytes.append(flat_bytes)
        offset += len(flat_bytes)

    # Concatenate all bytes
    blob = b"".join(all_bytes)
    print(f"\nTotal bytes: {len(blob):,}")

    # Pad to multiple of 3 (for RGB pixels)
    pad = (-len(blob)) % 3
    if pad:
        blob += b"\x00" * pad
        print(f"Padded: +{pad} bytes → {len(blob):,} bytes")

    # Convert to RGB pixels
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(-1, 3)
    num_pixels = arr.shape[0]

    # Make roughly square image
    width = int(np.ceil(np.sqrt(num_pixels)))
    height = int(np.ceil(num_pixels / width))

    print(f"\nImage dimensions: {width}x{height} ({width*height:,} pixels)")

    # Pad to fill rectangle
    padded = np.zeros((width * height, 3), dtype=np.uint8)
    padded[:num_pixels, :] = arr

    # Reshape to image
    img_array = padded.reshape(height, width, 3)

    # Save image (PNG format with .pxi extension)
    out_pxi.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(img_array, mode="RGB")
    img.save(out_pxi, format="PNG")

    # Create metadata
    meta = {
        "model_name": "pixellm_v0",
        "version": "0.0.1",
        "architecture": {
            "type": "mlp",
            "vocab_size": 1024,
            "model_dim": 128,
            "layers": ["embed", "hidden", "out"],
        },
        "image_path": str(out_pxi.name),
        "image_width": width,
        "image_height": height,
        "total_bytes": len(blob),
        "total_params": sum(t["length"] for t in tensors_meta),
        "quantization": "uint8_per_tensor",
        "tensors": tensors_meta,
    }

    meta_path.write_text(json.dumps(meta, indent=2))

    # Summary
    print()
    print("="*60)
    print("✅ ENCODING COMPLETE")
    print("="*60)
    print(f"Pixel image: {out_pxi}")
    print(f"  Size: {out_pxi.stat().st_size / 1024:.1f} KB")
    print(f"  Dimensions: {width}x{height}")
    print(f"Metadata: {meta_path}")
    print(f"  Size: {meta_path.stat().st_size / 1024:.1f} KB")
    print()
    print("Model is now pixel-native! 🎨")
    print()
    print("Next step:")
    print("  python3 pixel_llm/programs/pixellm_infer.py 'hello pixels'")
    print()


def main():
    """Encode pixellm_v0.npz to pixels."""
    npz_path = MODELS_DIR / "pixellm_v0.npz"
    out_pxi = MODELS_DIR / "pixellm_v0.pxi"
    meta_path = MODELS_DIR / "pixellm_v0.meta.json"

    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found")
        print("Run: python3 pixel_llm/models/pixellm_v0_train.py")
        return 1

    encode_model_to_pixels(npz_path, out_pxi, meta_path)
    return 0


if __name__ == "__main__":
    exit(main())
