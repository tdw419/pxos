#!/usr/bin/env python3
"""
pixel_model_loader.py - Load Neural Network Weights from Pixels

Reads pixellm_v0.pxi (pixel image) + metadata → reconstruct numpy tensors

This is the reverse of model_to_pixels.py:
1. Load pixel image as RGB array
2. Flatten to byte stream
3. Extract each tensor using metadata (offset, length)
4. Dequantize from uint8 → float32 using (w_min, w_max)

The weights truly live as pixels. This loader is their decoder.

Usage:
    from pixel_llm.core.pixel_model_loader import PixelModelLoader

    loader = PixelModelLoader(image_path, meta_path)
    W_embed = loader.load_tensor("embed")
    W_hidden = loader.load_tensor("hidden")
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "pixel_llm" / "models"


class PixelModelLoader:
    """
    Load quantized model weights from pixel images.

    The model's weights are stored as RGB pixel values. This loader:
    - Reads the pixel image
    - Converts pixels back to bytes
    - Dequantizes using per-tensor metadata
    - Returns numpy arrays ready for inference
    """

    def __init__(self, image_path: Path, meta_path: Path):
        """
        Initialize loader.

        Args:
            image_path: Path to .pxi file (PNG with model weights as pixels)
            meta_path: Path to .meta.json (shapes, scales, offsets)
        """
        self.image_path = image_path
        self.meta_path = meta_path

        # Load metadata
        self.meta = json.loads(meta_path.read_text())

        # Load pixel image
        img = Image.open(self.image_path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)  # [H, W, 3]

        # Flatten to byte stream
        self.raw_bytes = arr.reshape(-1, 3).tobytes()

        # Build tensor index
        self.tensor_index = {t["name"]: t for t in self.meta["tensors"]}

        # Cache for loaded tensors
        self._cache: Dict[str, np.ndarray] = {}

    def load_tensor(self, name: str, use_cache: bool = True) -> np.ndarray:
        """
        Load and dequantize a tensor by name.

        Args:
            name: Tensor name (e.g. "embed", "hidden", "out")
            use_cache: If True, cache loaded tensors

        Returns:
            numpy array (float32) with original shape

        Raises:
            KeyError: if tensor name not found
        """
        # Check cache
        if use_cache and name in self._cache:
            return self._cache[name]

        # Get metadata
        if name not in self.tensor_index:
            available = ", ".join(self.tensor_index.keys())
            raise KeyError(f"Tensor '{name}' not found. Available: {available}")

        t = self.tensor_index[name]

        # Extract bytes
        start = t["offset"]
        end = start + t["length"]
        buf = self.raw_bytes[start:end]

        # Convert to uint8 array
        q = np.frombuffer(buf, dtype=np.uint8)

        # Dequantize: uint8 [0,255] → float32 [w_min, w_max]
        w_min = t["w_min"]
        w_max = t["w_max"]

        if w_max == w_min:
            # Constant tensor (e.g. zero-initialized bias)
            f = np.full(q.shape, w_min, dtype="float32")
        else:
            # Scale from [0, 255] → [w_min, w_max]
            f = q.astype("float32") / 255.0 * (w_max - w_min) + w_min

        # Reshape to original shape
        result = f.reshape(t["shape"])

        # Cache if requested
        if use_cache:
            self._cache[name] = result

        return result

    def load_all_tensors(self) -> Dict[str, np.ndarray]:
        """Load all tensors and return as dict."""
        return {name: self.load_tensor(name) for name in self.tensor_index.keys()}

    def get_tensor_names(self) -> list[str]:
        """Get list of available tensor names."""
        return list(self.tensor_index.keys())

    def get_model_info(self) -> Dict:
        """Get model metadata (architecture, version, etc.)."""
        return {
            "model_name": self.meta.get("model_name", "unknown"),
            "version": self.meta.get("version", "unknown"),
            "architecture": self.meta.get("architecture", {}),
            "total_params": self.meta.get("total_params", 0),
            "total_bytes": self.meta.get("total_bytes", 0),
            "quantization": self.meta.get("quantization", "unknown"),
            "tensors": len(self.meta["tensors"]),
        }

    def __repr__(self):
        info = self.get_model_info()
        return (
            f"PixelModelLoader("
            f"model={info['model_name']}, "
            f"version={info['version']}, "
            f"tensors={info['tensors']}, "
            f"params={info['total_params']:,})"
        )


def get_default_loader() -> PixelModelLoader:
    """
    Get loader for the default pixellm_v0 model.

    Returns:
        PixelModelLoader instance

    Raises:
        FileNotFoundError: if model files don't exist
    """
    image_path = MODELS_DIR / "pixellm_v0.pxi"
    meta_path = MODELS_DIR / "pixellm_v0.meta.json"

    if not image_path.exists():
        raise FileNotFoundError(
            f"Model image not found: {image_path}\n"
            "Run: python3 pixel_llm/core/model_to_pixels.py"
        )

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {meta_path}\n"
            "Run: python3 pixel_llm/core/model_to_pixels.py"
        )

    return PixelModelLoader(image_path, meta_path)


def main():
    """Test loading weights from pixels."""
    print("="*60)
    print("PIXEL MODEL LOADER TEST")
    print("="*60)

    try:
        loader = get_default_loader()
        print(f"\n{loader}\n")

        info = loader.get_model_info()
        print("Model Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        print(f"\nAvailable tensors: {', '.join(loader.get_tensor_names())}")

        # Test loading each tensor
        print("\nLoading tensors from pixels...")
        for name in loader.get_tensor_names():
            tensor = loader.load_tensor(name)
            print(f"  {name:12s}: {str(tensor.shape):20s} "
                  f"| range [{tensor.min():+.4f}, {tensor.max():+.4f}]")

        print("\n✅ All tensors loaded successfully from pixels!")
        print()

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}\n")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
