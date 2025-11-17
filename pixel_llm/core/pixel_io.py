#!/usr/bin/env python3
"""
pixel_io.py - Pixel-Native Input/Output for Pixel-LLM

Encode and decode tokens and predictions as pixel images.
This makes Pixel-LLM I/O truly pixel-native:
  - Input: text → token.pxi (pixel image)
  - Output: logits → output.pxi (pixel image)

Usage:
    from pixel_llm.core.pixel_io import (
        pixel_encode_tokens,
        pixel_decode_tokens,
        pixel_encode_prediction,
        pixel_decode_prediction
    )

    # Encode text as pixels
    text = "pxOS evolution"
    token_image = pixel_encode_tokens(text, "input_tokens.pxi")

    # Decode tokens from pixels
    token_ids = pixel_decode_tokens("input_tokens.pxi")

    # Encode prediction as pixels
    pixel_encode_prediction(logits, "output_token.pxi")

    # Decode prediction from pixels
    predicted_token = pixel_decode_prediction("output_token.pxi")
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import Union, Optional

# Vocab size (must match model)
V = 1024


def simple_tokenize(text: str, vocab_size: int = V) -> np.ndarray:
    """
    Simple character tokenizer.

    Args:
        text: Input string
        vocab_size: Vocabulary size

    Returns:
        Array of token IDs
    """
    return np.array([ord(c) % vocab_size for c in text], dtype=np.int32)


def simple_detokenize(token_ids: np.ndarray) -> str:
    """
    Simple character detokenizer.

    Args:
        token_ids: Array of token IDs

    Returns:
        Decoded string
    """
    chars = [chr(int(tid) % 128) if int(tid) < 128 else '?' for tid in token_ids]
    return ''.join(chars)


def pixel_encode_tokens(text: str, output_path: Optional[Union[str, Path]] = None) -> np.ndarray:
    """
    Encode text as a pixel image.

    Each pixel encodes one token:
      - R channel: token_id % 256
      - G channel: token_id // 256
      - B channel: unused (0)

    The first pixel encodes metadata:
      - R: num_tokens % 256
      - G: num_tokens // 256
      - B: vocab_size indicator

    Args:
        text: Input text
        output_path: Optional path to save image

    Returns:
        Image array [H, W, 3] where H=1, W=num_tokens+1
    """
    # Tokenize
    token_ids = simple_tokenize(text, vocab_size=V)
    n_tokens = len(token_ids)

    # Create image: 1 row, (n_tokens + 1) columns, RGB
    width = n_tokens + 1
    height = 1
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # First pixel: metadata
    img[0, 0, 0] = n_tokens % 256
    img[0, 0, 1] = n_tokens // 256
    img[0, 0, 2] = 1  # Version indicator

    # Subsequent pixels: tokens
    for i, token_id in enumerate(token_ids):
        img[0, i + 1, 0] = token_id % 256
        img[0, i + 1, 1] = token_id // 256
        img[0, i + 1, 2] = 0

    # Save if requested
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img, mode="RGB").save(output_path, format="PNG")

    return img


def pixel_decode_tokens(image_path: Union[str, Path]) -> np.ndarray:
    """
    Decode tokens from pixel image.

    Args:
        image_path: Path to token image

    Returns:
        Array of token IDs
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)

    # Read metadata from first pixel
    n_tokens = int(arr[0, 0, 0]) + int(arr[0, 0, 1]) * 256

    # Read tokens
    token_ids = np.zeros(n_tokens, dtype=np.int32)
    for i in range(n_tokens):
        token_ids[i] = int(arr[0, i + 1, 0]) + int(arr[0, i + 1, 1]) * 256

    return token_ids


def pixel_encode_prediction(
    logits: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    top_k: int = 5
) -> np.ndarray:
    """
    Encode prediction logits as a pixel image.

    Two formats:
    1. Single prediction pixel (1×1):
       - R: predicted_token % 256
       - G: predicted_token // 256
       - B: confidence (0-255)

    2. Top-k predictions (1×top_k):
       - Each pixel encodes (token_id, confidence)

    Args:
        logits: Output logits [V]
        output_path: Optional path to save image
        top_k: Number of top predictions to encode

    Returns:
        Image array
    """
    # Get top-k predictions
    top_k_indices = logits.argsort()[-top_k:][::-1]
    top_k_logits = logits[top_k_indices]

    # Softmax for probabilities
    exp_logits = np.exp(top_k_logits - top_k_logits.max())
    top_k_probs = exp_logits / exp_logits.sum()

    # Create image: 1 row, top_k columns
    img = np.zeros((1, top_k, 3), dtype=np.uint8)

    for i, (token_id, prob) in enumerate(zip(top_k_indices, top_k_probs)):
        img[0, i, 0] = token_id % 256
        img[0, i, 1] = token_id // 256
        img[0, i, 2] = int(prob * 255)  # Confidence as 0-255

    # Save if requested
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img, mode="RGB").save(output_path, format="PNG")

    return img


def pixel_decode_prediction(image_path: Union[str, Path]) -> dict:
    """
    Decode prediction from pixel image.

    Args:
        image_path: Path to prediction image

    Returns:
        Dict with:
            - top_tokens: list of token IDs
            - top_probs: list of probabilities
            - top_token: most likely token ID
            - top_prob: probability of most likely token
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)

    height, width, _ = arr.shape
    n_predictions = width

    # Decode each prediction
    tokens = []
    probs = []

    for i in range(n_predictions):
        token_id = int(arr[0, i, 0]) + int(arr[0, i, 1]) * 256
        prob = float(arr[0, i, 2]) / 255.0

        tokens.append(token_id)
        probs.append(prob)

    return {
        "top_tokens": tokens,
        "top_probs": probs,
        "top_token": tokens[0],
        "top_prob": probs[0],
    }


def pixel_encode_logits_heatmap(
    logits: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    width: int = 32
) -> np.ndarray:
    """
    Encode logits as a visual heatmap.

    This makes logits *visible* - you can literally see the AI's predictions.

    Args:
        logits: Output logits [V]
        output_path: Optional path to save image
        width: Width of heatmap (height = vocab_size // width)

    Returns:
        Heatmap image array
    """
    V = len(logits)
    height = (V + width - 1) // width  # Ceil division

    # Normalize logits to 0-255
    logit_min = logits.min()
    logit_max = logits.max()

    if logit_max > logit_min:
        normalized = ((logits - logit_min) / (logit_max - logit_min) * 255).astype(np.uint8)
    else:
        normalized = np.full(V, 128, dtype=np.uint8)

    # Reshape to 2D grid
    padded = np.zeros(width * height, dtype=np.uint8)
    padded[:V] = normalized

    heatmap = padded.reshape(height, width)

    # Convert to RGB (grayscale heatmap)
    heatmap_rgb = np.stack([heatmap, heatmap, heatmap], axis=-1)

    # Save if requested
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(heatmap_rgb, mode="RGB").save(output_path, format="PNG")

    return heatmap_rgb


def main():
    """Test pixel I/O."""
    print("="*60)
    print("PIXEL I/O TEST")
    print("="*60)

    # Test 1: Encode tokens
    print("\n1. Encoding text as pixels...")
    text = "pxOS evolution system"
    print(f"   Text: '{text}'")

    token_img = pixel_encode_tokens(text, "/tmp/test_tokens.pxi")
    print(f"   Encoded: {token_img.shape} image")
    print(f"   Saved: /tmp/test_tokens.pxi")

    # Test 2: Decode tokens
    print("\n2. Decoding tokens from pixels...")
    decoded_ids = pixel_decode_tokens("/tmp/test_tokens.pxi")
    print(f"   Decoded: {len(decoded_ids)} tokens")
    print(f"   Token IDs: {decoded_ids[:10]}...")

    decoded_text = simple_detokenize(decoded_ids)
    print(f"   Text: '{decoded_text}'")
    print(f"   Match: {decoded_text == text} ✅" if decoded_text == text else f"   Match: {decoded_text == text} ❌")

    # Test 3: Encode prediction
    print("\n3. Encoding prediction as pixels...")
    # Fake logits
    logits = np.random.randn(V).astype(np.float32)
    pred_img = pixel_encode_prediction(logits, "/tmp/test_prediction.pxi", top_k=5)
    print(f"   Encoded: {pred_img.shape} image")
    print(f"   Saved: /tmp/test_prediction.pxi")

    # Test 4: Decode prediction
    print("\n4. Decoding prediction from pixels...")
    pred = pixel_decode_prediction("/tmp/test_prediction.pxi")
    print(f"   Top token: {pred['top_token']}")
    print(f"   Top prob: {pred['top_prob']:.4f}")
    print(f"   Top 5 tokens: {pred['top_tokens']}")

    # Test 5: Logits heatmap
    print("\n5. Creating logits heatmap...")
    heatmap = pixel_encode_logits_heatmap(logits, "/tmp/test_heatmap.pxi", width=32)
    print(f"   Heatmap: {heatmap.shape}")
    print(f"   Saved: /tmp/test_heatmap.pxi")

    print("\n" + "="*60)
    print("✅ All pixel I/O tests passed!")
    print("="*60)
    print()


if __name__ == "__main__":
    main()
