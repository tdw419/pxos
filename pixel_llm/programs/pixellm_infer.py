#!/usr/bin/env python3
"""
pixellm_infer.py - Pixel-LLM v0 Inference

Run a forward pass through Pixel-LLM using weights loaded from pixels.

Architecture:
    tokens → embedding → hidden (ReLU) → output → logits

The weights are loaded from pixellm_v0.pxi (pixel image), making this
the first neural network inference running on pixel-native weights.

Usage:
    python3 pixel_llm/programs/pixellm_infer.py "hello pixels"
    python3 pixel_llm/programs/pixellm_infer.py "pxOS is"
"""

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pixel_llm.core.pixel_model_loader import get_default_loader

# Model config (matches training)
V = 1024  # vocab size
D = 128   # model dimension


def simple_tokenize(text: str, vocab_size: int = V) -> np.ndarray:
    """
    Simple character-based tokenizer.

    For v0, we use a trivial tokenizer:
    - Map each character to token ID (ord(c) % vocab_size)
    - Real version would use BPE/SentencePiece

    Args:
        text: Input string
        vocab_size: Size of vocabulary

    Returns:
        Array of token IDs
    """
    ids = [ord(c) % vocab_size for c in text]
    return np.array(ids, dtype=np.int64)


def simple_detokenize(token_ids: np.ndarray) -> str:
    """
    Simple character-based detokenizer.

    Args:
        token_ids: Array of token IDs

    Returns:
        Decoded string
    """
    # For simple char-based tokenizer, just convert back to chars
    chars = [chr(int(tid) % 128) if int(tid) < 128 else '?' for tid in token_ids]
    return ''.join(chars)


def pixellm_forward(tokens: np.ndarray, loader, verbose: bool = False) -> np.ndarray:
    """
    Run forward pass through Pixel-LLM v0.

    Architecture:
        1. Embedding: tokens → [T, D]
        2. Pooling: [T, D] → [D] (mean over sequence)
        3. Hidden: [D] @ [D, D] + bias → [D], then ReLU
        4. Output: [D] @ [D, V] + bias → [V] logits

    Args:
        tokens: Array of token IDs [T]
        loader: PixelModelLoader instance
        verbose: Print intermediate shapes

    Returns:
        Logits [V] for next token prediction
    """
    if verbose:
        print("\n" + "="*60)
        print("PIXEL-LLM v0 FORWARD PASS")
        print("="*60)

    # Load weights from pixels
    if verbose:
        print("\n1. Loading weights from pixels...")

    W_embed = loader.load_tensor("embed")      # [V, D]
    W_hidden = loader.load_tensor("hidden")    # [D, D]
    W_out = loader.load_tensor("out")          # [D, V]
    b_hidden = loader.load_tensor("b_hidden")  # [D]
    b_out = loader.load_tensor("b_out")        # [V]

    if verbose:
        print(f"   embed: {W_embed.shape}")
        print(f"   hidden: {W_hidden.shape}")
        print(f"   out: {W_out.shape}")

    # Embedding lookup
    if verbose:
        print(f"\n2. Embedding lookup: tokens {tokens.shape} → embeddings")

    x = W_embed[tokens]  # [T, D]

    if verbose:
        print(f"   embeddings: {x.shape}")

    # Sequence pooling (mean)
    if verbose:
        print(f"\n3. Pooling: mean over sequence")

    x = x.mean(axis=0)  # [D]

    if verbose:
        print(f"   pooled: {x.shape}")

    # Hidden layer with ReLU
    if verbose:
        print(f"\n4. Hidden layer: [D] @ [D,D] + bias → ReLU")

    h = x @ W_hidden + b_hidden  # [D]
    h = np.maximum(0, h)         # ReLU

    if verbose:
        print(f"   hidden activations: {h.shape}")
        print(f"   non-zero activations: {(h > 0).sum()}/{len(h)}")

    # Output layer
    if verbose:
        print(f"\n5. Output layer: [D] @ [D,V] + bias")

    logits = h @ W_out + b_out  # [V]

    if verbose:
        print(f"   logits: {logits.shape}")
        print("="*60 + "\n")

    return logits


def print_top_k_tokens(logits: np.ndarray, k: int = 10):
    """
    Print top-k predicted tokens.

    Args:
        logits: Array of logits [V]
        k: Number of top predictions to show
    """
    # Get top-k indices
    topk_indices = logits.argsort()[-k:][::-1]
    topk_logits = logits[topk_indices]

    # Softmax for probabilities
    exp_logits = np.exp(topk_logits - topk_logits.max())
    probs = exp_logits / exp_logits.sum()

    print(f"\nTop {k} predicted tokens:")
    print("-" * 60)
    print(f"{'Rank':<6} {'Token ID':<10} {'Logit':<12} {'Probability':<12} {'Char':<6}")
    print("-" * 60)

    for rank, (idx, logit, prob) in enumerate(zip(topk_indices, topk_logits, probs), 1):
        char = chr(int(idx) % 128) if int(idx) < 128 else '?'
        print(f"{rank:<6} {int(idx):<10} {logit:<12.4f} {prob:<12.4f} '{char}'")

    print("-" * 60)


def main():
    """Run Pixel-LLM inference."""
    if len(sys.argv) < 2:
        print("Usage: pixellm_infer.py 'some text'")
        print("\nExamples:")
        print("  python3 pixel_llm/programs/pixellm_infer.py 'hello pixels'")
        print("  python3 pixel_llm/programs/pixellm_infer.py 'pxOS is'")
        sys.exit(1)

    text = sys.argv[1]
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print("="*60)
    print("PIXEL-LLM v0 - INFERENCE FROM PIXELS")
    print("="*60)
    print(f"\nInput text: '{text}'")

    # Load model
    print("\nLoading Pixel-LLM v0 from pixels...")
    try:
        loader = get_default_loader()
        info = loader.get_model_info()
        print(f"  Model: {info['model_name']} v{info['version']}")
        print(f"  Parameters: {info['total_params']:,}")
        print(f"  Tensors: {info['tensors']}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}\n")
        return 1

    # Tokenize
    tokens = simple_tokenize(text)
    print(f"\nTokenized: {len(tokens)} tokens")
    print(f"  Token IDs: {tokens.tolist()}")

    # Forward pass
    print("\nRunning forward pass...")
    logits = pixellm_forward(tokens, loader, verbose=verbose)

    # Results
    print(f"\n✅ Inference complete!")
    print(f"   Output: {len(logits)} logits")
    print(f"   Range: [{logits.min():.4f}, {logits.max():.4f}]")

    # Top-k predictions
    print_top_k_tokens(logits, k=10)

    # Sample next token (argmax)
    next_token_id = int(logits.argmax())
    next_char = chr(next_token_id % 128) if next_token_id < 128 else '?'

    print(f"\nGreedy prediction (argmax):")
    print(f"  Token ID: {next_token_id}")
    print(f"  Character: '{next_char}'")

    print("\n" + "="*60)
    print("🎨 Inference powered by weights living in pixels! 🧠")
    print("="*60 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
