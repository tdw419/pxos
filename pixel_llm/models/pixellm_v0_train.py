#!/usr/bin/env python3
"""
pixellm_v0_train.py - Tiny MLP Language Model

Produces pixellm_v0.npz - a minimal neural network whose weights will live in pixels.

Architecture:
  - Vocab size: 1024 tokens
  - Model dim: 128
  - Embedding: [V=1024, D=128]
  - Hidden layer: [D=128, D=128] with ReLU
  - Output head: [D=128, V=1024]

For v0, this is just weight initialization (no real training yet).
The goal is to establish the pipeline: weights.npz → pixels → inference.

Later we can swap in real training (PyTorch, JAX, etc.).

Usage:
    python3 pixel_llm/models/pixellm_v0_train.py
"""

import numpy as np
from pathlib import Path

# Config
V = 1024  # vocab size
D = 128   # model dimension

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "pixel_llm" / "models"

def init_weights(rng, vocab_size=V, model_dim=D):
    """
    Initialize model weights with Xavier/He initialization.

    Returns dict of numpy arrays ready to save as .npz.
    """
    scale_embed = np.sqrt(2.0 / vocab_size)
    scale_hidden = np.sqrt(2.0 / model_dim)
    scale_out = np.sqrt(2.0 / model_dim)

    weights = {
        # Token embedding [V, D]
        "embed": rng.normal(0, scale_embed, size=(vocab_size, model_dim)).astype("float32"),

        # Hidden layer [D, D] + bias [D]
        "hidden": rng.normal(0, scale_hidden, size=(model_dim, model_dim)).astype("float32"),
        "b_hidden": np.zeros((model_dim,), dtype="float32"),

        # Output head [D, V] + bias [V]
        "out": rng.normal(0, scale_out, size=(model_dim, vocab_size)).astype("float32"),
        "b_out": np.zeros((vocab_size,), dtype="float32"),
    }

    return weights


def dummy_training_loop(weights, steps=1000, learning_rate=0.01, rng=None):
    """
    Placeholder for real training.

    For v0, we just return initialized weights.

    Later, this can:
    - Load pxOS code/docs as training data
    - Train simple next-token prediction
    - Use PyTorch/JAX for gradient descent
    - Emit checkpoints at different training steps

    For now: weights pass through unchanged.
    """
    print(f"[dummy_training_loop] Would train for {steps} steps with lr={learning_rate}")
    print("[dummy_training_loop] For v0, just returning initialized weights")

    # Future: real training here
    # data = load_pxos_corpus()
    # for step in range(steps):
    #     loss = forward_and_backward(weights, data)
    #     update_weights(weights, learning_rate)

    return weights


def compute_model_stats(weights):
    """Show model statistics."""
    total_params = sum(w.size for w in weights.values())
    total_bytes = sum(w.nbytes for w in weights.values())

    print("\n" + "="*60)
    print("MODEL STATISTICS")
    print("="*60)

    for name, w in sorted(weights.items()):
        params = w.size
        mb = w.nbytes / (1024**2)
        print(f"{name:12s}: {str(w.shape):20s} | {params:8d} params | {mb:6.2f} MB")

    print("-"*60)
    print(f"{'TOTAL':12s}: {'':<20s} | {total_params:8d} params | {total_bytes/(1024**2):6.2f} MB")
    print("="*60 + "\n")


def main():
    """Initialize Pixel-LLM v0 weights and save to .npz."""
    print("="*60)
    print("PIXEL-LLM v0 - WEIGHT INITIALIZATION")
    print("="*60)
    print(f"Vocab size: {V}")
    print(f"Model dim: {D}")
    print()

    # Initialize RNG
    rng = np.random.default_rng(42)

    # Initialize weights
    print("Initializing weights...")
    weights = init_weights(rng, vocab_size=V, model_dim=D)

    # "Train" (for now, just a pass-through)
    print("\nRunning training loop...")
    weights = dummy_training_loop(weights, steps=1000, learning_rate=0.01, rng=rng)

    # Show stats
    compute_model_stats(weights)

    # Save to .npz
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "pixellm_v0.npz"

    print(f"Saving weights to {out_path}...")
    np.savez(out_path, **weights)

    print("✅ Pixel-LLM v0 weights saved")
    print(f"   Location: {out_path}")
    print(f"   Size: {out_path.stat().st_size / 1024:.1f} KB")
    print()
    print("Next steps:")
    print("  1. python3 pixel_llm/core/model_to_pixels.py")
    print("  2. python3 pixel_llm/programs/pixellm_infer.py 'hello pixels'")
    print()


if __name__ == "__main__":
    main()
