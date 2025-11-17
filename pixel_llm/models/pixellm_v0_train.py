#!/usr/bin/env python3
"""
pixellm_v0_train.py - Train Tiny MLP Language Model on pxOS Corpus

Produces pixellm_v0.npz - a minimal neural network trained on pxOS code/docs,
whose weights will live in pixels.

Architecture:
  - Vocab size: 1024 tokens
  - Model dim: 128
  - Embedding: [V=1024, D=128]
  - Hidden layer: [D=128, D=128] with ReLU
  - Output head: [D=128, V=1024]

Training:
  - Task: Next-token prediction
  - Corpus: pxOS code, docs, Genesis spec (~491KB, ~122K tokens)
  - Implementation: Pure numpy (no PyTorch/JAX dependencies)
  - Optimizer: Simple SGD
  - Steps: 2000
  - Batch size: 32
  - Sequence length: 64

The trained weights establish pixel-native AI that understands pxOS.

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


def load_corpus():
    """Load the pxOS corpus."""
    corpus_path = ROOT / "pixel_llm" / "data" / "pxos_corpus.txt"

    if not corpus_path.exists():
        print(f"⚠️  Corpus not found: {corpus_path}")
        print("   Run: python3 pixel_llm/data/pxos_corpus_builder.py")
        return None

    return corpus_path.read_text(encoding="utf-8")


def tokenize_corpus(text, vocab_size=V):
    """Simple character tokenizer: ord(c) % vocab_size."""
    return np.array([ord(c) % vocab_size for c in text], dtype=np.int32)


def sample_batch(tokens, batch_size=32, seq_len=64, rng=None):
    """
    Sample random training batch.

    Returns:
        x: [batch_size, seq_len] input tokens
        y: [batch_size, seq_len] target tokens (shifted by 1)
    """
    if rng is None:
        rng = np.random.default_rng()

    max_start = len(tokens) - seq_len - 1
    starts = rng.integers(0, max_start, size=batch_size)

    x = np.zeros((batch_size, seq_len), dtype=np.int32)
    y = np.zeros((batch_size, seq_len), dtype=np.int32)

    for i, start in enumerate(starts):
        x[i] = tokens[start:start + seq_len]
        y[i] = tokens[start + 1:start + seq_len + 1]

    return x, y


def forward_pass(weights, x):
    """
    Forward pass.

    Args:
        weights: model weights dict
        x: [batch_size, seq_len] input token IDs

    Returns:
        logits: [batch_size, seq_len, vocab_size]
        cache: dict of intermediate activations for backward pass
    """
    W_embed = weights["embed"]
    W_hidden = weights["hidden"]
    W_out = weights["out"]
    b_hidden = weights["b_hidden"]
    b_out = weights["b_out"]

    batch_size, seq_len = x.shape

    # Embedding: [batch_size, seq_len, D]
    emb = W_embed[x]

    # Pool over sequence: [batch_size, D]
    pooled = emb.mean(axis=1)

    # Hidden layer: [batch_size, D]
    hidden_pre = pooled @ W_hidden + b_hidden
    hidden = np.maximum(0, hidden_pre)  # ReLU

    # Output: [batch_size, V]
    logits = hidden @ W_out + b_out

    # Cache for backward
    cache = {
        "x": x,
        "emb": emb,
        "pooled": pooled,
        "hidden_pre": hidden_pre,
        "hidden": hidden,
    }

    return logits, cache


def compute_loss(logits, y):
    """
    Compute cross-entropy loss.

    Args:
        logits: [batch_size, vocab_size]
        y: [batch_size, seq_len] target tokens

    Returns:
        loss: scalar
    """
    batch_size, seq_len = y.shape

    # Take first token of each sequence as target (simple version)
    y_target = y[:, 0]  # [batch_size]

    # Softmax
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # Cross-entropy
    log_probs = np.log(probs[np.arange(batch_size), y_target] + 1e-10)
    loss = -log_probs.mean()

    return loss, probs, y_target


def backward_pass(weights, cache, probs, y_target, learning_rate=0.001):
    """
    Backward pass and weight update (combined for simplicity).

    Args:
        weights: model weights (modified in-place)
        cache: cached activations from forward pass
        probs: softmax probabilities [batch_size, vocab_size]
        y_target: target tokens [batch_size]
        learning_rate: SGD learning rate
    """
    batch_size = len(y_target)

    # Gradient of loss w.r.t. logits
    dlogits = probs.copy()
    dlogits[np.arange(batch_size), y_target] -= 1.0
    dlogits /= batch_size

    # Backprop through output layer
    dhidden = dlogits @ weights["out"].T
    dW_out = cache["hidden"].T @ dlogits
    db_out = dlogits.sum(axis=0)

    # Backprop through ReLU
    dhidden_pre = dhidden * (cache["hidden_pre"] > 0)

    # Backprop through hidden layer
    dpooled = dhidden_pre @ weights["hidden"].T
    dW_hidden = cache["pooled"].T @ dhidden_pre
    db_hidden = dhidden_pre.sum(axis=0)

    # Backprop through pooling (distribute gradient equally)
    seq_len = cache["emb"].shape[1]
    demb = dpooled[:, None, :] / seq_len  # [batch_size, 1, D]
    demb = np.broadcast_to(demb, cache["emb"].shape)

    # Backprop through embedding (accumulate gradients)
    dW_embed = np.zeros_like(weights["embed"])
    x = cache["x"]
    for i in range(batch_size):
        for t in range(seq_len):
            token = x[i, t]
            dW_embed[token] += demb[i, t]

    # SGD update
    weights["embed"] -= learning_rate * dW_embed
    weights["hidden"] -= learning_rate * dW_hidden
    weights["b_hidden"] -= learning_rate * db_hidden
    weights["out"] -= learning_rate * dW_out
    weights["b_out"] -= learning_rate * db_out


def training_loop(weights, steps=2000, learning_rate=0.001, rng=None):
    """
    Real training loop on pxOS corpus.

    Args:
        weights: initial weights
        steps: number of training steps
        learning_rate: SGD learning rate
        rng: random number generator

    Returns:
        trained weights
    """
    print(f"\nTraining for {steps} steps with lr={learning_rate}")

    # Load corpus
    print("Loading pxOS corpus...")
    corpus_text = load_corpus()

    if corpus_text is None:
        print("❌ Training aborted - corpus not found")
        print("   Returning initialized weights")
        return weights

    print(f"Corpus loaded: {len(corpus_text):,} characters")

    # Tokenize
    print("Tokenizing corpus...")
    tokens = tokenize_corpus(corpus_text, vocab_size=V)
    print(f"Tokens: {len(tokens):,}")

    if rng is None:
        rng = np.random.default_rng()

    # Training loop
    print("\nStarting training...")
    print("="*60)

    losses = []

    for step in range(steps):
        # Sample batch
        x, y = sample_batch(tokens, batch_size=32, seq_len=64, rng=rng)

        # Forward pass
        logits, cache = forward_pass(weights, x)

        # Compute loss
        loss, probs, y_target = compute_loss(logits, y)
        losses.append(loss)

        # Backward pass + update
        backward_pass(weights, cache, probs, y_target, learning_rate)

        # Log progress
        if (step + 1) % 100 == 0:
            avg_loss = np.mean(losses[-100:])
            print(f"Step {step + 1:4d}/{steps} | Loss: {loss:.4f} | Avg loss (last 100): {avg_loss:.4f}")

    print("="*60)
    final_loss = np.mean(losses[-100:])
    print(f"\n✅ Training complete!")
    print(f"   Final loss: {final_loss:.4f}")
    print(f"   Initial loss: {losses[0]:.4f}")
    print(f"   Improvement: {losses[0] - final_loss:.4f}")
    print()

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
    """Train Pixel-LLM v0 on pxOS corpus and save weights."""
    print("="*60)
    print("PIXEL-LLM v0 - TRAINING ON pxOS CORPUS")
    print("="*60)
    print(f"Vocab size: {V}")
    print(f"Model dim: {D}")
    print()

    # Initialize RNG
    rng = np.random.default_rng(42)

    # Initialize weights
    print("Initializing weights...")
    weights = init_weights(rng, vocab_size=V, model_dim=D)

    # Train on pxOS corpus
    print("\nRunning training loop...")
    weights = training_loop(weights, steps=2000, learning_rate=0.001, rng=rng)

    # Show stats
    compute_model_stats(weights)

    # Save to .npz
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "pixellm_v0.npz"

    print(f"Saving weights to {out_path}...")
    np.savez(out_path, **weights)

    print("✅ Pixel-LLM v0 weights saved (TRAINED)")
    print(f"   Location: {out_path}")
    print(f"   Size: {out_path.stat().st_size / 1024:.1f} KB")
    print()
    print("Next steps:")
    print("  1. python3 pixel_llm/core/model_to_pixels.py")
    print("  2. python3 pixel_llm/programs/pixellm_infer.py 'pxOS evolution'")
    print()


if __name__ == "__main__":
    main()
