#!/usr/bin/env python3
"""
pixel_llm/train_tiny_ref_model.py

Train a tiny character-level language model for semantic text generation.

This script trains a simple 2-layer neural network on a small text corpus,
producing weights that can be compiled into a pixel program for real text generation.

Architecture:
- Embedding layer: tokens → hidden states
- Hidden layer: h = relu(h @ W_hidden + b_hidden)
- Output layer: logits = h @ W_out + b_out

The trained weights will be saved in the same format as pixellm_v0.npz,
allowing them to be compiled by the existing assembler.

Usage:
    python3 pixel_llm/train_tiny_ref_model.py
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Tuple

# Training hyperparameters (v0.4.0 - Production Model)
HIDDEN_DIM = 256          # Increased from 128 for better capacity
BASE_LEARNING_RATE = 0.003  # Starting LR
MIN_LEARNING_RATE = 0.0003  # Minimum LR after decay
LR_DECAY_STEPS = 100      # Decay every N epochs
LR_DECAY_GAMMA = 0.7      # Multiply LR by this factor
EPOCHS = 500              # Longer training for better quality
BATCH_SIZE = 64           # Bigger batches
SEQUENCE_LENGTH = 40      # Longer context for better patterns

# Default corpus path
DEFAULT_CORPUS_PATH = Path(__file__).parent / "data" / "pxos_corpus.txt"

# Character vocabulary (same as generate_text.py)
VOCAB = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?'-\n")
VOCAB_SIZE = len(VOCAB)
CHAR_TO_ID = {c: i for i, c in enumerate(VOCAB)}
ID_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}


def tokenize(text: str) -> List[int]:
    """Convert text to token IDs."""
    tokens = []
    for char in text:
        if char in CHAR_TO_ID:
            tokens.append(CHAR_TO_ID[char])
        else:
            tokens.append(CHAR_TO_ID[' '])  # Unknown → space
    return tokens


def create_training_sequences(tokens: List[int], seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training sequences from token list.

    Args:
        tokens: List of token IDs
        seq_len: Sequence length for training

    Returns:
        (inputs, targets) where inputs[i] predicts targets[i]
    """
    inputs = []
    targets = []

    for i in range(len(tokens) - seq_len):
        inputs.append(tokens[i:i+seq_len])
        targets.append(tokens[i+1:i+seq_len+1])

    return np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """ReLU gradient."""
    return (x > 0).astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute cross-entropy loss.

    Args:
        logits: (batch_size, vocab_size)
        targets: (batch_size,) token IDs

    Returns:
        Average loss
    """
    batch_size = logits.shape[0]
    probs = softmax(logits)

    # Clip probabilities to avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)

    # Negative log likelihood
    log_probs = np.log(probs[np.arange(batch_size), targets])
    loss = -np.mean(log_probs)

    return loss


def forward_pass(
    token_ids: np.ndarray,
    W_embed: np.ndarray,
    W_hidden: np.ndarray,
    b_hidden: np.ndarray,
    W_out: np.ndarray,
    b_out: np.ndarray
) -> Tuple[np.ndarray, dict]:
    """
    Forward pass through the network.

    Args:
        token_ids: (batch_size,) token IDs
        W_embed: (vocab_size, hidden_dim)
        W_hidden: (hidden_dim, hidden_dim)
        b_hidden: (hidden_dim,)
        W_out: (hidden_dim, vocab_size)
        b_out: (vocab_size,)

    Returns:
        logits, cache (for backward pass)
    """
    # Embedding lookup
    h_in = W_embed[token_ids]  # (batch_size, hidden_dim)

    # Hidden layer
    h_pre = h_in @ W_hidden + b_hidden  # (batch_size, hidden_dim)
    h = relu(h_pre)

    # Output layer
    logits = h @ W_out + b_out  # (batch_size, vocab_size)

    cache = {
        'h_in': h_in,
        'h_pre': h_pre,
        'h': h,
        'token_ids': token_ids
    }

    return logits, cache


def backward_pass(
    grad_logits: np.ndarray,
    cache: dict,
    W_embed: np.ndarray,
    W_hidden: np.ndarray,
    W_out: np.ndarray
) -> dict:
    """
    Backward pass to compute gradients.

    Args:
        grad_logits: (batch_size, vocab_size) gradient from loss
        cache: Forward pass cache
        W_embed, W_hidden, W_out: Model weights

    Returns:
        Dictionary of gradients
    """
    h_in = cache['h_in']
    h_pre = cache['h_pre']
    h = cache['h']
    token_ids = cache['token_ids']

    batch_size = grad_logits.shape[0]

    # Output layer gradients
    grad_W_out = h.T @ grad_logits / batch_size
    grad_b_out = np.sum(grad_logits, axis=0) / batch_size

    # Hidden layer gradients
    grad_h = grad_logits @ W_out.T
    grad_h_pre = grad_h * relu_grad(h_pre)

    grad_W_hidden = h_in.T @ grad_h_pre / batch_size
    grad_b_hidden = np.sum(grad_h_pre, axis=0) / batch_size

    # Embedding gradients
    grad_h_in = grad_h_pre @ W_hidden.T
    grad_W_embed = np.zeros_like(W_embed)
    for i, token_id in enumerate(token_ids):
        grad_W_embed[token_id] += grad_h_in[i]
    grad_W_embed /= batch_size

    return {
        'W_embed': grad_W_embed,
        'W_hidden': grad_W_hidden,
        'b_hidden': grad_b_hidden,
        'W_out': grad_W_out,
        'b_out': grad_b_out
    }


def train_model(
    corpus: str,
    hidden_dim: int = HIDDEN_DIM,
    base_lr: float = BASE_LEARNING_RATE,
    min_lr: float = MIN_LEARNING_RATE,
    lr_decay_steps: int = LR_DECAY_STEPS,
    lr_decay_gamma: float = LR_DECAY_GAMMA,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQUENCE_LENGTH
):
    """
    Train a character-level language model (v0.4.0 - Production Model).

    Args:
        corpus: Training text
        hidden_dim: Hidden layer dimension
        base_lr: Base learning rate (will decay)
        min_lr: Minimum learning rate after decay
        lr_decay_steps: Decay LR every N epochs
        lr_decay_gamma: Multiply LR by this factor on decay
        epochs: Number of training epochs
        batch_size: Batch size
        seq_len: Sequence length for context
    """
    print("=" * 70)
    print(" TRAINING PRODUCTION MODEL (v0.4.0)")
    print("=" * 70)
    print()
    print(f"Hidden dim: {hidden_dim}")
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Base LR: {base_lr} → {min_lr}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print()

    # Tokenize corpus
    tokens = tokenize(corpus)
    print(f"Corpus length: {len(corpus)} characters")
    print(f"Token count: {len(tokens)}")
    print()

    # Create training sequences
    inputs, targets = create_training_sequences(tokens, seq_len)
    print(f"Training sequences: {len(inputs)}")
    print()

    # Initialize weights
    print("Initializing weights...")
    np.random.seed(42)

    W_embed = np.random.randn(VOCAB_SIZE, hidden_dim).astype(np.float32) * 0.1
    W_hidden = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.1
    b_hidden = np.zeros(hidden_dim, dtype=np.float32)
    W_out = np.random.randn(hidden_dim, VOCAB_SIZE).astype(np.float32) * 0.1
    b_out = np.zeros(VOCAB_SIZE, dtype=np.float32)

    print(f"  W_embed: {W_embed.shape}")
    print(f"  W_hidden: {W_hidden.shape}")
    print(f"  b_hidden: {b_hidden.shape}")
    print(f"  W_out: {W_out.shape}")
    print(f"  b_out: {b_out.shape}")
    print()

    # Training loop with LR decay
    print("Training...")
    print("-" * 70)

    learning_rate = base_lr

    for epoch in range(epochs):
        # Shuffle sequences
        perm = np.random.permutation(len(inputs))
        inputs_shuffled = inputs[perm]
        targets_shuffled = targets[perm]

        total_loss = 0.0
        num_batches = 0

        # Mini-batch training
        for i in range(0, len(inputs_shuffled), batch_size):
            batch_inputs = inputs_shuffled[i:i+batch_size]
            batch_targets = targets_shuffled[i:i+batch_size]

            # For each sequence, train on each position
            for pos in range(seq_len):
                # Get tokens at this position
                token_ids = batch_inputs[:, pos]
                target_ids = batch_targets[:, pos]

                # Forward pass
                logits, cache = forward_pass(
                    token_ids, W_embed, W_hidden, b_hidden, W_out, b_out
                )

                # Compute loss
                loss = cross_entropy_loss(logits, target_ids)
                total_loss += loss
                num_batches += 1

                # Backward pass
                probs = softmax(logits)
                grad_logits = probs.copy()
                grad_logits[np.arange(len(target_ids)), target_ids] -= 1

                grads = backward_pass(
                    grad_logits, cache, W_embed, W_hidden, W_out
                )

                # Update weights
                W_embed -= learning_rate * grads['W_embed']
                W_hidden -= learning_rate * grads['W_hidden']
                b_hidden -= learning_rate * grads['b_hidden']
                W_out -= learning_rate * grads['W_out']
                b_out -= learning_rate * grads['b_out']

        avg_loss = total_loss / num_batches

        # Learning rate decay
        if (epoch + 1) % lr_decay_steps == 0:
            old_lr = learning_rate
            learning_rate = max(min_lr, learning_rate * lr_decay_gamma)
            if old_lr != learning_rate:
                print(f"Epoch {epoch+1:3d}/{epochs} - Loss: {avg_loss:.4f} - LR: {learning_rate:.6f} (decayed)")

        # Progress logging
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} - Loss: {avg_loss:.4f} - LR: {learning_rate:.6f}")

    print("-" * 70)
    print()

    return {
        'embed': W_embed,
        'hidden': W_hidden,
        'b_hidden': b_hidden,
        'out': W_out,
        'b_out': b_out
    }


def main():
    """Train and save production model (v0.4.0)."""
    # Load corpus
    corpus_path = DEFAULT_CORPUS_PATH

    if not corpus_path.exists():
        print("=" * 70)
        print(" ERROR: Training corpus not found")
        print("=" * 70)
        print()
        print(f"Expected: {corpus_path}")
        print()
        print("To create the corpus, run:")
        print("  python3 pixel_llm/build_corpus.py")
        print()
        print("Or place .txt files in pixel_llm/data/raw/ and run build_corpus.py")
        print()
        return 1

    print(f"Loading corpus from: {corpus_path}")
    corpus = corpus_path.read_text(encoding='utf-8')
    print(f"  Corpus size: {len(corpus):,} characters")
    print()

    # Train model
    weights = train_model(corpus)

    # Save weights
    output_path = Path(__file__).parent / "models" / "tiny_ref_prod.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving weights to: {output_path}")
    np.savez(output_path, **weights)

    file_size = output_path.stat().st_size
    print(f"  File size: {file_size:,} bytes")
    print()

    print("=" * 70)
    print(" TRAINING COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Export to pixels: python3 pixel_llm/export_to_pixels.py --model models/tiny_ref_prod.npz")
    print("  2. Generate text: python3 -m pxvm.examples.generate_text --program pixel_llm/programs/tiny_ref_prod.pxi")
    print()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
