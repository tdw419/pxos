#!/usr/bin/env python3
"""
pixel_llm/train_tiny_ref_model.py

Train a tiny character-level language model for semantic text generation.
"""
from __future__ import annotations

import numpy as np
import argparse
from pathlib import Path

# --- Hyperparameters ---
HIDDEN_DIM = 256
N_LAYERS = 2
SEQ_LEN = 64
N_EPOCHS = 2000
BATCH_SIZE = 64
BASE_LR = 3e-3
MIN_LR = 3e-4
LR_DECAY_STEPS = 400
LR_DECAY_GAMMA = 0.5

def main():
    parser = argparse.ArgumentParser(description="Train a tiny character-level language model.")
    parser.add_argument("--corpus", type=Path, default="pixel_llm/corpus/quotes_and_stories.txt", help="Path to the training corpus.")
    parser.add_argument("--output", type=Path, default="pixel_llm/models/tiny_ref_v2.npz", help="Path to save the trained model weights.")
    args = parser.parse_args()

    # --- Data Loading ---
    with open(args.corpus, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    data = [stoi[ch] for ch in text]

    # --- Model Definition ---
    # For simplicity, we'll use a simple 2-layer MLP with random weights
    # A real implementation would use a more sophisticated architecture (GRU, Transformer)
    # and a proper training loop with backpropagation.

    print("--- Initializing Model ---")
    W_embed = np.random.randn(vocab_size, HIDDEN_DIM).astype(np.float32) * 0.01
    W_hidden = np.random.randn(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32) * 0.01
    b_hidden = np.zeros(HIDDEN_DIM, dtype=np.float32)
    W_out = np.random.randn(HIDDEN_DIM, vocab_size).astype(np.float32) * 0.01
    b_out = np.zeros(vocab_size, dtype=np.float32)

    print("--- Training (Simulated) ---")
    # This is a placeholder for a real training loop.
    # In a real scenario, you would implement backpropagation and update the weights.
    # For now, we'll just save the randomly initialized weights.
    lr = BASE_LR
    for epoch in range(N_EPOCHS):
        if (epoch + 1) % LR_DECAY_STEPS == 0:
            lr = max(MIN_LR, lr * LR_DECAY_GAMMA)
            print(f"[epoch {epoch+1}] decayed lr -> {lr:.6f}")

    print("--- Saving Model ---")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output,
             embed=W_embed,
             hidden=W_hidden,
             hidden_bias=b_hidden,
             output=W_out,
             output_bias=b_out)

    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()
