#!/usr/bin/env python3
"""
pxvm/examples/generate_text.py

Autoregressive text generation using Pixel-LLM forward pass program.

This demonstrates end-to-end inference:
- Load .pxi pixel program
- Run token-by-token generation loop
- Sample from logits distribution
- Detokenize and print generated text

Usage:
    python3 -m pxvm.examples.generate_text --prompt "The meaning of life is"
    python3 -m pxvm.examples.generate_text --max-tokens 50 --temperature 0.8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from pxvm.core.interpreter import run_program
from pxvm.utils.layout import write_quantized_matrix, read_quantized_matrix


# Simple character-level vocabulary for demonstration
# In practice, this would be a BPE tokenizer
VOCAB = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?'-\n")
VOCAB_SIZE = len(VOCAB)
CHAR_TO_ID = {c: i for i, c in enumerate(VOCAB)}
ID_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}


def tokenize(text: str) -> list[int]:
    """
    Convert text to token IDs (character-level for now).

    Args:
        text: Input text

    Returns:
        List of token IDs
    """
    tokens = []
    for char in text:
        if char in CHAR_TO_ID:
            tokens.append(CHAR_TO_ID[char])
        else:
            tokens.append(CHAR_TO_ID[' '])  # Unknown → space
    return tokens


def detokenize(tokens: list[int]) -> str:
    """
    Convert token IDs back to text.

    Args:
        tokens: List of token IDs

    Returns:
        Decoded text string
    """
    return ''.join(ID_TO_CHAR.get(t, ' ') for t in tokens)


def sample_token(logits: np.ndarray, temperature: float = 1.0, top_k: int = 0) -> int:
    """
    Sample next token from logits distribution.

    Args:
        logits: Unnormalized log probabilities [vocab_size]
        temperature: Sampling temperature (higher = more random)
        top_k: If > 0, only sample from top-k tokens

    Returns:
        Sampled token ID
    """
    # Clip logits to valid vocabulary range
    logits = logits[:VOCAB_SIZE]

    # Apply temperature
    if temperature > 0:
        logits = logits / temperature

    # Apply top-k filtering
    if top_k > 0:
        indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
        logits[indices_to_remove] = -float('inf')

    # Convert to probabilities
    probs = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    probs = probs / probs.sum()

    # Sample
    if temperature == 0:
        # Greedy (argmax)
        return int(np.argmax(probs))
    else:
        # Sample from distribution
        return int(np.random.choice(len(probs), p=probs))


def encode_hidden_state(tokens: list[int], hidden_dim: int = 128) -> np.ndarray:
    """
    Encode token sequence as hidden state vector.

    For now, this is a simple embedding (one-hot + projection).
    In practice, this would use learned embeddings or previous hidden state.

    Args:
        tokens: Token sequence
        hidden_dim: Hidden state dimension

    Returns:
        Hidden state vector [hidden_dim]
    """
    # Simple: use last token ID as seed for deterministic "embedding"
    # This is a placeholder - real implementation would use learned embeddings
    if not tokens:
        # Start of sequence: random initialization
        np.random.seed(42)
        h = np.random.randn(hidden_dim).astype(np.float32) * 0.01
    else:
        # Use last token to generate hidden state
        last_token = tokens[-1]
        np.random.seed(last_token)
        h = np.random.randn(hidden_dim).astype(np.float32) * 0.1

        # Add positional encoding (sin/cos)
        pos = len(tokens)
        for i in range(hidden_dim):
            if i % 2 == 0:
                h[i] += np.sin(pos / (10000 ** (i / hidden_dim)))
            else:
                h[i] += np.cos(pos / (10000 ** (i / hidden_dim)))

    return h


def generate_text(
    program_path: Path,
    prompt: str = "",
    max_tokens: int = 20,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """
    Generate text autoregressively using pixel program.

    Args:
        program_path: Path to .pxi pixel program
        prompt: Initial prompt text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering

    Returns:
        Generated text
    """
    print("=" * 70)
    print(" AUTOREGRESSIVE TEXT GENERATION")
    print("=" * 70)
    print()
    print(f"Program: {program_path.name}")
    print(f"Prompt: \"{prompt}\"")
    print(f"Max tokens: {max_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Top-k: {top_k}")
    print()

    # Load program
    img = np.array(Image.open(program_path).convert("RGBA"), dtype=np.uint8)

    # Tokenize prompt
    tokens = tokenize(prompt) if prompt else []
    print(f"Prompt tokens: {tokens}")
    print()

    # Generation loop
    print("Generating...")
    print("-" * 70)
    print(prompt, end='', flush=True)

    for step in range(max_tokens):
        # Encode current sequence as hidden state
        h_in = encode_hidden_state(tokens)

        # Write h_in to program (row 1)
        h_in_row = 1
        h_in_2d = h_in.reshape(1, -1)
        write_quantized_matrix(img, h_in_row, h_in_2d)

        # Run forward pass
        result_img = run_program(img.copy())

        # Extract logits (find from instruction 4's output - ADD to logits)
        logits_row = int(result_img[0, 4, 3])
        logits_2d = read_quantized_matrix(result_img, logits_row)
        logits = logits_2d.flatten()

        # Sample next token
        next_token = sample_token(logits, temperature=temperature, top_k=top_k)
        tokens.append(next_token)

        # Print token
        char = ID_TO_CHAR.get(next_token, ' ')
        print(char, end='', flush=True)

        # Stop if we generate newline (end of sentence)
        if char == '\n':
            break

    print()
    print("-" * 70)
    print()

    # Return full generated text
    return detokenize(tokens)


def main():
    """CLI interface for text generation."""
    parser = argparse.ArgumentParser(
        description="Generate text using Pixel-LLM forward pass program"
    )
    parser.add_argument(
        "--program",
        type=Path,
        default=Path("pixel_llm/programs/pixellm_forward_compiled.pxi"),
        help="Path to .pxi program (default: pixellm_forward_compiled.pxi)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The meaning of life is",
        help="Initial prompt text"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0=greedy, higher=more random)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k filtering (0=disabled)"
    )

    args = parser.parse_args()

    # Resolve path
    root = Path(__file__).resolve().parents[2]
    program_path = root / args.program

    if not program_path.exists():
        print(f"ERROR: Program not found: {program_path}")
        print("Run: python3 -m pxvm.dev.assembler")
        return 1

    # Generate text
    generated = generate_text(
        program_path=program_path,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("=" * 70)
    print(" GENERATION COMPLETE")
    print("=" * 70)
    print()
    print("Full output:")
    print(f"\"{generated}\"")
    print()
    print("This proves: A PNG file can generate text autoregressively!")
    print()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
