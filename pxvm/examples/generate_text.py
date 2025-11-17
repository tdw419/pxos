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
from pxvm.visual.text_render import FontAtlas, create_text_image


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


def encode_hidden_state(
    tokens: list[int],
    hidden_dim: int = 128,
    W_embed: np.ndarray = None
) -> np.ndarray:
    """
    Encode token sequence as hidden state vector.

    Args:
        tokens: Token sequence
        hidden_dim: Hidden state dimension
        W_embed: Optional embedding matrix [vocab_size, hidden_dim].
                 If provided, uses learned embeddings. Otherwise uses random.

    Returns:
        Hidden state vector [hidden_dim]
    """
    if W_embed is not None:
        # Use learned embeddings
        if not tokens:
            # Start of sequence: return zero vector or BOS embedding
            # For now, use zero vector
            h = np.zeros(hidden_dim, dtype=np.float32)
        else:
            # Look up last token in embedding matrix
            last_token = tokens[-1]
            h = W_embed[last_token].copy()

            # Add positional encoding (helps model distinguish positions)
            pos = len(tokens)
            for i in range(hidden_dim):
                if i % 2 == 0:
                    h[i] += np.sin(pos / (10000 ** (i / hidden_dim)))
                else:
                    h[i] += np.cos(pos / (10000 ** (i / hidden_dim)))
    else:
        # Fallback: use random embeddings (for untrained models)
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
    weights_path: Path = None,
    output_image: Path = None,
) -> str:
    """
    Generate text autoregressively using pixel program.

    Args:
        program_path: Path to .pxi pixel program
        prompt: Initial prompt text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        weights_path: Optional path to .npz weights file for embeddings
        output_image: Optional path to save visual rendering of generated text

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

    # Load embeddings if weights provided
    W_embed = None
    if weights_path is not None:
        print(f"Loading embeddings: {weights_path.name}")
        weights = np.load(weights_path)
        if 'embed' in weights:
            W_embed = weights['embed']
            print(f"  Embedding matrix: {W_embed.shape}")
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
        h_in = encode_hidden_state(tokens, W_embed=W_embed)

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

    # Get full generated text
    generated_text = detokenize(tokens)

    # Render visual output if requested
    if output_image is not None:
        print("Rendering visual output...")

        # Load font atlas
        root = Path(__file__).resolve().parents[2]
        font_png = root / "fonts" / "ascii_16x16.png"
        font_json = root / "fonts" / "ascii_16x16.json"

        if not font_png.exists():
            print(f"WARNING: Font atlas not found at {font_png}")
            print("Run: python3 -m pxvm.visual.font_atlas")
            print("Skipping visual output.")
        else:
            # Load font
            font = FontAtlas(font_png, font_json)

            # Create image with rendered text
            visual_img = create_text_image(
                generated_text,
                font,
                padding=20,
                background=(20, 20, 30, 255),
                text_color=(200, 220, 255)
            )

            # Save
            Image.fromarray(visual_img).save(output_image)
            print(f"  Saved visual output: {output_image}")
            print(f"  Image size: {visual_img.shape[1]}×{visual_img.shape[0]} RGBA")
            print()

    # Return full generated text
    return generated_text


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
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to .npz weights file for embeddings (optional)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save visual output as PNG (optional)"
    )

    args = parser.parse_args()

    # Resolve paths
    root = Path(__file__).resolve().parents[2]
    program_path = root / args.program

    if not program_path.exists():
        print(f"ERROR: Program not found: {program_path}")
        print("Run: python3 -m pxvm.dev.assembler")
        return 1

    weights_path = None
    if args.weights is not None:
        weights_path = root / args.weights
        if not weights_path.exists():
            print(f"ERROR: Weights not found: {weights_path}")
            return 1

    output_image = None
    if args.output is not None:
        output_image = root / args.output
        # Create parent directory if needed
        output_image.parent.mkdir(parents=True, exist_ok=True)

    # Generate text
    generated = generate_text(
        program_path=program_path,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        weights_path=weights_path,
        output_image=output_image,
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
