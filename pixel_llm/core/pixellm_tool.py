#!/usr/bin/env python3
"""
pixellm_tool.py - Clean API for Pixel-LLM Integration

This provides a stable, simple API that other agents (including Claude Code Research)
can use to get predictions from Pixel-LLM.

Usage:
    from pixel_llm.core.pixellm_tool import run_pixellm, score_options

    # Get prediction for text
    result = run_pixellm("pxOS evolution system")

    # Score multiple options
    options = ["option_a", "option_b", "option_c"]
    scores = score_options(options)
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# Ensure pxOS root is in path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pixel_llm.core.pixel_model_loader import get_default_loader, PixelModelLoader
from pixel_llm.programs.pixellm_infer import simple_tokenize, pixellm_forward

# Global loader (lazy initialization)
_loader: PixelModelLoader = None


def _get_loader() -> PixelModelLoader:
    """Get or initialize the global Pixel-LLM loader."""
    global _loader
    if _loader is None:
        _loader = get_default_loader()
    return _loader


def run_pixellm(text: str, verbose: bool = False) -> Dict:
    """
    Run Pixel-LLM inference on text.

    Args:
        text: Input text
        verbose: If True, print detailed info

    Returns:
        dict with:
            - text: input text
            - num_tokens: number of tokens
            - top_token: predicted token ID
            - top_k_tokens: list of top 5 token IDs
            - top_k_probs: list of top 5 probabilities
            - logits_shape: shape of output logits
            - mean_logit: mean logit value
            - max_logit: max logit value
    """
    loader = _get_loader()

    # Tokenize and run forward pass
    tokens = simple_tokenize(text)
    logits = pixellm_forward(tokens, loader, verbose=verbose)

    # Get top-k predictions
    k = min(5, len(logits))
    top_k_indices = logits.argsort()[-k:][::-1]
    top_k_logits = logits[top_k_indices]

    # Compute probabilities (softmax)
    exp_logits = np.exp(top_k_logits - top_k_logits.max())
    top_k_probs = exp_logits / exp_logits.sum()

    return {
        "text": text,
        "num_tokens": int(len(tokens)),
        "top_token": int(logits.argmax()),
        "top_k_tokens": top_k_indices.tolist(),
        "top_k_probs": top_k_probs.tolist(),
        "logits_shape": tuple(logits.shape),
        "mean_logit": float(logits.mean()),
        "max_logit": float(logits.max()),
    }


def score_options(options: List[str], verbose: bool = False) -> List[Tuple[str, float]]:
    """
    Score multiple options using Pixel-LLM.

    Args:
        options: List of text options to score
        verbose: If True, print details

    Returns:
        List of (option, score) tuples, sorted by score descending
    """
    results = []

    for option in options:
        result = run_pixellm(option, verbose=verbose)
        # Use max logit as score (simple heuristic)
        score = result["max_logit"]
        results.append((option, score))

    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def get_embedding(text: str) -> np.ndarray:
    """
    Get the hidden representation of text (after pooling, before output).

    This can be used for similarity comparisons.

    Args:
        text: Input text

    Returns:
        numpy array [D] of hidden activations
    """
    loader = _get_loader()

    # Load weights
    W_embed = loader.load_tensor("embed")
    W_hidden = loader.load_tensor("hidden")
    b_hidden = loader.load_tensor("b_hidden")

    # Tokenize
    tokens = simple_tokenize(text)

    # Embedding + pooling
    x = W_embed[tokens].mean(axis=0)

    # Hidden layer (this is the embedding)
    h = x @ W_hidden + b_hidden
    h = np.maximum(0, h)  # ReLU

    return h


def compare_similarity(text_a: str, text_b: str) -> float:
    """
    Compare similarity of two texts using Pixel-LLM embeddings.

    Args:
        text_a: First text
        text_b: Second text

    Returns:
        Cosine similarity [-1, 1]
    """
    emb_a = get_embedding(text_a)
    emb_b = get_embedding(text_b)

    # Cosine similarity
    dot = np.dot(emb_a, emb_b)
    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


def main():
    """Test the Pixel-LLM tool API."""
    print("="*60)
    print("PIXEL-LLM TOOL API TEST")
    print("="*60)

    # Test 1: Basic inference
    print("\n1. Basic inference:")
    result = run_pixellm("pxOS evolution system")
    print(f"   Text: '{result['text']}'")
    print(f"   Tokens: {result['num_tokens']}")
    print(f"   Top token: {result['top_token']}")
    print(f"   Max logit: {result['max_logit']:.4f}")

    # Test 2: Score options
    print("\n2. Scoring options:")
    options = [
        "simple architecture",
        "complex architecture",
        "modular design",
    ]
    scores = score_options(options)
    for option, score in scores:
        print(f"   {score:+.4f} - {option}")

    # Test 3: Similarity
    print("\n3. Text similarity:")
    sim = compare_similarity("pxOS hypervisor", "cartridge manager")
    print(f"   Similarity: {sim:.4f}")

    print("\n" + "="*60)
    print("✅ Pixel-LLM tool API ready for use!")
    print("="*60)
    print()


if __name__ == "__main__":
    main()
