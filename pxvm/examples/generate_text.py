#!/usr/bin/env python3
"""
pxvm/examples/generate_text.py

Autoregressive text generation using Pixel-LLM forward pass program.
"""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# This is a placeholder for the real interpreter.
# I'll need to create this module next.
from pxvm.core.interpreter import run_program
from pxvm.utils.layout import write_quantized_matrix, read_quantized_matrix
from pxvm.visual.text_render import FontAtlas, create_text_image

def tokenize(text: str, stoi: dict) -> list[int]:
    return [stoi[ch] for ch in text]

def detokenize(tokens: list[int], itos: dict) -> str:
    return "".join([itos[i] for i in tokens])

def encode_hidden_state(tokens: list[int], W_embed: np.ndarray) -> np.ndarray:
    if not tokens:
        return np.zeros((1, W_embed.shape[1]), dtype=np.float32)
    last_token = tokens[-1]
    return W_embed[last_token].reshape(1, -1)

def sample_logits(logits: np.ndarray, temperature: float, top_k: int) -> int:
    if temperature == 0:
        return np.argmax(logits)

    logits = logits / temperature
    if top_k > 0:
        top_k_indices = np.argsort(logits)[-top_k:]
        top_k_logits = logits[top_k_indices]
        probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits))
        return np.random.choice(top_k_indices, p=probs)
    else:
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return np.random.choice(len(logits), p=probs)

def generate_text(
    program_path: Path,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    weights_path: Path,
    output_image: Path | None,
) -> str:
    """
    Generate text autoregressively using pixel program.
    """
    print("--- Autoregressive Text Generation ---")
    print(f"Program: {program_path.name}")
    print(f"Prompt: \"{prompt}\"")
    print(f"Max tokens: {max_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Top-k: {top_k}")
    print()

    # Load weights for embeddings
    weights = np.load(weights_path)
    W_embed = weights['embed']

    # Create vocab
    with open("pixel_llm/corpus/quotes_and_stories.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    # Load program
    img = np.array(Image.open(program_path).convert("RGBA"), dtype=np.uint8)

    tokens = tokenize(prompt, stoi)

    for _ in range(max_tokens):
        h_in = encode_hidden_state(tokens, W_embed)

        # Write h_in to the program image
        write_quantized_matrix(img, row_start=1, data=h_in, image_width=img.shape[1])

        # Run the program
        result_img = run_program(img.copy())

        # Read the logits
        logits = read_quantized_matrix(result_img, row_start=87, image_width=img.shape[1]).flatten()

        next_token = sample_logits(logits, temperature, top_k)
        tokens.append(next_token)

    generated_text = detokenize(tokens, itos)

    if output_image:
        print("Rendering visual output...")
        root = Path(__file__).resolve().parents[2]
        font_png = root / "fonts" / "ascii_16x16.png"
        font_json = root / "fonts" / "ascii_16x16.json"

        if not font_png.exists() or not font_json.exists():
            print("Font atlas not found. Please run `python3 -m pxvm.visual.font_atlas`")
        else:
            font = FontAtlas(font_png, font_json)
            text_img = create_text_image(generated_text, font, padding=20, bg_color=(20, 20, 30))
            text_img.save(output_image)
            print(f"Visual output saved to {output_image}")

    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Generate text autoregressively from a pixel program.")
    parser.add_argument("--program", type=Path, required=True, help="Path to the .pxi pixel program.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to the .npz weights file for embeddings.")
    parser.add_argument("--prompt", type=str, default="The meaning of life is", help="Initial prompt text.")
    parser.add_argument("--max-tokens", type=int, default=40, help="Maximum tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0 for greedy).")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k filtering (0 for disabled).")
    parser.add_argument("--output", type=Path, help="Path to save the visual output as a PNG.")
    args = parser.parse_args()

    generated_text = generate_text(
        program_path=args.program,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        weights_path=args.weights,
        output_image=args.output,
    )

    print("\n--- Generated Text ---")
    print(generated_text)

if __name__ == "__main__":
    main()
