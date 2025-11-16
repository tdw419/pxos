#!/usr/bin/env python3
"""
pixellm_infer_pure.py - Pure Pixel Inference

Run Pixel-LLM inference where input and output are pixel files.

Input:  input_tokens.pxi  (pixel image encoding tokens)
Output: output_token.pxi  (pixel image encoding prediction)
        logits_heatmap.pxi (visual heatmap of all logits)

This makes Pixel-LLM feel pixel-native even though computation
is still numpy internally.

Usage:
    # From text (creates pixel input file)
    python3 pixel_llm/programs/pixellm_infer_pure.py "pxOS evolution"

    # From pixel file
    python3 pixel_llm/programs/pixellm_infer_pure.py --input input_tokens.pxi

    # Output files created:
    #   output_token.pxi - Prediction as pixels
    #   logits_heatmap.pxi - Visual logits heatmap
"""

import sys
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pixel_llm.core.pixel_model_loader import get_default_loader
from pixel_llm.core.pixel_io import (
    pixel_encode_tokens,
    pixel_decode_tokens,
    pixel_encode_prediction,
    pixel_decode_prediction,
    pixel_encode_logits_heatmap,
    simple_detokenize,
)
from pixel_llm.programs.pixellm_infer import pixellm_forward

# Output paths
OUTPUT_DIR = ROOT / "pixel_llm" / "outputs"
OUTPUT_TOKEN_PATH = OUTPUT_DIR / "output_token.pxi"
OUTPUT_HEATMAP_PATH = OUTPUT_DIR / "logits_heatmap.pxi"
INPUT_TOKENS_PATH = OUTPUT_DIR / "input_tokens.pxi"


def run_pure_pixel_inference(
    input_text: str = None,
    input_pixel_path: Path = None,
    verbose: bool = False
) -> dict:
    """
    Run pure pixel inference.

    Args:
        input_text: Text to encode as pixels (if no input_pixel_path)
        input_pixel_path: Path to existing token pixel file
        verbose: Print detailed info

    Returns:
        dict with paths to output files
    """
    print("="*60)
    print("PIXEL-LLM - PURE PIXEL INFERENCE")
    print("="*60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Get input as pixels
    if input_pixel_path is not None:
        print(f"\n1. Loading input from pixel file...")
        print(f"   Path: {input_pixel_path}")
        token_ids = pixel_decode_tokens(input_pixel_path)
        text = simple_detokenize(token_ids)
    else:
        print(f"\n1. Encoding input text as pixels...")
        print(f"   Text: '{input_text}'")
        pixel_encode_tokens(input_text, INPUT_TOKENS_PATH)
        print(f"   Saved: {INPUT_TOKENS_PATH}")
        token_ids = pixel_decode_tokens(INPUT_TOKENS_PATH)
        text = input_text

    print(f"   Tokens: {len(token_ids)}")

    # Step 2: Load model weights from pixels
    print(f"\n2. Loading Pixel-LLM weights from pixels...")
    loader = get_default_loader()
    info = loader.get_model_info()
    print(f"   Model: {info['model_name']} v{info['version']}")
    print(f"   Parameters: {info['total_params']:,}")
    print(f"   ✅ Weights loaded from pixel image!")

    # Step 3: Run forward pass
    print(f"\n3. Running forward pass...")
    if verbose:
        print("   (Computation uses numpy - future: GPU shaders or pxVM)")

    logits = pixellm_forward(token_ids, loader, verbose=verbose)

    print(f"   ✅ Inference complete!")
    print(f"   Output: {len(logits)} logits")

    # Step 4: Encode output as pixels
    print(f"\n4. Encoding output as pixels...")

    # Prediction pixel
    pixel_encode_prediction(logits, OUTPUT_TOKEN_PATH, top_k=5)
    print(f"   Saved: {OUTPUT_TOKEN_PATH}")

    # Logits heatmap
    pixel_encode_logits_heatmap(logits, OUTPUT_HEATMAP_PATH, width=32)
    print(f"   Saved: {OUTPUT_HEATMAP_PATH}")

    # Step 5: Decode prediction from pixels
    print(f"\n5. Decoding prediction from pixels...")
    pred = pixel_decode_prediction(OUTPUT_TOKEN_PATH)

    print(f"   Top token ID: {pred['top_token']}")
    print(f"   Top probability: {pred['top_prob']:.4f}")

    # Try to convert to character
    top_char = chr(pred['top_token'] % 128) if pred['top_token'] < 128 else '?'
    print(f"   Top character: '{top_char}'")

    # Show top 5
    print(f"\n   Top 5 predictions:")
    for i, (token, prob) in enumerate(zip(pred['top_tokens'], pred['top_probs']), 1):
        char = chr(token % 128) if token < 128 else '?'
        print(f"      {i}. Token {token:4d} ('{char}') - prob: {prob:.4f}")

    print("\n" + "="*60)
    print("🎨 PURE PIXEL INFERENCE COMPLETE 🎨")
    print("="*60)
    print()
    print("Input:  Pixels (token image)")
    print("Weights: Pixels (weight image)")
    print("Output: Pixels (prediction image + heatmap)")
    print()
    print("Files created:")
    if input_text and input_pixel_path is None:
        print(f"  • {INPUT_TOKENS_PATH} (input tokens)")
    print(f"  • {OUTPUT_TOKEN_PATH} (prediction)")
    print(f"  • {OUTPUT_HEATMAP_PATH} (logits heatmap)")
    print("="*60)
    print()

    return {
        "input_path": str(input_pixel_path or INPUT_TOKENS_PATH),
        "output_token_path": str(OUTPUT_TOKEN_PATH),
        "output_heatmap_path": str(OUTPUT_HEATMAP_PATH),
        "prediction": pred,
        "input_text": text,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pixel-LLM Pure Pixel Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From text (creates pixel input file)
  python3 pixellm_infer_pure.py "pxOS evolution"

  # From existing pixel file
  python3 pixellm_infer_pure.py --input input_tokens.pxi

  # Verbose mode
  python3 pixellm_infer_pure.py "hypervisor" --verbose

Output files:
  pixel_llm/outputs/output_token.pxi - Prediction as pixels
  pixel_llm/outputs/logits_heatmap.pxi - Visual logits heatmap
        """
    )

    parser.add_argument(
        "text",
        nargs="?",
        help="Input text (if not using --input)"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        help="Input token pixel file"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Validate args
    if args.input is None and args.text is None:
        print("Error: Provide either text or --input file")
        print()
        parser.print_help()
        sys.exit(1)

    if args.input and not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Run inference
    try:
        result = run_pure_pixel_inference(
            input_text=args.text,
            input_pixel_path=args.input,
            verbose=args.verbose
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
