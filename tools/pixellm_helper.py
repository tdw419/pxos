#!/usr/bin/env python3
"""
Pixel-LLM Helper for Claude Code Research

This provides a clean API for using Pixel-LLM as a pixel coprocessor.
All interaction happens through pixel files - no numpy, no internal access.

Usage:
    from tools.pixellm_helper import ask_pixellm

    result = ask_pixellm("pxOS evolution")
    print(f"Top token: {result['top_token']}")
    print(f"Confidence: {result['top_prob']:.1%}")

Command line:
    python3 tools/pixellm_helper.py "text to query"
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict

# Add pxOS root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pixel_llm.core.pixel_io import pixel_encode_tokens, pixel_decode_prediction


def ask_pixellm(text: str, cleanup: bool = True, verbose: bool = False) -> Dict:
    """
    Ask Pixel-LLM a question using only pixels.

    This treats Pixel-LLM as a black-box pixel coprocessor:
    1. Text → pixel file (input_tokens.pxi)
    2. Invoke Pixel-LLM (opaque execution)
    3. Pixel file → result (output_token.pxi)

    Args:
        text: Input text to query
        cleanup: Remove temporary input file after use
        verbose: Print execution details

    Returns:
        dict with:
            - top_token: most likely token ID
            - top_prob: confidence (0.0-1.0)
            - top_tokens: list of top-5 token IDs
            - top_probs: list of top-5 probabilities

    Example:
        result = ask_pixellm("pxOS hypervisor")
        print(f"Pixel-LLM predicts token {result['top_token']} "
              f"with {result['top_prob']:.1%} confidence")
    """
    # Step 1: Create input pixels
    input_path = ROOT / "temp_pixellm_query.pxi"

    if verbose:
        print(f"[pixellm] Creating input pixels: {input_path}")

    pixel_encode_tokens(text, str(input_path))

    # Step 2: Invoke Pixel-LLM (black box)
    if verbose:
        print(f"[pixellm] Invoking Pixel-LLM coprocessor...")

    result = subprocess.run([
        sys.executable,
        str(ROOT / "pixel_llm" / "programs" / "pixellm_infer_pure.py"),
        "--input", str(input_path)
    ], capture_output=True, text=True, check=True)

    if verbose:
        print(f"[pixellm] Execution complete")

    # Step 3: Read output pixels
    output_path = ROOT / "pixel_llm" / "outputs" / "output_token.pxi"

    if not output_path.exists():
        raise FileNotFoundError(f"Pixel-LLM did not produce output: {output_path}")

    if verbose:
        print(f"[pixellm] Reading output pixels: {output_path}")

    prediction = pixel_decode_prediction(str(output_path))

    # Cleanup
    if cleanup:
        input_path.unlink(missing_ok=True)
        if verbose:
            print(f"[pixellm] Cleaned up temporary files")

    return prediction


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pixel-LLM Helper - Query Pixel-LLM using only pixels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tools/pixellm_helper.py "pxOS evolution"
  python3 tools/pixellm_helper.py "modular architecture" --verbose
  python3 tools/pixellm_helper.py "pixel substrate" --show-all
        """
    )

    parser.add_argument(
        "text",
        help="Text to query Pixel-LLM"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show execution details"
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all top-5 predictions"
    )

    args = parser.parse_args()

    # Query Pixel-LLM
    print(f"Query: '{args.text}'")
    print()

    result = ask_pixellm(args.text, verbose=args.verbose)

    # Show results
    print("Pixel-LLM Response:")
    print(f"  Top token: {result['top_token']}")
    print(f"  Confidence: {result['top_prob']:.1%}")

    if args.show_all:
        print()
        print("  Top 5 predictions:")
        for i, (token, prob) in enumerate(zip(result['top_tokens'], result['top_probs']), 1):
            marker = "→" if i == 1 else " "
            print(f"    {marker} {i}. Token {token:4d} ({prob:.1%})")

    print()
    print("✅ Query complete (pixel coprocessor API used)")


if __name__ == "__main__":
    main()
